"""
TensorLoom Lexer — Hand-written tokenizer with indentation tracking.

Converts raw TensorLoom source code into a stream of tokens.  Handles:
  - Indentation-based block detection (INDENT / DEDENT)
  - All operators including |> (pipe) and @ (matmul)
  - String literals and f-strings
  - Single-line (//) and multi-line (/* */) comments
  - Numeric literals (int, float)
"""
from __future__ import annotations

from tensorloom.lexer.tokens import Token, TokenType, KEYWORDS


class LexerError(Exception):
    """Raised when the lexer encounters an invalid character sequence."""

    def __init__(self, message: str, line: int, column: int) -> None:
        self.line = line
        self.column = column
        super().__init__(f"LexerError at L{line}:{column}: {message}")


class Lexer:
    """Tokenizes TensorLoom source code."""

    def __init__(self, source: str, filename: str = "<stdin>") -> None:
        self.source = source
        self.filename = filename
        self.pos = 0
        self.line = 1
        self.column = 1
        self.tokens: list[Token] = []

        # Indentation state
        self._indent_stack: list[int] = [0]  # tracks nesting depth
        self._at_line_start = True
        self._paren_depth = 0  # suppress NEWLINE inside () [] {}

    # ── Public API ────────────────────────────────────────────

    def tokenize(self) -> list[Token]:
        """Run the lexer and return the full token list."""
        while not self._at_end():
            self._scan_token()

        # Emit remaining DEDENTs at EOF
        while len(self._indent_stack) > 1:
            self._indent_stack.pop()
            self._emit(TokenType.DEDENT, "")

        self._emit(TokenType.EOF, "")
        return self.tokens

    # ── Core Scanner ──────────────────────────────────────────

    def _scan_token(self) -> None:
        # At the start of a logical line, handle indentation
        if self._at_line_start and self._paren_depth == 0:
            self._handle_indentation()
            self._at_line_start = False

        if self._at_end():
            return

        ch = self._peek()

        # Skip inline whitespace (not newlines)
        if ch in (" ", "\t") and not self._at_line_start:
            self._advance()
            return

        # Newlines
        if ch == "\n":
            self._advance()
            if self._paren_depth == 0:
                # Don't emit consecutive NEWLINEs
                if not self.tokens or self.tokens[-1].type != TokenType.NEWLINE:
                    self._emit(TokenType.NEWLINE, "\\n")
            self._at_line_start = True
            return

        if ch == "\r":
            self._advance()
            return

        # Comments
        if ch == "/" and self._peek_next() == "/":
            self._skip_line_comment()
            return
        if ch == "/" and self._peek_next() == "*":
            self._skip_block_comment()
            return

        # f-strings
        if ch == "f" and self._peek_next() == '"':
            self._read_fstring()
            return

        # Strings
        if ch == '"':
            self._read_string()
            return

        # Numbers
        if ch.isdigit() or (ch == "." and self._peek_next().isdigit()):
            self._read_number()
            return

        # Identifiers / keywords
        if ch.isalpha() or ch == "_":
            self._read_identifier()
            return

        # Multi-char operators
        if ch == "|" and self._peek_next() == ">":
            self._advance()
            self._advance()
            self._emit(TokenType.PIPE_ARROW, "|>")
            return

        if ch == "-" and self._peek_next() == ">":
            self._advance()
            self._advance()
            self._emit(TokenType.ARROW, "->")
            return

        if ch == "*" and self._peek_next() == "*":
            self._advance()
            self._advance()
            self._emit(TokenType.DOUBLE_STAR, "**")
            return

        if ch == "=" and self._peek_next() == "=":
            self._advance()
            self._advance()
            self._emit(TokenType.DOUBLE_EQUALS, "==")
            return

        if ch == "!" and self._peek_next() == "=":
            self._advance()
            self._advance()
            self._emit(TokenType.NOT_EQUALS, "!=")
            return

        if ch == "<" and self._peek_next() == "=":
            self._advance()
            self._advance()
            self._emit(TokenType.LESS_EQUAL, "<=")
            return

        if ch == ">" and self._peek_next() == "=":
            self._advance()
            self._advance()
            self._emit(TokenType.GREATER_EQUAL, ">=")
            return

        # Single-char operators and delimiters
        single_map: dict[str, TokenType] = {
            "+": TokenType.PLUS,
            "-": TokenType.MINUS,
            "*": TokenType.STAR,
            "/": TokenType.SLASH,
            "%": TokenType.PERCENT,
            "@": TokenType.AT,
            "<": TokenType.LESS,
            ">": TokenType.GREATER,
            "=": TokenType.EQUALS,
            ".": TokenType.DOT,
            ",": TokenType.COMMA,
            ":": TokenType.COLON,
            ";": TokenType.SEMICOLON,
        }

        if ch in single_map:
            self._advance()
            self._emit(single_map[ch], ch)
            return

        # Bracketing (tracked for NEWLINE suppression)
        if ch == "(":
            self._advance()
            self._paren_depth += 1
            self._emit(TokenType.LPAREN, "(")
            return
        if ch == ")":
            self._advance()
            self._paren_depth = max(0, self._paren_depth - 1)
            self._emit(TokenType.RPAREN, ")")
            return
        if ch == "[":
            self._advance()
            self._paren_depth += 1
            self._emit(TokenType.LBRACKET, "[")
            return
        if ch == "]":
            self._advance()
            self._paren_depth = max(0, self._paren_depth - 1)
            self._emit(TokenType.RBRACKET, "]")
            return
        if ch == "{":
            self._advance()
            self._paren_depth += 1
            self._emit(TokenType.LBRACE, "{")
            return
        if ch == "}":
            self._advance()
            self._paren_depth = max(0, self._paren_depth - 1)
            self._emit(TokenType.RBRACE, "}")
            return

        raise LexerError(f"Unexpected character: {ch!r}", self.line, self.column)

    # ── Indentation ───────────────────────────────────────────

    def _handle_indentation(self) -> None:
        """Measure leading whitespace and emit INDENT/DEDENT tokens."""
        indent = 0
        while not self._at_end() and self._peek() in (" ", "\t"):
            if self._peek() == "\t":
                indent += 4  # treat tab as 4 spaces
            else:
                indent += 1
            self._advance()

        # Skip blank lines and comment-only lines
        if self._at_end() or self._peek() == "\n" or self._peek() == "\r":
            return
        if self._peek() == "/" and self.pos + 1 < len(self.source) and self.source[self.pos + 1] == "/":
            return

        current_indent = self._indent_stack[-1]

        if indent > current_indent:
            self._indent_stack.append(indent)
            self._emit(TokenType.INDENT, "")
        elif indent < current_indent:
            while self._indent_stack and self._indent_stack[-1] > indent:
                self._indent_stack.pop()
                self._emit(TokenType.DEDENT, "")
            if self._indent_stack[-1] != indent:
                raise LexerError(
                    f"Inconsistent indentation: expected {self._indent_stack[-1]} spaces, got {indent}",
                    self.line, self.column,
                )

    # ── Reading Helpers ───────────────────────────────────────

    def _read_identifier(self) -> None:
        """Read an identifier or keyword."""
        start = self.pos
        while not self._at_end() and (self._peek().isalnum() or self._peek() == "_"):
            self._advance()
        text = self.source[start : self.pos]
        token_type = KEYWORDS.get(text, TokenType.IDENTIFIER)
        self._emit(token_type, text)

    def _read_number(self) -> None:
        """Read an integer or floating-point number."""
        start = self.pos
        is_float = False

        while not self._at_end() and self._peek().isdigit():
            self._advance()

        if not self._at_end() and self._peek() == "." and self._peek_next().isdigit():
            is_float = True
            self._advance()  # consume '.'
            while not self._at_end() and self._peek().isdigit():
                self._advance()

        # Scientific notation: 1e10, 1.5e-3
        if not self._at_end() and self._peek() in ("e", "E"):
            is_float = True
            self._advance()
            if not self._at_end() and self._peek() in ("+", "-"):
                self._advance()
            while not self._at_end() and self._peek().isdigit():
                self._advance()

        text = self.source[start : self.pos]
        self._emit(TokenType.FLOAT if is_float else TokenType.INTEGER, text)

    def _read_string(self) -> None:
        """Read a quoted string literal."""
        self._advance()  # consume opening "
        start = self.pos
        while not self._at_end() and self._peek() != '"':
            if self._peek() == "\\":
                self._advance()  # skip escape char
            self._advance()
        if self._at_end():
            raise LexerError("Unterminated string literal", self.line, self.column)
        text = self.source[start : self.pos]
        self._advance()  # consume closing "
        self._emit(TokenType.STRING, text)

    def _read_fstring(self) -> None:
        """Read an f-string: f"text {expr} more text"
        
        For Phase 1 we emit the entire f-string as a single STRING token.
        The code generator will handle f-string semantics.
        """
        self._advance()  # consume 'f'
        self._advance()  # consume opening '"'
        start = self.pos
        depth = 0
        while not self._at_end():
            ch = self._peek()
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            elif ch == '"' and depth == 0:
                break
            elif ch == "\\":
                self._advance()
            self._advance()
        if self._at_end():
            raise LexerError("Unterminated f-string", self.line, self.column)
        text = self.source[start : self.pos]
        self._advance()  # consume closing "
        self._emit(TokenType.STRING, f"f\"{text}\"")

    # ── Comments ──────────────────────────────────────────────

    def _skip_line_comment(self) -> None:
        """Skip // to end of line."""
        while not self._at_end() and self._peek() != "\n":
            self._advance()

    def _skip_block_comment(self) -> None:
        """Skip /* ... */ (non-nested)."""
        self._advance()  # /
        self._advance()  # *
        while not self._at_end():
            if self._peek() == "*" and self._peek_next() == "/":
                self._advance()  # *
                self._advance()  # /
                return
            if self._peek() == "\n":
                self.line += 1
                self.column = 0
            self._advance()
        raise LexerError("Unterminated block comment", self.line, self.column)

    # ── Low-Level Helpers ─────────────────────────────────────

    def _peek(self) -> str:
        if self.pos >= len(self.source):
            return "\0"
        return self.source[self.pos]

    def _peek_next(self) -> str:
        if self.pos + 1 >= len(self.source):
            return "\0"
        return self.source[self.pos + 1]

    def _advance(self) -> str:
        ch = self.source[self.pos]
        self.pos += 1
        if ch == "\n":
            self.line += 1
            self.column = 1
        else:
            self.column += 1
        return ch

    def _at_end(self) -> bool:
        return self.pos >= len(self.source)

    def _emit(self, token_type: TokenType, value: str) -> None:
        self.tokens.append(Token(token_type, value, self.line, self.column))
