"""
TensorLoom Token Definitions.

Defines all token types for the TensorLoom language, the keyword map,
and the Token dataclass used throughout the compiler pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class TokenType(Enum):
    """Every token type recognized by the TensorLoom lexer."""

    # ── Literals ──────────────────────────────────────────────
    INTEGER = auto()
    FLOAT = auto()
    STRING = auto()
    FSTRING_START = auto()   # f"
    FSTRING_TEXT = auto()    # plain text inside f-string
    FSTRING_EXPR = auto()    # {expr} inside f-string
    FSTRING_END = auto()     # closing "

    # ── Identifiers ───────────────────────────────────────────
    IDENTIFIER = auto()

    # ── Keywords ──────────────────────────────────────────────
    LET = auto()
    MODEL = auto()
    LAYER = auto()
    FN = auto()
    RETURN = auto()
    IMPORT = auto()
    TRAIN = auto()
    ON = auto()
    IF = auto()
    ELSE = auto()
    ELIF = auto()
    FOR = auto()
    IN = auto()
    WHILE = auto()
    TRUE = auto()
    FALSE = auto()
    NONE = auto()
    SELF = auto()
    EVERY = auto()

    # ── Type Keywords ─────────────────────────────────────────
    TENSOR_TYPE = auto()     # Tensor (capital T — the type hint)

    # ── Operators ─────────────────────────────────────────────
    PLUS = auto()            # +
    MINUS = auto()           # -
    STAR = auto()            # *
    SLASH = auto()           # /
    PERCENT = auto()         # %
    DOUBLE_STAR = auto()     # **
    AT = auto()              # @  (matmul)
    PIPE_ARROW = auto()      # |> (pipe / chain)
    ARROW = auto()           # -> (return type)
    EQUALS = auto()          # =  (assignment)
    DOUBLE_EQUALS = auto()   # ==
    NOT_EQUALS = auto()      # !=
    LESS = auto()            # <
    GREATER = auto()         # >
    LESS_EQUAL = auto()      # <=
    GREATER_EQUAL = auto()   # >=
    AND = auto()             # and
    OR = auto()              # or
    NOT = auto()             # not
    DOT = auto()             # .

    # ── Delimiters ────────────────────────────────────────────
    LPAREN = auto()          # (
    RPAREN = auto()          # )
    LBRACKET = auto()        # [
    RBRACKET = auto()        # ]
    LBRACE = auto()          # {
    RBRACE = auto()          # }
    COMMA = auto()           # ,
    COLON = auto()           # :
    SEMICOLON = auto()       # ;

    # ── Structural ────────────────────────────────────────────
    NEWLINE = auto()
    INDENT = auto()
    DEDENT = auto()

    # ── NML Decorators ────────────────────────────────────────
    AT_DECORATOR = auto()    # @model, @config, @layers, @forward

    # ── Special ───────────────────────────────────────────────
    EOF = auto()


# ── Keyword Lookup Table ──────────────────────────────────────
KEYWORDS: dict[str, TokenType] = {
    "let":    TokenType.LET,
    "model":  TokenType.MODEL,
    "layer":  TokenType.LAYER,
    "fn":     TokenType.FN,
    "return": TokenType.RETURN,
    "import": TokenType.IMPORT,
    "train":  TokenType.TRAIN,
    "on":     TokenType.ON,
    "if":     TokenType.IF,
    "else":   TokenType.ELSE,
    "elif":   TokenType.ELIF,
    "for":    TokenType.FOR,
    "in":     TokenType.IN,
    "while":  TokenType.WHILE,
    "true":   TokenType.TRUE,
    "false":  TokenType.FALSE,
    "none":   TokenType.NONE,
    "self":   TokenType.SELF,
    "every":  TokenType.EVERY,
    "and":    TokenType.AND,
    "or":     TokenType.OR,
    "not":    TokenType.NOT,
    "Tensor": TokenType.TENSOR_TYPE,
}


@dataclass(frozen=True, slots=True)
class Token:
    """A single token produced by the TensorLoom lexer."""

    type: TokenType
    value: str
    line: int
    column: int

    def __repr__(self) -> str:
        if self.type in (TokenType.NEWLINE, TokenType.INDENT, TokenType.DEDENT, TokenType.EOF):
            return f"Token({self.type.name}, L{self.line})"
        return f"Token({self.type.name}, {self.value!r}, L{self.line}:{self.column})"
