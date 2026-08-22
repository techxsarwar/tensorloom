"""
Tests for the TensorLoom Lexer.
"""
import pytest
from tensorloom.lexer.lexer import Lexer, LexerError
from tensorloom.lexer.tokens import TokenType


class TestBasicTokens:
    """Test tokenization of basic language constructs."""

    def test_empty_source(self):
        tokens = Lexer("").tokenize()
        assert tokens[-1].type == TokenType.EOF

    def test_single_let(self):
        tokens = Lexer("let x = 42\n").tokenize()
        types = [t.type for t in tokens]
        assert TokenType.LET in types
        assert TokenType.IDENTIFIER in types
        assert TokenType.EQUALS in types
        assert TokenType.INTEGER in types

    def test_float_literal(self):
        tokens = Lexer("let y = 3.14\n").tokenize()
        float_tokens = [t for t in tokens if t.type == TokenType.FLOAT]
        assert len(float_tokens) == 1
        assert float_tokens[0].value == "3.14"

    def test_string_literal(self):
        tokens = Lexer('let s = "hello"\n').tokenize()
        str_tokens = [t for t in tokens if t.type == TokenType.STRING]
        assert len(str_tokens) == 1
        assert str_tokens[0].value == "hello"

    def test_boolean_tokens(self):
        tokens = Lexer("let a = true\nlet b = false\n").tokenize()
        types = [t.type for t in tokens]
        assert TokenType.TRUE in types
        assert TokenType.FALSE in types


class TestOperators:
    """Test tokenization of all operators."""

    def test_pipe_arrow(self):
        tokens = Lexer("x |> relu\n").tokenize()
        types = [t.type for t in tokens]
        assert TokenType.PIPE_ARROW in types

    def test_matmul(self):
        tokens = Lexer("x @ y\n").tokenize()
        types = [t.type for t in tokens]
        assert TokenType.AT in types

    def test_arrow(self):
        tokens = Lexer("-> Tensor\n").tokenize()
        types = [t.type for t in tokens]
        assert TokenType.ARROW in types

    def test_comparison_operators(self):
        tokens = Lexer("a == b != c <= d >= e < f > g\n").tokenize()
        types = [t.type for t in tokens]
        assert TokenType.DOUBLE_EQUALS in types
        assert TokenType.NOT_EQUALS in types
        assert TokenType.LESS_EQUAL in types
        assert TokenType.GREATER_EQUAL in types
        assert TokenType.LESS in types
        assert TokenType.GREATER in types

    def test_arithmetic(self):
        tokens = Lexer("a + b - c * d / e ** f\n").tokenize()
        types = [t.type for t in tokens]
        assert TokenType.PLUS in types
        assert TokenType.MINUS in types
        assert TokenType.STAR in types
        assert TokenType.SLASH in types
        assert TokenType.DOUBLE_STAR in types


class TestKeywords:
    """Test that all keywords are recognized."""

    @pytest.mark.parametrize("keyword,expected", [
        ("let", TokenType.LET),
        ("model", TokenType.MODEL),
        ("layer", TokenType.LAYER),
        ("fn", TokenType.FN),
        ("return", TokenType.RETURN),
        ("import", TokenType.IMPORT),
        ("train", TokenType.TRAIN),
        ("on", TokenType.ON),
        ("if", TokenType.IF),
        ("else", TokenType.ELSE),
        ("for", TokenType.FOR),
        ("in", TokenType.IN),
        ("self", TokenType.SELF),
        ("every", TokenType.EVERY),
    ])
    def test_keyword(self, keyword, expected):
        tokens = Lexer(f"{keyword}\n").tokenize()
        assert tokens[0].type == expected


class TestIndentation:
    """Test INDENT/DEDENT token generation."""

    def test_simple_indent(self):
        source = "model Foo:\n    layer x = Linear(10, 5)\n"
        tokens = Lexer(source).tokenize()
        types = [t.type for t in tokens]
        assert TokenType.INDENT in types
        assert TokenType.DEDENT in types

    def test_nested_indent(self):
        source = "if true:\n    if true:\n        let x = 1\n"
        tokens = Lexer(source).tokenize()
        types = [t.type for t in tokens]
        indent_count = types.count(TokenType.INDENT)
        dedent_count = types.count(TokenType.DEDENT)
        assert indent_count == 2
        assert dedent_count == 2


class TestComments:
    """Test comment handling."""

    def test_single_line_comment(self):
        source = "// this is a comment\nlet x = 1\n"
        tokens = Lexer(source).tokenize()
        types = [t.type for t in tokens]
        # Comments should be skipped entirely
        assert TokenType.LET in types

    def test_inline_comment_after_code(self):
        source = "let x = 1\n// trailing comment\n"
        tokens = Lexer(source).tokenize()
        types = [t.type for t in tokens]
        assert TokenType.LET in types

    def test_block_comment(self):
        source = "/* multi\nline\ncomment */\nlet x = 1\n"
        tokens = Lexer(source).tokenize()
        types = [t.type for t in tokens]
        assert TokenType.LET in types


class TestFStrings:
    """Test f-string tokenization."""

    def test_fstring(self):
        source = 'let s = f"value={x}"\n'
        tokens = Lexer(source).tokenize()
        str_tokens = [t for t in tokens if t.type == TokenType.STRING]
        assert len(str_tokens) == 1
        assert "f\"" in str_tokens[0].value


class TestNewlineSuppression:
    """Test that newlines inside brackets are suppressed."""

    def test_no_newline_in_parens(self):
        source = "func(\n    a,\n    b\n)\n"
        tokens = Lexer(source).tokenize()
        # NEWLINEs between ( and ) should be suppressed
        inside_parens = False
        for tok in tokens:
            if tok.type == TokenType.LPAREN:
                inside_parens = True
            elif tok.type == TokenType.RPAREN:
                inside_parens = False
            elif tok.type == TokenType.NEWLINE and inside_parens:
                pytest.fail("NEWLINE found inside parentheses")


class TestCompleteProgram:
    """Test tokenization of complete TensorLoom programs."""

    def test_hello_program(self):
        source = """let x = tensor([1.0, 2.0, 3.0], dtype=float32, device=gpu)
let z = x @ y + 0.5
print(z)
"""
        tokens = Lexer(source).tokenize()
        assert tokens[-1].type == TokenType.EOF
        # Should have no lexer errors (implicit — no exception raised)

    def test_model_definition(self):
        source = """model Net:
    layer fc1 = Linear(784, 256)
    fn forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x) |> relu
        return x
"""
        tokens = Lexer(source).tokenize()
        types = [t.type for t in tokens]
        assert TokenType.MODEL in types
        assert TokenType.LAYER in types
        assert TokenType.FN in types
        assert TokenType.PIPE_ARROW in types
        assert TokenType.ARROW in types
        assert TokenType.TENSOR_TYPE in types
