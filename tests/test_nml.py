"""
Tests for TensorLoom NML (.nml) declarative model specification.

Validates the complete NML pipeline:
  - Parsing of @model, @config, @layers, @forward blocks
  - Config → overridable __init__ kwargs
  - Layer config reference → self.X rewriting
  - Forward body → self.layer() auto-prefixing
  - Valid Python output
"""
import pytest
from tensorloom.lexer.lexer import Lexer
from tensorloom.parser.parser import Parser
from tensorloom.parser.ast_nodes import NMLModel
from tensorloom.codegen.pytorch_backend import PyTorchBackend


def compile_source(source: str) -> str:
    """Helper: lex + parse + codegen."""
    tokens = Lexer(source).tokenize()
    ast = Parser(tokens).parse()
    return PyTorchBackend().generate(ast)


def parse_source(source: str):
    """Helper: lex + parse only."""
    tokens = Lexer(source).tokenize()
    return Parser(tokens).parse()


# ── Minimal NML sources ──────────────────────────────────────

SIMPLE_NML = """@model SimpleNet:
    @config:
        hidden = 128

    @layers:
        fc = Linear(hidden, 10)

    @forward(x):
        return fc(x)
"""

TRANSFORMER_NML = """@model TransformerBlock:
    @config:
        d_model = 512
        n_heads = 8
        dropout_rate = 0.1

    @layers:
        attention = MultiHeadAttention(d_model, n_heads)
        norm1 = LayerNorm(d_model)
        norm2 = LayerNorm(d_model)
        ff = Linear(d_model, d_model)
        dropout = Dropout(dropout_rate)

    @forward(x):
        let residual = x
        x = norm1(x)
        x = residual + dropout(attention(x))
        let residual2 = x
        x = norm2(x)
        x = residual2 + dropout(ff(x))
        return x
"""

NO_CONFIG_NML = """@model Bare:
    @layers:
        fc = Linear(10, 5)

    @forward(x):
        return fc(x)
"""

MULTI_PARAM_FORWARD = """@model MultiInput:
    @config:
        dim = 64

    @layers:
        proj = Linear(dim, dim)

    @forward(x, mask):
        let out = proj(x)
        return out
"""


# ── Parsing Tests ─────────────────────────────────────────────

class TestNMLParsing:
    def test_nml_model_parsed(self):
        ast = parse_source(SIMPLE_NML)
        models = [s for s in ast.statements if isinstance(s, NMLModel)]
        assert len(models) == 1

    def test_nml_model_name(self):
        ast = parse_source(SIMPLE_NML)
        model = [s for s in ast.statements if isinstance(s, NMLModel)][0]
        assert model.name == "SimpleNet"

    def test_config_parsed(self):
        ast = parse_source(SIMPLE_NML)
        model = [s for s in ast.statements if isinstance(s, NMLModel)][0]
        assert "hidden" in model.config

    def test_layers_parsed(self):
        ast = parse_source(SIMPLE_NML)
        model = [s for s in ast.statements if isinstance(s, NMLModel)][0]
        assert len(model.layers) == 1
        assert model.layers[0].name == "fc"

    def test_forward_params(self):
        ast = parse_source(SIMPLE_NML)
        model = [s for s in ast.statements if isinstance(s, NMLModel)][0]
        assert model.forward_params == ["x"]

    def test_forward_body_parsed(self):
        ast = parse_source(SIMPLE_NML)
        model = [s for s in ast.statements if isinstance(s, NMLModel)][0]
        assert len(model.forward_body) > 0

    def test_transformer_multi_layers(self):
        ast = parse_source(TRANSFORMER_NML)
        model = [s for s in ast.statements if isinstance(s, NMLModel)][0]
        assert len(model.layers) == 5

    def test_transformer_multi_config(self):
        ast = parse_source(TRANSFORMER_NML)
        model = [s for s in ast.statements if isinstance(s, NMLModel)][0]
        assert len(model.config) == 3

    def test_no_config_model(self):
        ast = parse_source(NO_CONFIG_NML)
        model = [s for s in ast.statements if isinstance(s, NMLModel)][0]
        assert len(model.config) == 0

    def test_multi_param_forward(self):
        ast = parse_source(MULTI_PARAM_FORWARD)
        model = [s for s in ast.statements if isinstance(s, NMLModel)][0]
        assert model.forward_params == ["x", "mask"]


# ── Config → __init__ Tests ──────────────────────────────────

class TestNMLConfig:
    def test_init_with_config_kwargs(self):
        code = compile_source(SIMPLE_NML)
        assert "def __init__(self, hidden=128):" in code

    def test_init_stores_config(self):
        code = compile_source(SIMPLE_NML)
        assert "self.hidden = hidden" in code

    def test_transformer_init_kwargs(self):
        code = compile_source(TRANSFORMER_NML)
        assert "d_model=512" in code
        assert "n_heads=8" in code
        assert "dropout_rate=0.1" in code

    def test_no_config_init(self):
        code = compile_source(NO_CONFIG_NML)
        assert "def __init__(self):" in code

    def test_super_called(self):
        code = compile_source(SIMPLE_NML)
        assert "super().__init__()" in code


# ── Layer → self.X Config Reference Tests ─────────────────────

class TestNMLLayers:
    def test_layer_uses_self_config(self):
        code = compile_source(SIMPLE_NML)
        assert "self.fc = nn.Linear(self.hidden, 10)" in code

    def test_transformer_layers_self_prefix(self):
        code = compile_source(TRANSFORMER_NML)
        assert "self.attention = nn.MultiheadAttention(self.d_model, self.n_heads)" in code
        assert "self.norm1 = nn.LayerNorm(self.d_model)" in code
        assert "self.ff = nn.Linear(self.d_model, self.d_model)" in code
        assert "self.dropout = nn.Dropout(self.dropout_rate)" in code


# ── Forward → self.layer() Auto-Prefix Tests ─────────────────

class TestNMLForward:
    def test_forward_signature(self):
        code = compile_source(SIMPLE_NML)
        assert "def forward(self, x):" in code

    def test_layer_call_prefixed(self):
        code = compile_source(SIMPLE_NML)
        assert "self.fc(x)" in code

    def test_transformer_forward_layers(self):
        code = compile_source(TRANSFORMER_NML)
        assert "self.norm1(x)" in code
        assert "self.norm2(x)" in code
        assert "self.dropout(" in code
        assert "self.attention(x)" in code
        assert "self.ff(x)" in code

    def test_forward_return(self):
        code = compile_source(TRANSFORMER_NML)
        assert "return x" in code

    def test_let_in_forward(self):
        code = compile_source(TRANSFORMER_NML)
        assert "residual = x" in code
        assert "residual2 = x" in code

    def test_binary_op_in_forward(self):
        code = compile_source(TRANSFORMER_NML)
        # residual + dropout(attention(x))
        assert "residual +" in code

    def test_multi_param_forward_sig(self):
        code = compile_source(MULTI_PARAM_FORWARD)
        assert "def forward(self, x, mask):" in code


# ── Class Structure Tests ─────────────────────────────────────

class TestNMLClassStructure:
    def test_extends_nn_module(self):
        code = compile_source(SIMPLE_NML)
        assert "class SimpleNet(nn.Module):" in code

    def test_docstring_generated(self):
        code = compile_source(SIMPLE_NML)
        assert "NML declarative model: SimpleNet" in code


# ── Valid Python Output ───────────────────────────────────────

class TestNMLValidPython:
    def test_simple_compiles(self):
        code = compile_source(SIMPLE_NML)
        compile(code, "<test>", "exec")

    def test_transformer_compiles(self):
        code = compile_source(TRANSFORMER_NML)
        compile(code, "<test>", "exec")

    def test_no_config_compiles(self):
        code = compile_source(NO_CONFIG_NML)
        compile(code, "<test>", "exec")

    def test_multi_param_compiles(self):
        code = compile_source(MULTI_PARAM_FORWARD)
        compile(code, "<test>", "exec")


# ── Regression: No Impact on .tl Code ─────────────────────────

class TestNMLRegression:
    def test_tl_model_still_works(self):
        source = """model Net:
    layer fc = Linear(10, 5)
    fn forward(self, x: Tensor) -> Tensor:
        return self.fc(x)
"""
        code = compile_source(source)
        assert "class Net(nn.Module):" in code

    def test_kernel_still_works(self):
        source = """@kernel def add(x_ptr, n, BLOCK: tl.constexpr):
    let pid = tl.program_id(axis=0)
"""
        code = compile_source(source)
        assert "@triton.jit" in code
