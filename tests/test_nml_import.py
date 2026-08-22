"""
Tests for TensorLoom Phase 6: Cross-File .nml Import System.

Validates the complete cross-file compilation pipeline:
  - Parsing 'import path.nml as Alias' syntax
  - NML file detection (is_nml flag)
  - Alias resolution and class renaming
  - Sub-compilation: .nml -> injected PyTorch class in .tl output
  - Config kwarg preservation across files
  - Model name registration for train block compatibility
"""
import os
import tempfile
import pytest
from tensorloom.lexer.lexer import Lexer
from tensorloom.parser.parser import Parser
from tensorloom.parser.ast_nodes import ImportStatement, NMLModel
from tensorloom.codegen.pytorch_backend import PyTorchBackend


def parse_source(source: str):
    """Helper: lex + parse."""
    tokens = Lexer(source).tokenize()
    return Parser(tokens).parse()


def compile_source(source: str, source_dir: str = ".") -> str:
    """Helper: lex + parse + codegen with source_dir for NML resolution."""
    tokens = Lexer(source).tokenize()
    ast = Parser(tokens).parse()
    return PyTorchBackend(source_dir=source_dir).generate(ast)


# ── Shared NML source for tests ──────────────────────────────

SIMPLE_NML_SOURCE = """@model SimpleNet:
    @config:
        hidden = 64

    @layers:
        fc1 = Linear(hidden, 32)
        fc2 = Linear(32, 10)

    @forward(x):
        x = fc1(x)
        return fc2(x)
"""


# ── Import Parsing Tests ─────────────────────────────────────

class TestNMLImportParsing:
    def test_nml_import_detected(self):
        ast = parse_source("import network.nml as MyNet\n")
        imports = [s for s in ast.statements if isinstance(s, ImportStatement)]
        assert len(imports) == 1
        assert imports[0].is_nml is True

    def test_nml_import_alias(self):
        ast = parse_source("import network.nml as MyNet\n")
        imp = [s for s in ast.statements if isinstance(s, ImportStatement)][0]
        assert imp.alias == "MyNet"

    def test_nml_import_path(self):
        ast = parse_source("import network.nml as MyNet\n")
        imp = [s for s in ast.statements if isinstance(s, ImportStatement)][0]
        assert imp.module_path == ["network", "nml"]

    def test_regular_import_not_nml(self):
        ast = parse_source("import std.io\n")
        imp = [s for s in ast.statements if isinstance(s, ImportStatement)][0]
        assert imp.is_nml is False
        assert imp.alias is None

    def test_regular_import_with_as(self):
        ast = parse_source("import std.io as io\n")
        imp = [s for s in ast.statements if isinstance(s, ImportStatement)][0]
        assert imp.alias == "io"
        assert imp.is_nml is False

    def test_nml_import_preserves_line(self):
        ast = parse_source("import arch.nml as Block\n")
        imp = [s for s in ast.statements if isinstance(s, ImportStatement)][0]
        assert imp.line == 1

    def test_nml_subdirectory_import(self):
        ast = parse_source("import models.transformer.nml as T\n")
        imp = [s for s in ast.statements if isinstance(s, ImportStatement)][0]
        assert imp.module_path == ["models", "transformer", "nml"]
        assert imp.is_nml is True
        assert imp.alias == "T"


# ── Cross-File Sub-Compilation Tests ─────────────────────────

class TestCrossFileCompilation:
    @pytest.fixture(autouse=True)
    def setup_nml_file(self, tmp_path):
        """Write a .nml file to tmp for sub-compilation tests."""
        self.nml_path = tmp_path / "network.nml"
        self.nml_path.write_text(SIMPLE_NML_SOURCE, encoding="utf-8")
        self.source_dir = str(tmp_path)

    def test_class_injected(self):
        source = "import network.nml as MyNet\n"
        code = compile_source(source, self.source_dir)
        assert "class MyNet(nn.Module):" in code

    def test_alias_renames_class(self):
        source = "import network.nml as CustomBlock\n"
        code = compile_source(source, self.source_dir)
        assert "class CustomBlock(nn.Module):" in code
        assert "class SimpleNet" not in code

    def test_config_preserved(self):
        source = "import network.nml as MyNet\n"
        code = compile_source(source, self.source_dir)
        assert "hidden=64" in code

    def test_layers_emitted(self):
        source = "import network.nml as MyNet\n"
        code = compile_source(source, self.source_dir)
        assert "self.fc1 = nn.Linear(self.hidden, 32)" in code
        assert "self.fc2 = nn.Linear(32, 10)" in code

    def test_forward_emitted(self):
        source = "import network.nml as MyNet\n"
        code = compile_source(source, self.source_dir)
        assert "def forward(self, x):" in code
        assert "self.fc1(x)" in code
        assert "self.fc2(x)" in code

    def test_import_comment(self):
        source = "import network.nml as MyNet\n"
        code = compile_source(source, self.source_dir)
        assert "NML Import: network.nml as MyNet" in code

    def test_nml_docstring(self):
        source = "import network.nml as MyNet\n"
        code = compile_source(source, self.source_dir)
        assert "NML declarative model: MyNet" in code


# ── Model Registration Tests ─────────────────────────────────

class TestNMLImportRegistration:
    @pytest.fixture(autouse=True)
    def setup_nml_file(self, tmp_path):
        self.nml_path = tmp_path / "arch.nml"
        self.nml_path.write_text(SIMPLE_NML_SOURCE, encoding="utf-8")
        self.source_dir = str(tmp_path)

    def test_alias_registered_as_model(self):
        """NML alias should be usable in let net = Alias() instantiation."""
        source = """import arch.nml as MyArch
let net = MyArch(hidden=128)
"""
        code = compile_source(source, self.source_dir)
        # Should get .to(device) + torch.compile treatment
        assert "MyArch(hidden=128).to(device)" in code
        assert "torch.compile" in code

    def test_train_block_references_import(self):
        """Train block should work with NML-imported model."""
        source = """import arch.nml as Encoder
let net = Encoder()
let data = load_dataset()
train net on data:
    epochs = 3
    optimizer = SGD(lr=0.01)
    loss = CrossEntropy
"""
        code = compile_source(source, self.source_dir)
        assert "class Encoder(nn.Module):" in code
        assert "for epoch in range(3):" in code
        assert "optim.SGD" in code


# ── Error Handling Tests ──────────────────────────────────────

class TestNMLImportErrors:
    def test_missing_nml_file_comment(self):
        """When .nml file doesn't exist, should emit an error comment."""
        source = "import nonexistent.nml as Ghost\n"
        code = compile_source(source, source_dir="/fake/path")
        assert "[ERROR] NML file not found" in code


# ── Valid Python Tests ────────────────────────────────────────

class TestCrossFileValidPython:
    @pytest.fixture(autouse=True)
    def setup_nml_file(self, tmp_path):
        self.nml_path = tmp_path / "network.nml"
        self.nml_path.write_text(SIMPLE_NML_SOURCE, encoding="utf-8")
        self.source_dir = str(tmp_path)

    def test_simple_import_compiles(self):
        source = "import network.nml as MyNet\n"
        code = compile_source(source, self.source_dir)
        compile(code, "<test>", "exec")

    def test_import_with_instantiation_compiles(self):
        source = """import network.nml as MyNet
let net = MyNet(hidden=32)
"""
        code = compile_source(source, self.source_dir)
        compile(code, "<test>", "exec")

    def test_import_with_train_compiles(self):
        source = """import network.nml as MyNet
let net = MyNet()
let data = load_dataset()
train net on data:
    epochs = 1
    optimizer = Adam(lr=0.001)
    loss = CrossEntropy
"""
        code = compile_source(source, self.source_dir)
        compile(code, "<test>", "exec")


# ── Regression Tests ──────────────────────────────────────────

class TestCrossFileRegression:
    def test_regular_imports_unaffected(self):
        code = compile_source("import std.io\n")
        assert "TensorLoom import: std.io" in code

    def test_nml_inline_still_works(self):
        source = """@model InlineNet:
    @config:
        dim = 16
    @layers:
        fc = Linear(dim, 8)
    @forward(x):
        return fc(x)
"""
        code = compile_source(source)
        assert "class InlineNet(nn.Module):" in code
        assert "dim=16" in code

    def test_existing_model_syntax_unaffected(self):
        source = """model Net:
    layer fc = Linear(10, 5)
    fn forward(self, x: Tensor) -> Tensor:
        return self.fc(x)
"""
        code = compile_source(source)
        assert "class Net(nn.Module):" in code

    def test_kernel_unaffected(self):
        source = """@kernel def scale(x_ptr, n, BLOCK: tl.constexpr):
    let pid = tl.program_id(axis=0)
"""
        code = compile_source(source)
        assert "@triton.jit" in code
