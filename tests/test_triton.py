"""
Tests for TensorLoom Triton Kernel (@kernel) code generation.

Validates the complete @kernel pipeline:
  - Parsing of @kernel def with constexpr parameters
  - @triton.jit decorator emission
  - Kernel body transpilation (let → assignment)
  - Auto-generated launcher wrapper with grid calculation
  - Triton import injection
  - Valid Python output
"""
import pytest
from tensorloom.lexer.lexer import Lexer
from tensorloom.parser.parser import Parser
from tensorloom.parser.ast_nodes import KernelDef, KernelParam
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


# ── Minimal kernel sources ────────────────────────────────────

VECTOR_ADD = """@kernel def vector_add(x_ptr, y_ptr, z_ptr, num_elements, BLOCK_SIZE: tl.constexpr):
    let pid = tl.program_id(axis=0)
    let block_start = pid * BLOCK_SIZE
    let offsets = block_start + tl.arange(0, BLOCK_SIZE)
    let mask = offsets < num_elements
    let x = tl.load(x_ptr + offsets, mask=mask)
    let y = tl.load(y_ptr + offsets, mask=mask)
    let z = x + y
    tl.store(z_ptr + offsets, z, mask=mask)
"""

SCALE_KERNEL = """@kernel def scale(x_ptr, num_elements, factor, BLOCK_SIZE: constexpr):
    let pid = tl.program_id(axis=0)
    let offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    let mask = offsets < num_elements
    let x = tl.load(x_ptr + offsets, mask=mask)
    let result = x * factor
    tl.store(x_ptr + offsets, result, mask=mask)
"""


# ── Parsing Tests ─────────────────────────────────────────────

class TestKernelParsing:
    def test_kernel_is_parsed(self):
        ast = parse_source(VECTOR_ADD)
        kernels = [s for s in ast.statements if isinstance(s, KernelDef)]
        assert len(kernels) == 1

    def test_kernel_name(self):
        ast = parse_source(VECTOR_ADD)
        kernel = [s for s in ast.statements if isinstance(s, KernelDef)][0]
        assert kernel.name == "vector_add"

    def test_kernel_params_count(self):
        ast = parse_source(VECTOR_ADD)
        kernel = [s for s in ast.statements if isinstance(s, KernelDef)][0]
        assert len(kernel.params) == 5

    def test_constexpr_param(self):
        ast = parse_source(VECTOR_ADD)
        kernel = [s for s in ast.statements if isinstance(s, KernelDef)][0]
        block_param = kernel.params[-1]
        assert block_param.name == "BLOCK_SIZE"
        assert block_param.is_constexpr is True

    def test_non_constexpr_params(self):
        ast = parse_source(VECTOR_ADD)
        kernel = [s for s in ast.statements if isinstance(s, KernelDef)][0]
        for p in kernel.params[:-1]:
            assert p.is_constexpr is False

    def test_kernel_body_has_statements(self):
        ast = parse_source(VECTOR_ADD)
        kernel = [s for s in ast.statements if isinstance(s, KernelDef)][0]
        assert len(kernel.body) > 0

    def test_bare_constexpr_hint(self):
        """Test constexpr without tl. prefix."""
        ast = parse_source(SCALE_KERNEL)
        kernel = [s for s in ast.statements if isinstance(s, KernelDef)][0]
        block_param = kernel.params[-1]
        assert block_param.is_constexpr is True


# ── Import Tests ──────────────────────────────────────────────

class TestKernelImports:
    def test_triton_import(self):
        code = compile_source(VECTOR_ADD)
        assert "import triton" in code

    def test_triton_language_import(self):
        code = compile_source(VECTOR_ADD)
        assert "import triton.language as tl" in code

    def test_no_triton_without_kernel(self):
        """Standard code should NOT import triton."""
        source = """let x = 42
"""
        code = compile_source(source)
        assert "triton" not in code


# ── Kernel Emission Tests ─────────────────────────────────────

class TestKernelEmission:
    def test_triton_jit_decorator(self):
        code = compile_source(VECTOR_ADD)
        assert "@triton.jit" in code

    def test_kernel_function_def(self):
        code = compile_source(VECTOR_ADD)
        assert "def vector_add(" in code

    def test_constexpr_annotation(self):
        code = compile_source(VECTOR_ADD)
        assert "BLOCK_SIZE: tl.constexpr" in code

    def test_let_transpiled_to_assignment(self):
        code = compile_source(VECTOR_ADD)
        assert "pid = tl.program_id(axis=0)" in code

    def test_tl_load_preserved(self):
        code = compile_source(VECTOR_ADD)
        assert "tl.load(" in code

    def test_tl_store_preserved(self):
        code = compile_source(VECTOR_ADD)
        assert "tl.store(" in code

    def test_tl_arange_preserved(self):
        code = compile_source(VECTOR_ADD)
        assert "tl.arange(0, BLOCK_SIZE)" in code

    def test_pointer_arithmetic(self):
        code = compile_source(VECTOR_ADD)
        assert "(x_ptr + offsets)" in code


# ── Launcher Tests ────────────────────────────────────────────

class TestLauncherGeneration:
    def test_launcher_function_exists(self):
        code = compile_source(VECTOR_ADD)
        assert "def vector_add_launcher(" in code

    def test_launcher_ptr_to_tensor(self):
        """Launcher should accept tensor names, not _ptr names."""
        code = compile_source(VECTOR_ADD)
        assert "def vector_add_launcher(x, y, z, num_elements" in code

    def test_launcher_default_block_size(self):
        code = compile_source(VECTOR_ADD)
        assert "BLOCK_SIZE=1024" in code

    def test_grid_with_cdiv(self):
        code = compile_source(VECTOR_ADD)
        assert "triton.cdiv(num_elements, meta['BLOCK_SIZE'])" in code

    def test_grid_lambda(self):
        code = compile_source(VECTOR_ADD)
        assert "grid = lambda meta:" in code

    def test_kernel_dispatch_with_grid(self):
        code = compile_source(VECTOR_ADD)
        assert "vector_add[grid](" in code

    def test_output_tensor_allocation(self):
        code = compile_source(VECTOR_ADD)
        assert "torch.empty_like(x)" in code

    def test_launcher_returns_output(self):
        code = compile_source(VECTOR_ADD)
        assert "return z" in code


# ── In-Place Kernel Tests ─────────────────────────────────────

class TestInPlaceKernel:
    def test_inplace_no_output_alloc(self):
        """Scale kernel modifies x_ptr in-place — no output allocation."""
        code = compile_source(SCALE_KERNEL)
        assert "torch.empty_like" not in code

    def test_inplace_returns_first_tensor(self):
        code = compile_source(SCALE_KERNEL)
        assert "return x" in code

    def test_inplace_launcher_name(self):
        code = compile_source(SCALE_KERNEL)
        assert "def scale_launcher(" in code


# ── Valid Python ──────────────────────────────────────────────

class TestKernelValidPython:
    def test_vector_add_compiles(self):
        code = compile_source(VECTOR_ADD)
        compile(code, "<test>", "exec")

    def test_scale_compiles(self):
        code = compile_source(SCALE_KERNEL)
        compile(code, "<test>", "exec")


# ── Regression: No Impact on Non-Kernel Code ──────────────────

class TestKernelRegression:
    def test_model_still_works(self):
        source = """model Net:
    layer fc = Linear(10, 5)
    fn forward(self, x: Tensor) -> Tensor:
        return self.fc(x)

let net = Net()
"""
        code = compile_source(source)
        assert "class Net(nn.Module):" in code
        assert "triton" not in code

    def test_existing_tests_unaffected(self):
        """Non-kernel code should produce identical output."""
        source = """let x = 42
"""
        code = compile_source(source)
        assert "x = 42" in code
        assert "@triton.jit" not in code
