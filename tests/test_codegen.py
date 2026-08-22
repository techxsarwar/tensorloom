"""
Tests for the TensorLoom PyTorch Code Generator.
"""
import pytest
from tensorloom.lexer.lexer import Lexer
from tensorloom.parser.parser import Parser
from tensorloom.codegen.pytorch_backend import PyTorchBackend


def compile_tl(source: str) -> str:
    """Helper: compile TensorLoom source to Python code string."""
    tokens = Lexer(source).tokenize()
    ast = Parser(tokens).parse()
    backend = PyTorchBackend()
    return backend.generate(ast)


class TestBasicCodegen:
    """Test basic code generation."""

    def test_let_number(self):
        code = compile_tl("let x = 42\n")
        assert "x = 42" in code

    def test_let_tensor(self):
        code = compile_tl("let x = tensor([1.0, 2.0], dtype=float32, device=gpu)\n")
        assert "torch.tensor" in code
        assert "torch.float32" in code

    def test_print_statement(self):
        code = compile_tl("print(x)\n")
        assert "print(x)" in code

    def test_matmul(self):
        code = compile_tl("let z = x @ y\n")
        assert "@" in code

    def test_imports(self):
        code = compile_tl("let x = 1\n")
        assert "import torch" in code


class TestPipeOperator:
    """Test the pipe operator |> desugaring."""

    def test_single_pipe(self):
        code = compile_tl("let z = x |> relu\n")
        assert "torch.relu" in code

    def test_pipe_chain(self):
        code = compile_tl("let z = x |> relu |> sigmoid\n")
        # Should nest: sigmoid(relu(x))
        assert "torch.sigmoid" in code
        assert "torch.relu" in code

    def test_pipe_with_args(self):
        code = compile_tl("let z = x |> dropout(0.2)\n")
        assert "dropout" in code
        assert "0.2" in code

    def test_softmax_pipe(self):
        code = compile_tl("let z = x |> softmax\n")
        # softmax should get dim=-1 automatically
        assert "softmax" in code
        assert "dim=-1" in code


class TestModelCodegen:
    """Test model definition → nn.Module generation."""

    def test_model_class(self):
        source = """model Net:
    layer fc1 = Linear(784, 256)
    layer fc2 = Linear(256, 10)
    fn forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x) |> relu
        return x
"""
        code = compile_tl(source)
        assert "class Net(nn.Module):" in code
        assert "def __init__(self):" in code
        assert "super().__init__()" in code
        assert "self.fc1 = nn.Linear(784, 256)" in code
        assert "self.fc2 = nn.Linear(256, 10)" in code
        assert "def forward(self, x):" in code

    def test_model_with_pipe_in_forward(self):
        source = """model Net:
    layer fc1 = Linear(10, 5)
    fn forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x) |> relu |> dropout(0.5)
        return x
"""
        code = compile_tl(source)
        assert "torch.relu" in code
        assert "dropout" in code


class TestTrainBlockCodegen:
    """Test train block → training loop generation."""

    def test_basic_train(self):
        source = """model Net:
    layer fc = Linear(10, 5)
    fn forward(self, x: Tensor) -> Tensor:
        return self.fc(x)

let net = Net()

train net on data:
    epochs = 5
    optimizer = Adam(lr=0.001)
    loss = CrossEntropy
"""
        code = compile_tl(source)
        assert "optim.Adam" in code
        assert "nn.CrossEntropyLoss" in code
        assert "for epoch in range(5):" in code
        assert "optimizer.zero_grad()" in code
        assert "loss.backward()" in code

    def test_train_with_fp16(self):
        source = """model Net:
    layer fc = Linear(10, 5)
    fn forward(self, x: Tensor) -> Tensor:
        return self.fc(x)

let net = Net()

train net on data:
    epochs = 3
    optimizer = Adam(lr=0.001)
    loss = CrossEntropy
    precision = fp16
"""
        code = compile_tl(source)
        assert "autocast" in code or "GradScaler" in code
        assert "scaler" in code

    def test_train_with_checkpoint(self):
        source = """model Net:
    layer fc = Linear(10, 5)
    fn forward(self, x: Tensor) -> Tensor:
        return self.fc(x)

let net = Net()

train net on data:
    epochs = 10
    optimizer = Adam(lr=0.001)
    loss = CrossEntropy
    checkpoint every 2 epochs
"""
        code = compile_tl(source)
        assert "torch.save" in code
        assert "checkpoint" in code.lower()


class TestModelInstantiation:
    """Test that model instantiation adds .to(device) and torch.compile."""

    def test_model_to_device(self):
        source = """model Net:
    layer fc = Linear(10, 5)
    fn forward(self, x: Tensor) -> Tensor:
        return self.fc(x)

let net = Net()
"""
        code = compile_tl(source)
        assert ".to(device)" in code
        assert "torch.compile" in code


class TestDevicePlacement:
    """Test automatic device configuration."""

    def test_device_setup(self):
        code = compile_tl("let x = 1\n")
        assert 'torch.device("cuda"' in code
        assert "device" in code


class TestGeneratedCodeValidity:
    """Test that generated code is valid Python."""

    def test_compiles_as_python(self):
        source = """let x = tensor([1.0, 2.0, 3.0], dtype=float32, device=gpu)
let y = x + 1.0
print(y)
"""
        code = compile_tl(source)
        # Should compile without syntax errors
        compile(code, "<test>", "exec")

    def test_model_compiles(self):
        source = """model Net:
    layer fc = Linear(10, 5)
    fn forward(self, x: Tensor) -> Tensor:
        return self.fc(x)
"""
        code = compile_tl(source)
        compile(code, "<test>", "exec")
