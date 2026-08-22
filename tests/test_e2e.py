"""
End-to-end tests for TensorLoom.

Tests the full pipeline: .tl source → lex → parse → codegen → valid Python.
"""
import pytest
from tensorloom.lexer.lexer import Lexer
from tensorloom.parser.parser import Parser
from tensorloom.codegen.pytorch_backend import PyTorchBackend
from tensorloom.analyzer.type_checker import TypeChecker
from tensorloom.analyzer.optimizer import ASTOptimizer


def full_pipeline(source: str) -> tuple[str, list, list]:
    """Run the complete TensorLoom compilation pipeline.
    
    Returns: (generated_python, analysis_errors, analysis_warnings)
    """
    # Lex
    tokens = Lexer(source).tokenize()
    
    # Parse
    ast = Parser(tokens).parse()
    
    # Analyze
    checker = TypeChecker()
    errors, warnings = checker.analyze(ast)
    
    # Optimize
    optimizer = ASTOptimizer()
    report = optimizer.optimize(ast)
    
    # Generate
    backend = PyTorchBackend()
    code = backend.generate(ast)
    
    return code, errors, warnings


class TestHelloWorld:
    """Test the hello.tl example end-to-end."""

    HELLO_SOURCE = """let x = tensor([1.0, 2.0, 3.0], dtype=float32, device=gpu)
let y = tensor([4.0, 5.0, 6.0], dtype=float32, device=gpu)
let z = x @ y + 0.5
let scaled = z * 2.0
let activated = scaled |> relu
print(activated)
"""

    def test_compiles(self):
        code, errors, warnings = full_pipeline(self.HELLO_SOURCE)
        assert len(errors) == 0
        compile(code, "<hello>", "exec")  # Valid Python

    def test_has_torch_tensor(self):
        code, _, _ = full_pipeline(self.HELLO_SOURCE)
        assert "torch.tensor" in code

    def test_has_matmul(self):
        code, _, _ = full_pipeline(self.HELLO_SOURCE)
        assert "@" in code

    def test_has_relu(self):
        code, _, _ = full_pipeline(self.HELLO_SOURCE)
        assert "relu" in code


class TestMNISTModel:
    """Test a full MNIST classifier pipeline."""

    MNIST_SOURCE = """import nn

model MNISTClassifier:
    layer input  = Linear(784, 256)
    layer hidden = Linear(256, 128)
    layer output = Linear(128, 10)
    fn forward(self, x: Tensor) -> Tensor:
        x = self.input(x) |> relu |> dropout(0.2)
        x = self.hidden(x) |> relu |> dropout(0.2)
        x = self.output(x) |> softmax
        return x

let net = MNISTClassifier()
let data = load(batch_size=64, shuffle=true)

train net on data:
    epochs = 10
    optimizer = Adam(lr=0.001)
    loss = CrossEntropy
    precision = fp16
"""

    def test_compiles(self):
        code, errors, warnings = full_pipeline(self.MNIST_SOURCE)
        assert len(errors) == 0
        compile(code, "<mnist>", "exec")

    def test_model_class(self):
        code, _, _ = full_pipeline(self.MNIST_SOURCE)
        assert "class MNISTClassifier(nn.Module):" in code

    def test_layers(self):
        code, _, _ = full_pipeline(self.MNIST_SOURCE)
        assert "nn.Linear(784, 256)" in code
        assert "nn.Linear(256, 128)" in code
        assert "nn.Linear(128, 10)" in code

    def test_training_loop(self):
        code, _, _ = full_pipeline(self.MNIST_SOURCE)
        assert "for epoch in range(10):" in code

    def test_mixed_precision(self):
        code, _, _ = full_pipeline(self.MNIST_SOURCE)
        assert "autocast" in code or "GradScaler" in code

    def test_torch_compile(self):
        code, _, _ = full_pipeline(self.MNIST_SOURCE)
        assert "torch.compile" in code

    def test_pipe_desugaring(self):
        code, _, _ = full_pipeline(self.MNIST_SOURCE)
        # Pipes should be desugared to nested calls
        assert "torch.relu" in code
        assert "dropout" in code
        assert "softmax" in code


class TestOptimizationDetection:
    """Test that the optimizer detects fusion opportunities."""

    def test_pipe_fusion_detected(self):
        source = """model Net:
    layer fc = Linear(10, 5)
    fn forward(self, x: Tensor) -> Tensor:
        x = self.fc(x) |> relu |> dropout(0.2)
        return x
"""
        tokens = Lexer(source).tokenize()
        ast = Parser(tokens).parse()
        optimizer = ASTOptimizer()
        report = optimizer.optimize(ast)
        assert len(report.fusions) > 0

    def test_memory_estimation(self):
        source = """model Big:
    layer fc1 = Linear(1024, 512)
    layer fc2 = Linear(512, 256)
    fn forward(self, x: Tensor) -> Tensor:
        return x
"""
        tokens = Lexer(source).tokenize()
        ast = Parser(tokens).parse()
        optimizer = ASTOptimizer()
        report = optimizer.optimize(ast)
        assert len(report.memory_estimates) == 1
        est = report.memory_estimates[0]
        assert est.total_params > 0
        assert est.param_memory_mb > 0

    def test_precision_hint_detection(self):
        source = """train net on data:
    epochs = 5
    precision = fp16
"""
        tokens = Lexer(source).tokenize()
        ast = Parser(tokens).parse()
        optimizer = ASTOptimizer()
        report = optimizer.optimize(ast)
        assert "net" in report.precision_hints
        assert report.precision_hints["net"] == "fp16"


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_program(self):
        code, errors, warnings = full_pipeline("")
        assert len(errors) == 0
        compile(code, "<empty>", "exec")

    def test_comments_only(self):
        code, errors, warnings = full_pipeline("// just a comment\n")
        assert len(errors) == 0

    def test_multiple_models(self):
        source = """model Encoder:
    layer fc = Linear(784, 128)
    fn forward(self, x: Tensor) -> Tensor:
        return self.fc(x)

model Decoder:
    layer fc = Linear(128, 784)
    fn forward(self, x: Tensor) -> Tensor:
        return self.fc(x)
"""
        code, errors, warnings = full_pipeline(source)
        assert len(errors) == 0
        assert "class Encoder(nn.Module):" in code
        assert "class Decoder(nn.Module):" in code
