"""
Tests for the TensorLoom Shape Inference Engine.

Validates compile-time shape tracking, dimension mismatch detection,
and shape propagation through model layer chains.
"""
import pytest
from tensorloom.lexer.lexer import Lexer
from tensorloom.parser.parser import Parser
from tensorloom.analyzer.shape_inference import (
    Shape,
    ShapeInferenceEngine,
    SCALAR_SHAPE,
    UNKNOWN_SHAPE,
    _linear_shape,
    _conv2d_shape,
    _pool2d_shape,
    _flatten_shape,
)


def infer_shapes(source: str):
    """Helper: lex + parse + infer shapes."""
    tokens = Lexer(source).tokenize()
    ast = Parser(tokens).parse()
    engine = ShapeInferenceEngine()
    return engine.infer(ast)


# ── Shape Object Tests ────────────────────────────────────────

class TestShapeBasics:
    def test_shape_repr(self):
        assert str(Shape([3, 4])) == "[3, 4]"
        assert str(Shape(["?", 10])) == "[?, 10]"

    def test_scalar_shape(self):
        assert SCALAR_SHAPE.ndim == 0
        assert SCALAR_SHAPE.known

    def test_unknown_shape(self):
        assert UNKNOWN_SHAPE.ndim == 1
        assert not UNKNOWN_SHAPE.known

    def test_broadcast_compatible(self):
        a = Shape([3, 4])
        b = Shape([3, 4])
        assert a.compatible_with(b)

    def test_broadcast_incompatible(self):
        a = Shape([3, 4])
        b = Shape([3, 5])
        assert not a.compatible_with(b)

    def test_broadcast_scalar(self):
        a = Shape([3, 4])
        b = SCALAR_SHAPE
        assert a.compatible_with(b)

    def test_broadcast_widening(self):
        a = Shape([3, 1])
        b = Shape([3, 4])
        assert a.compatible_with(b)

    def test_broadcast_result(self):
        a = Shape([3, 1])
        b = Shape([1, 4])
        result = a.broadcast_with(b)
        assert result.dims == [3, 4]

    def test_broadcast_different_ndim(self):
        a = Shape([4])
        b = Shape([3, 4])
        assert a.compatible_with(b)
        result = a.broadcast_with(b)
        assert result.dims == [3, 4]


# ── Layer Shape Rule Tests ────────────────────────────────────

class TestLayerShapeRules:
    def test_linear_shape(self):
        result = _linear_shape([784, 256], Shape(["?", 784]))
        assert result.dims == ["?", 256]

    def test_linear_shape_1d(self):
        result = _linear_shape([784, 256], Shape([784]))
        assert result.dims == [256]

    def test_conv2d_shape(self):
        result = _conv2d_shape([3, 16, 3], {}, Shape(["?", 3, 32, 32]))
        assert result.dims == ["?", 16, 30, 30]

    def test_conv2d_with_padding(self):
        result = _conv2d_shape([3, 16, 3], {"padding": 1}, Shape(["?", 3, 32, 32]))
        assert result.dims == ["?", 16, 32, 32]

    def test_pool2d_shape(self):
        result = _pool2d_shape([2], {}, Shape(["?", 16, 30, 30]))
        assert result.dims == ["?", 16, 15, 15]

    def test_flatten_shape(self):
        result = _flatten_shape(Shape(["?", 16, 15, 15]))
        assert result.dims == ["?", 3600]  # 16 * 15 * 15

    def test_flatten_2d_noop(self):
        result = _flatten_shape(Shape(["?", 784]))
        assert result.dims == ["?", 784]


# ── Tensor Literal Shape Inference ────────────────────────────

class TestTensorShapeInference:
    def test_1d_tensor(self):
        report = infer_shapes("let x = tensor([1.0, 2.0, 3.0], dtype=float32, device=gpu)\n")
        assert "x" in report.variable_shapes
        assert report.variable_shapes["x"].dims == [3]

    def test_scalar(self):
        report = infer_shapes("let x = 42\n")
        assert "x" in report.variable_shapes
        assert report.variable_shapes["x"].ndim == 0

    def test_list_literal(self):
        report = infer_shapes("let x = [1, 2, 3, 4]\n")
        assert "x" in report.variable_shapes
        assert report.variable_shapes["x"].dims == [4]


# ── Binary Operation Shape Inference ─────────────────────────

class TestBinaryOpShapes:
    def test_dot_product(self):
        source = """let x = tensor([1.0, 2.0, 3.0], dtype=float32, device=gpu)
let y = tensor([4.0, 5.0, 6.0], dtype=float32, device=gpu)
let z = x @ y
"""
        report = infer_shapes(source)
        assert "z" in report.variable_shapes
        assert report.variable_shapes["z"].ndim == 0  # scalar (dot product)

    def test_dot_product_mismatch(self):
        source = """let x = tensor([1.0, 2.0, 3.0], dtype=float32, device=gpu)
let y = tensor([4.0, 5.0], dtype=float32, device=gpu)
let z = x @ y
"""
        report = infer_shapes(source)
        assert len(report.errors) == 1
        assert "Dimension mismatch" in report.errors[0].message
        assert "3" in report.errors[0].message
        assert "2" in report.errors[0].message

    def test_elementwise_add(self):
        source = """let x = tensor([1.0, 2.0, 3.0], dtype=float32, device=gpu)
let y = tensor([4.0, 5.0, 6.0], dtype=float32, device=gpu)
let z = x + y
"""
        report = infer_shapes(source)
        assert report.variable_shapes["z"].dims == [3]
        assert len(report.errors) == 0

    def test_elementwise_shape_mismatch(self):
        source = """let a = tensor([1.0, 2.0, 3.0], dtype=float32, device=gpu)
let b = tensor([4.0, 5.0], dtype=float32, device=gpu)
let c = a + b
"""
        report = infer_shapes(source)
        assert len(report.errors) == 1
        assert "broadcastable" in report.errors[0].message

    def test_scalar_broadcast(self):
        source = """let x = tensor([1.0, 2.0, 3.0], dtype=float32, device=gpu)
let y = x + 0.5
"""
        report = infer_shapes(source)
        assert report.variable_shapes["y"].dims == [3]
        assert len(report.errors) == 0


# ── Pipe Chain Shape Inference ────────────────────────────────

class TestPipeShapes:
    def test_relu_preserves_shape(self):
        source = """let x = tensor([1.0, 2.0, 3.0], dtype=float32, device=gpu)
let y = x |> relu
"""
        report = infer_shapes(source)
        assert report.variable_shapes["y"].dims == [3]

    def test_pipe_chain_preserves(self):
        source = """let x = tensor([1.0, 2.0, 3.0], dtype=float32, device=gpu)
let y = x |> relu |> sigmoid |> dropout(0.5)
"""
        report = infer_shapes(source)
        assert report.variable_shapes["y"].dims == [3]


# ── Model Shape Flow Inference ────────────────────────────────

class TestModelShapeFlow:
    def test_simple_model(self):
        source = """model Net:
    layer fc1 = Linear(784, 256)
    layer fc2 = Linear(256, 10)
    fn forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x) |> relu
        x = self.fc2(x) |> softmax
        return x
"""
        report = infer_shapes(source)
        assert "Net" in report.model_shapes
        shapes = report.model_shapes["Net"]
        assert len(shapes) == 2

        # fc1: [?, 784] -> [?, 256]
        assert shapes[0].layer_name == "fc1"
        assert shapes[0].input_shape.dims == ["?", 784]
        assert shapes[0].output_shape.dims == ["?", 256]

        # fc2: [?, 256] -> [?, 10]
        assert shapes[1].layer_name == "fc2"
        assert shapes[1].input_shape.dims == ["?", 256]
        assert shapes[1].output_shape.dims == ["?", 10]

    def test_model_dimension_mismatch(self):
        """THE key test: detect mismatched layer dimensions at compile time."""
        source = """model BadNet:
    layer fc1 = Linear(784, 256)
    layer fc2 = Linear(128, 10)
    fn forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x) |> relu
        x = self.fc2(x) |> softmax
        return x
"""
        report = infer_shapes(source)
        assert len(report.errors) == 1
        error = report.errors[0]
        assert "Dimension mismatch" in error.message
        assert "fc2" in error.message
        assert "128" in error.message
        assert "256" in error.message

    def test_three_layer_chain(self):
        source = """model Deep:
    layer l1 = Linear(1024, 512)
    layer l2 = Linear(512, 256)
    layer l3 = Linear(256, 10)
    fn forward(self, x: Tensor) -> Tensor:
        x = self.l1(x) |> relu
        x = self.l2(x) |> relu
        x = self.l3(x) |> softmax
        return x
"""
        report = infer_shapes(source)
        assert len(report.errors) == 0
        shapes = report.model_shapes["Deep"]
        assert len(shapes) == 3
        assert shapes[0].output_shape.dims == ["?", 512]
        assert shapes[1].output_shape.dims == ["?", 256]
        assert shapes[2].output_shape.dims == ["?", 10]

    def test_three_layer_mismatch_middle(self):
        """Detect mismatch in the middle of a layer chain."""
        source = """model BadDeep:
    layer l1 = Linear(784, 256)
    layer l2 = Linear(512, 128)
    layer l3 = Linear(128, 10)
    fn forward(self, x: Tensor) -> Tensor:
        x = self.l1(x) |> relu
        x = self.l2(x) |> relu
        x = self.l3(x) |> softmax
        return x
"""
        report = infer_shapes(source)
        assert len(report.errors) == 1
        assert "l2" in report.errors[0].message
        assert "512" in report.errors[0].message
        assert "256" in report.errors[0].message


class TestMNISTShapeFlow:
    """Test shape inference on the full MNIST classifier pipeline."""

    MNIST_SOURCE = """model MNISTClassifier:
    layer input  = Linear(784, 256)
    layer hidden = Linear(256, 128)
    layer output = Linear(128, 10)
    fn forward(self, x: Tensor) -> Tensor:
        x = self.input(x) |> relu |> dropout(0.2)
        x = self.hidden(x) |> relu |> dropout(0.2)
        x = self.output(x) |> softmax
        return x
"""

    def test_no_errors(self):
        report = infer_shapes(self.MNIST_SOURCE)
        assert len(report.errors) == 0

    def test_shape_flow(self):
        report = infer_shapes(self.MNIST_SOURCE)
        shapes = report.model_shapes["MNISTClassifier"]
        assert len(shapes) == 3

        # input: [?, 784] -> [?, 256]
        assert shapes[0].input_shape.dims == ["?", 784]
        assert shapes[0].output_shape.dims == ["?", 256]

        # hidden: [?, 256] -> [?, 128]
        assert shapes[1].input_shape.dims == ["?", 256]
        assert shapes[1].output_shape.dims == ["?", 128]

        # output: [?, 128] -> [?, 10]
        assert shapes[2].input_shape.dims == ["?", 128]
        assert shapes[2].output_shape.dims == ["?", 10]


# ── Gradient Checkpointing Code Generation Tests ─────────────

class TestGradientCheckpointCodegen:
    """Test that gradient checkpointing generates correct activation checkpoint code."""

    def _compile(self, source: str) -> str:
        from tensorloom.codegen.pytorch_backend import PyTorchBackend
        tokens = Lexer(source).tokenize()
        ast = Parser(tokens).parse()
        return PyTorchBackend().generate(ast)

    def test_checkpoint_generates_forward_body(self):
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
        code = self._compile(source)
        # Should have _forward_body method
        assert "_forward_body" in code
        # Should have activation_checkpoint call
        assert "activation_checkpoint" in code
        # Should have use_reentrant=False
        assert "use_reentrant=False" in code
        # Should still have weight snapshot saving
        assert "torch.save" in code
        # Should NOT call it "Gradient Checkpointing" anymore
        assert "Weight Snapshot" in code

    def test_checkpoint_import(self):
        source = """model Net:
    layer fc = Linear(10, 5)
    fn forward(self, x: Tensor) -> Tensor:
        return self.fc(x)

let net = Net()

train net on data:
    epochs = 5
    optimizer = Adam(lr=0.001)
    loss = CrossEntropy
    checkpoint every 1 epochs
"""
        code = self._compile(source)
        assert "from torch.utils.checkpoint import checkpoint as activation_checkpoint" in code

    def test_no_checkpoint_normal_forward(self):
        """Without checkpoint, forward should be normal (no _forward_body)."""
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
        code = self._compile(source)
        assert "_forward_body" not in code
        assert "activation_checkpoint" not in code

    def test_generated_code_is_valid_python(self):
        source = """model Net:
    layer fc1 = Linear(10, 5)
    layer fc2 = Linear(5, 2)
    fn forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x) |> relu
        x = self.fc2(x)
        return x

let net = Net()

train net on data:
    epochs = 5
    optimizer = Adam(lr=0.001)
    loss = CrossEntropy
    checkpoint every 2 epochs
"""
        code = self._compile(source)
        # Must compile as valid Python
        compile(code, "<test>", "exec")
