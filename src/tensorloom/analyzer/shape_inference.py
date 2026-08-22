"""
TensorLoom Shape Inference Engine — Automated Dimension Tracking.

Tracks tensor shapes through the entire program at compile time and detects
dimension mismatches *before* any GPU time is wasted.

Capabilities:
  - Track shapes through tensor literals, binary ops, function calls
  - Propagate shapes through model layer chains (Linear, Conv2d, etc.)
  - Validate pipe chains for shape compatibility
  - Detect matmul dimension mismatches
  - Auto-infer output dimensions of convolutional + pooling pipelines
  - Report clear, actionable error messages with line numbers

Shape representation:
  - Shape = list of int | "?" (unknown) dimensions
  - "?" propagates through unknown operations
  - Batch dim represented as -1 (any)
"""
from __future__ import annotations

from dataclasses import dataclass, field

from tensorloom.parser.ast_nodes import (
    ASTNode,
    AssignStatement,
    BinaryOp,
    BooleanLiteral,
    ExpressionStatement,
    ForLoop,
    FunctionCall,
    FunctionDef,
    Identifier,
    IfStatement,
    IndexAccess,
    LetStatement,
    ListLiteral,
    MemberAccess,
    ModelDefinition,
    NumberLiteral,
    PipeExpression,
    PipeStage,
    Program,
    ReturnStatement,
    StringLiteral,
    TensorLiteral,
    TrainBlock,
)


# ── Shape Types ───────────────────────────────────────────────

# A dimension is either a known integer or unknown ("?")
Dim = int | str  # int for known, "?" for unknown


@dataclass
class Shape:
    """Represents a tensor shape. Dims can be int or '?' for unknown."""
    dims: list[Dim]

    @property
    def ndim(self) -> int:
        return len(self.dims)

    @property
    def known(self) -> bool:
        """True if all dimensions are known integers."""
        return all(isinstance(d, int) for d in self.dims)

    def __repr__(self) -> str:
        parts = [str(d) if isinstance(d, int) else "?" for d in self.dims]
        return f"[{', '.join(parts)}]"

    def compatible_with(self, other: "Shape") -> bool:
        """Check if two shapes are broadcastable."""
        if self.ndim == 0 or other.ndim == 0:
            return True  # scalars broadcast with anything

        # Align from right
        for a, b in zip(reversed(self.dims), reversed(other.dims)):
            if a == "?" or b == "?":
                continue
            if a != b and a != 1 and b != 1:
                return False
        return True

    def broadcast_with(self, other: "Shape") -> "Shape":
        """Compute the result shape of broadcasting two shapes."""
        if self.ndim == 0:
            return Shape(list(other.dims))
        if other.ndim == 0:
            return Shape(list(self.dims))

        result: list[Dim] = []
        max_ndim = max(self.ndim, other.ndim)

        a_padded = [1] * (max_ndim - self.ndim) + list(self.dims)
        b_padded = [1] * (max_ndim - other.ndim) + list(other.dims)

        for a, b in zip(a_padded, b_padded):
            if a == "?" or b == "?":
                result.append("?")
            elif a == 1:
                result.append(b)
            elif b == 1:
                result.append(a)
            else:
                result.append(a)  # a == b already validated

        return Shape(result)


UNKNOWN_SHAPE = Shape(["?"])
SCALAR_SHAPE = Shape([])


# ── Shape Errors ──────────────────────────────────────────────

@dataclass
class ShapeError:
    """A shape mismatch detected at compile time."""
    message: str
    line: int
    severity: str = "error"  # "error" or "warning"


@dataclass
class ShapeInfo:
    """Shape information for a tracked variable."""
    name: str
    shape: Shape
    line: int


@dataclass
class LayerShapeSpec:
    """The input/output shape transformation for a neural network layer."""
    layer_name: str
    input_shape: Shape
    output_shape: Shape


@dataclass
class ShapeReport:
    """Complete shape analysis report."""
    errors: list[ShapeError] = field(default_factory=list)
    warnings: list[ShapeError] = field(default_factory=list)
    variable_shapes: dict[str, Shape] = field(default_factory=dict)
    model_shapes: dict[str, list[LayerShapeSpec]] = field(default_factory=dict)


# ── Layer Shape Rules ─────────────────────────────────────────

def _linear_shape(args: list, input_shape: Shape) -> Shape:
    """Linear(in_features, out_features) -> [..., out_features]"""
    if len(args) >= 2 and isinstance(args[1], int):
        if input_shape.ndim >= 1:
            return Shape(list(input_shape.dims[:-1]) + [args[1]])
        return Shape([args[1]])
    return UNKNOWN_SHAPE


def _conv2d_shape(args: list, kwargs: dict, input_shape: Shape) -> Shape:
    """Conv2d(in_ch, out_ch, kernel_size, stride=1, padding=0)
       Input:  [B, C_in, H, W]
       Output: [B, C_out, H', W']
       H' = (H + 2*padding - kernel_size) // stride + 1
    """
    if len(args) < 3:
        return UNKNOWN_SHAPE

    out_channels = args[1] if isinstance(args[1], int) else None
    kernel_size = args[2] if isinstance(args[2], int) else None
    stride = kwargs.get("stride", 1) if isinstance(kwargs.get("stride", 1), int) else 1
    padding = kwargs.get("padding", 0) if isinstance(kwargs.get("padding", 0), int) else 0

    if out_channels is None or kernel_size is None:
        return UNKNOWN_SHAPE

    if input_shape.ndim == 4:
        batch = input_shape.dims[0]
        h_in = input_shape.dims[2]
        w_in = input_shape.dims[3]

        if isinstance(h_in, int) and isinstance(w_in, int):
            h_out = (h_in + 2 * padding - kernel_size) // stride + 1
            w_out = (w_in + 2 * padding - kernel_size) // stride + 1
            return Shape([batch, out_channels, h_out, w_out])

        return Shape([batch, out_channels, "?", "?"])

    return Shape(["?", out_channels, "?", "?"])


def _pool2d_shape(args: list, kwargs: dict, input_shape: Shape) -> Shape:
    """MaxPool2d / AvgPool2d(kernel_size, stride=None, padding=0)
       Reduces spatial dims: H' = (H - k) // s + 1
    """
    kernel_size = args[0] if len(args) >= 1 and isinstance(args[0], int) else None
    stride = kwargs.get("stride", kernel_size)
    if not isinstance(stride, int):
        stride = kernel_size
    padding = kwargs.get("padding", 0) if isinstance(kwargs.get("padding", 0), int) else 0

    if kernel_size is None or input_shape.ndim != 4:
        return UNKNOWN_SHAPE

    batch = input_shape.dims[0]
    channels = input_shape.dims[1]
    h_in = input_shape.dims[2]
    w_in = input_shape.dims[3]

    if isinstance(h_in, int) and isinstance(w_in, int) and stride is not None:
        h_out = (h_in + 2 * padding - kernel_size) // stride + 1
        w_out = (w_in + 2 * padding - kernel_size) // stride + 1
        return Shape([batch, channels, h_out, w_out])

    return Shape([batch, channels, "?", "?"])


def _flatten_shape(input_shape: Shape) -> Shape:
    """Flatten() -> [batch, product_of_remaining_dims]"""
    if input_shape.ndim < 2:
        return input_shape

    batch = input_shape.dims[0]
    remaining = input_shape.dims[1:]

    if all(isinstance(d, int) for d in remaining):
        flat = 1
        for d in remaining:
            flat *= d
        return Shape([batch, flat])

    return Shape([batch, "?"])


# Layer shape dispatch table
LAYER_SHAPE_RULES: dict[str, callable] = {}  # Populated in the engine


# ── Shape Inference Engine ────────────────────────────────────

class ShapeInferenceEngine:
    """Tracks tensor shapes through a TensorLoom program at compile time.

    Detects dimension mismatches and shape incompatibilities before any
    GPU time is wasted. Can auto-calculate flattened feature counts
    after Conv2d + Pooling chains.
    """

    def __init__(self) -> None:
        self.report = ShapeReport()
        self._shapes: dict[str, Shape] = {}  # variable -> shape
        self._layer_shapes: dict[str, dict[str, tuple[Shape, callable]]] = {}  # model -> layer -> (input_spec, transform)
        self._model_layer_specs: dict[str, dict[str, tuple[str, list[int], dict]]] = {}  # model -> layer -> (type, args, kwargs)

    def infer(self, program: Program) -> ShapeReport:
        """Run shape inference on the entire program."""
        for stmt in program.statements:
            self._infer_statement(stmt)

        self.report.variable_shapes = dict(self._shapes)
        return self.report

    # ── Statement Inference ───────────────────────────────────

    def _infer_statement(self, node: ASTNode) -> None:
        if isinstance(node, LetStatement):
            shape = self._infer_expr(node.value, node.line)
            if shape is not None:
                self._shapes[node.name] = shape

        elif isinstance(node, AssignStatement):
            shape = self._infer_expr(node.value, node.line)
            if isinstance(node.target, Identifier) and shape is not None:
                self._shapes[node.target.name] = shape

        elif isinstance(node, ModelDefinition):
            self._infer_model(node)

        elif isinstance(node, ForLoop):
            for stmt in node.body:
                self._infer_statement(stmt)

        elif isinstance(node, IfStatement):
            for stmt in node.body:
                self._infer_statement(stmt)

        elif isinstance(node, FunctionDef):
            for stmt in node.body:
                self._infer_statement(stmt)

        elif isinstance(node, ExpressionStatement):
            self._infer_expr(node.expression, node.line)

    # ── Expression Shape Inference ────────────────────────────

    def _infer_expr(self, node: ASTNode, line: int) -> Shape | None:
        """Infer the shape of an expression. Returns None if unknown."""
        if isinstance(node, NumberLiteral):
            return SCALAR_SHAPE

        elif isinstance(node, StringLiteral):
            return None  # strings have no tensor shape

        elif isinstance(node, BooleanLiteral):
            return SCALAR_SHAPE

        elif isinstance(node, TensorLiteral):
            return self._infer_tensor_literal(node)

        elif isinstance(node, ListLiteral):
            return Shape([len(node.elements)])

        elif isinstance(node, Identifier):
            return self._shapes.get(node.name)

        elif isinstance(node, BinaryOp):
            return self._infer_binary_op(node, line)

        elif isinstance(node, PipeExpression):
            return self._infer_pipe(node, line)

        elif isinstance(node, FunctionCall):
            return self._infer_function_call(node, line)

        elif isinstance(node, MemberAccess):
            return self._infer_member_access(node, line)

        elif isinstance(node, IndexAccess):
            # Indexing reduces dimensionality — approximation
            return UNKNOWN_SHAPE

        return None

    def _infer_tensor_literal(self, node: TensorLiteral) -> Shape:
        """Infer shape of tensor([...])."""
        # Check for nested lists
        elements = node.elements
        if not elements:
            return Shape([0])

        # Simple 1D tensor
        if not any(isinstance(e, ListLiteral) for e in elements):
            return Shape([len(elements)])

        # 2D tensor: [[...], [...], ...]
        if all(isinstance(e, ListLiteral) for e in elements):
            inner_lens = [len(e.elements) for e in elements]
            if len(set(inner_lens)) == 1:
                return Shape([len(elements), inner_lens[0]])
            else:
                self.report.warnings.append(ShapeError(
                    f"Ragged tensor literal: inner dimensions differ ({inner_lens})",
                    node.line,
                    severity="warning",
                ))
                return Shape([len(elements), "?"])

        return Shape([len(elements)])

    def _infer_binary_op(self, node: BinaryOp, line: int) -> Shape | None:
        """Infer shape of a binary operation."""
        left = self._infer_expr(node.left, line)
        right = self._infer_expr(node.right, line)

        if left is None or right is None:
            return left or right  # propagate what we know

        if node.op == "@":
            return self._infer_matmul(left, right, line)

        elif node.op in ("+", "-", "*", "/", "**"):
            # Broadcast rules
            if not left.compatible_with(right):
                self.report.errors.append(ShapeError(
                    f"Shape mismatch in '{node.op}': {left} is not broadcastable with {right}",
                    line,
                ))
                return left
            return left.broadcast_with(right)

        elif node.op in ("==", "!=", "<", ">", "<=", ">="):
            # Comparison ops preserve shape
            return left.broadcast_with(right)

        return UNKNOWN_SHAPE

    def _infer_matmul(self, left: Shape, right: Shape, line: int) -> Shape | None:
        """Infer shape of matrix multiplication (@ operator)."""
        if left.ndim == 0 or right.ndim == 0:
            self.report.errors.append(ShapeError(
                f"Cannot matmul a scalar: {left} @ {right}",
                line,
            ))
            return None

        # 1D @ 1D -> scalar (dot product)
        if left.ndim == 1 and right.ndim == 1:
            l_dim = left.dims[0]
            r_dim = right.dims[0]
            if isinstance(l_dim, int) and isinstance(r_dim, int) and l_dim != r_dim:
                self.report.errors.append(ShapeError(
                    f"Dimension mismatch in dot product: {left} @ {right} "
                    f"(need matching inner dims, got {l_dim} vs {r_dim})",
                    line,
                ))
            return SCALAR_SHAPE

        # 2D @ 2D -> 2D
        if left.ndim == 2 and right.ndim == 2:
            m, k1 = left.dims
            k2, n = right.dims
            if isinstance(k1, int) and isinstance(k2, int) and k1 != k2:
                self.report.errors.append(ShapeError(
                    f"Dimension mismatch in matmul: {left} @ {right} "
                    f"(inner dims must match: {k1} != {k2})",
                    line,
                ))
            return Shape([m, n])

        # 1D @ 2D -> 1D
        if left.ndim == 1 and right.ndim == 2:
            return Shape([right.dims[1]])

        # 2D @ 1D -> 1D
        if left.ndim == 2 and right.ndim == 1:
            return Shape([left.dims[0]])

        # Batched matmul: [..., m, k] @ [..., k, n] -> [..., m, n]
        if left.ndim >= 2 and right.ndim >= 2:
            batch_dims = list(left.dims[:-2])
            return Shape(batch_dims + [left.dims[-2], right.dims[-1]])

        return UNKNOWN_SHAPE

    def _infer_pipe(self, node: PipeExpression, line: int) -> Shape | None:
        """Infer shape through a pipe chain: value |> f1 |> f2 |> ..."""
        current_shape = self._infer_expr(node.value, node.line)

        for stage in node.stages:
            if current_shape is None:
                break
            current_shape = self._apply_pipe_stage(stage, current_shape, node.line)

        return current_shape

    def _apply_pipe_stage(self, stage: PipeStage, input_shape: Shape, line: int) -> Shape | None:
        """Apply a pipe stage to a shape and return the output shape."""
        name = stage.func_name

        # Shape-preserving activations
        if name in ("relu", "sigmoid", "tanh", "gelu", "silu",
                     "dropout", "batch_norm", "layer_norm"):
            return input_shape

        # Softmax: preserves shape
        if name == "softmax" or name == "log_softmax":
            return input_shape

        # Flatten: collapse non-batch dims
        if name == "flatten":
            return _flatten_shape(input_shape)

        # Reshape: depends on args
        if name == "reshape":
            # We'd need the target shape args — for now, return unknown
            return UNKNOWN_SHAPE

        # Normalize: preserves shape
        if name == "normalize":
            return input_shape

        # Layer calls: check model context
        # (handled via member access in assign statements)

        return input_shape  # Default: assume shape-preserving

    def _infer_function_call(self, node: FunctionCall, line: int) -> Shape | None:
        """Infer shape of a function call."""
        if isinstance(node.callee, Identifier):
            name = node.callee.name

            # Model instantiation: no shape yet
            if name in ("load", "MNIST", "mnist"):
                return None

            # range(), len(), etc. return scalars
            if name in ("range", "len", "type", "int", "float", "abs", "min", "max"):
                return SCALAR_SHAPE

            # print returns None
            if name == "print":
                return None

            # tensor([...]) — handled as TensorLiteral normally
            if name == "tensor":
                if node.args and isinstance(node.args[0], ListLiteral):
                    return Shape([len(node.args[0].elements)])

        return None

    def _infer_member_access(self, node: MemberAccess, line: int) -> Shape | None:
        """Infer shape from member access like self.layer(x)."""
        # For self.fc1(x) — this is handled in the assign context
        # when we see x = self.fc1(x) in a model forward method
        return None

    # ── Model Shape Analysis ──────────────────────────────────

    def _infer_model(self, model: ModelDefinition) -> None:
        """Analyze shape flow through a model's layers and forward pass."""
        model_name = model.name
        layer_specs: dict[str, tuple[str, list[int], dict]] = {}
        layer_shapes: list[LayerShapeSpec] = []

        # Extract layer specifications
        for layer in model.layers:
            if isinstance(layer.layer_type, FunctionCall):
                call = layer.layer_type
                if isinstance(call.callee, Identifier):
                    layer_type_name = call.callee.name
                    args = []
                    for a in call.args:
                        if isinstance(a, NumberLiteral):
                            args.append(int(a.value) if a.is_integer else a.value)
                        else:
                            args.append("?")

                    kwargs = {}
                    for k, v in call.kwargs.items():
                        if isinstance(v, NumberLiteral):
                            kwargs[k] = int(v.value) if v.is_integer else v.value

                    layer_specs[layer.name] = (layer_type_name, args, kwargs)

        self._model_layer_specs[model_name] = layer_specs

        # Trace shape flow through the forward method
        for method in model.methods:
            if method.name == "forward":
                self._trace_forward(model_name, method, layer_specs, layer_shapes)

        if layer_shapes:
            self.report.model_shapes[model_name] = layer_shapes

    def _trace_forward(
        self,
        model_name: str,
        method: FunctionDef,
        layer_specs: dict[str, tuple[str, list[int], dict]],
        layer_shapes: list[LayerShapeSpec],
    ) -> None:
        """Trace tensor shapes through a forward method."""
        # Determine input shape from first layer's input dimension
        # e.g., if first layer is Linear(784, 256), input is [batch, 784]
        input_shape: Shape | None = None

        for _name, (layer_type, args, _kwargs) in layer_specs.items():
            if layer_type == "Linear" and len(args) >= 1 and isinstance(args[0], int):
                input_shape = Shape(["?", args[0]])  # [batch, in_features]
                break
            elif layer_type == "Conv2d" and len(args) >= 1 and isinstance(args[0], int):
                input_shape = Shape(["?", args[0], "?", "?"])  # [batch, channels, H, W]
                break

        if input_shape is None:
            return

        # Track shape through each statement in forward()
        # Set up scope: "x" (or the first param) starts with input_shape
        local_shapes: dict[str, Shape] = {}
        if len(method.params) >= 2:
            local_shapes[method.params[1].name] = input_shape

        for stmt in method.body:
            if isinstance(stmt, AssignStatement):
                shape = self._trace_forward_expr(
                    stmt.value, local_shapes, layer_specs, layer_shapes, stmt.line
                )
                if isinstance(stmt.target, Identifier) and shape is not None:
                    local_shapes[stmt.target.name] = shape

            elif isinstance(stmt, ReturnStatement) and stmt.value:
                self._trace_forward_expr(
                    stmt.value, local_shapes, layer_specs, layer_shapes, stmt.line
                )

    def _trace_forward_expr(
        self,
        node: ASTNode,
        local_shapes: dict[str, Shape],
        layer_specs: dict[str, tuple[str, list[int], dict]],
        layer_shapes: list[LayerShapeSpec],
        line: int,
    ) -> Shape | None:
        """Trace shape through an expression inside a forward method."""
        if isinstance(node, Identifier):
            return local_shapes.get(node.name)

        elif isinstance(node, PipeExpression):
            # Get the initial shape of the pipe value
            current_shape = self._trace_forward_expr(
                node.value, local_shapes, layer_specs, layer_shapes, line
            )

            for stage in node.stages:
                if current_shape is None:
                    break
                current_shape = self._apply_pipe_stage(stage, current_shape, line)

            return current_shape

        elif isinstance(node, FunctionCall):
            # Check if this is a self.layer_name(x) call
            if isinstance(node.callee, MemberAccess):
                if (isinstance(node.callee.object, Identifier)
                        and node.callee.object.name == "self"):
                    layer_name = node.callee.member

                    if layer_name in layer_specs:
                        layer_type, args, kwargs = layer_specs[layer_name]

                        # Get input shape from the argument
                        arg_shape = None
                        if node.args:
                            arg_shape = self._trace_forward_expr(
                                node.args[0], local_shapes, layer_specs, layer_shapes, line
                            )

                        if arg_shape is None:
                            arg_shape = UNKNOWN_SHAPE

                        # Apply layer shape transformation
                        output_shape = self._apply_layer_shape(
                            layer_type, args, kwargs, arg_shape, line
                        )

                        # Validate input dimension for Linear layers
                        if layer_type == "Linear" and len(args) >= 1:
                            expected_in = args[0]
                            if (isinstance(expected_in, int)
                                    and arg_shape.ndim >= 1
                                    and isinstance(arg_shape.dims[-1], int)
                                    and arg_shape.dims[-1] != expected_in):
                                self.report.errors.append(ShapeError(
                                    f"Dimension mismatch at layer '{layer_name}': "
                                    f"expected input features={expected_in}, "
                                    f"got {arg_shape.dims[-1]} "
                                    f"(input shape: {arg_shape})",
                                    line,
                                ))

                        layer_shapes.append(LayerShapeSpec(
                            layer_name=layer_name,
                            input_shape=arg_shape,
                            output_shape=output_shape,
                        ))

                        return output_shape

            # Generic function call
            if node.args:
                return self._trace_forward_expr(
                    node.args[0], local_shapes, layer_specs, layer_shapes, line
                )

        elif isinstance(node, BinaryOp):
            left = self._trace_forward_expr(
                node.left, local_shapes, layer_specs, layer_shapes, line
            )
            right = self._trace_forward_expr(
                node.right, local_shapes, layer_specs, layer_shapes, line
            )
            if left is not None and right is not None:
                if node.op == "@":
                    return self._infer_matmul(left, right, line)
                elif node.op in ("+", "-", "*", "/"):
                    if not left.compatible_with(right):
                        self.report.errors.append(ShapeError(
                            f"Shape mismatch in forward pass: "
                            f"{left} {node.op} {right} is not broadcastable",
                            line,
                        ))
                    return left.broadcast_with(right)
            return left or right

        return None

    def _apply_layer_shape(
        self,
        layer_type: str,
        args: list,
        kwargs: dict,
        input_shape: Shape,
        line: int,
    ) -> Shape:
        """Apply a known layer type's shape transformation."""
        if layer_type == "Linear":
            return _linear_shape(args, input_shape)

        elif layer_type == "Conv2d":
            return _conv2d_shape(args, kwargs, input_shape)

        elif layer_type == "Conv1d":
            # Similar to Conv2d but 1D
            if len(args) >= 2 and isinstance(args[1], int):
                return Shape(list(input_shape.dims[:-1]) + ["?"])
            return UNKNOWN_SHAPE

        elif layer_type in ("MaxPool2d", "AvgPool2d"):
            return _pool2d_shape(args, kwargs, input_shape)

        elif layer_type == "AdaptiveAvgPool2d":
            if len(args) >= 1 and isinstance(args[0], int):
                if input_shape.ndim == 4:
                    return Shape([input_shape.dims[0], input_shape.dims[1], args[0], args[0]])
            return UNKNOWN_SHAPE

        elif layer_type in ("BatchNorm2d", "LayerNorm"):
            return input_shape  # shape-preserving

        elif layer_type == "Dropout":
            return input_shape  # shape-preserving

        elif layer_type == "Flatten":
            return _flatten_shape(input_shape)

        elif layer_type == "Embedding":
            # Embedding(vocab, dim) -> [..., dim]
            if len(args) >= 2 and isinstance(args[1], int):
                return Shape(list(input_shape.dims) + [args[1]])
            return UNKNOWN_SHAPE

        elif layer_type in ("LSTM", "GRU"):
            # Output shape depends on configuration
            return UNKNOWN_SHAPE

        elif layer_type in ("GELU", "ReLU", "Sigmoid", "Tanh", "SiLU"):
            return input_shape  # activation layers preserve shape

        elif layer_type == "MultiHeadAttention":
            return input_shape  # approximate

        return UNKNOWN_SHAPE

    # ── Utility: Auto-calculate flattened features ────────────

    def calculate_flattened_features(
        self,
        model: ModelDefinition,
    ) -> int | None:
        """Calculate the number of features after conv + pool + flatten.

        Useful for automatically computing the input dimension of the
        first Linear layer after a convolutional pipeline.

        Returns the flat feature count, or None if it can't be determined.
        """
        if model.name not in self.report.model_shapes:
            return None

        shapes = self.report.model_shapes[model.name]

        # Find the last flatten layer and get its output shape
        for spec in reversed(shapes):
            if spec.layer_name.lower() in ("flatten",) or \
               (spec.output_shape.ndim == 2 and spec.input_shape.ndim > 2):
                last_dim = spec.output_shape.dims[-1]
                if isinstance(last_dim, int):
                    return last_dim

        return None
