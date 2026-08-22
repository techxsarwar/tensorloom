"""
TensorLoom AST Optimizer — Pre-codegen optimization passes.

Operates on the AST before code generation to:
  - Detect fusible operation chains
  - Estimate GPU memory requirements
  - Propagate precision hints
  - Eliminate dead code
"""
from __future__ import annotations

from dataclasses import dataclass, field

from tensorloom.parser.ast_nodes import (
    ASTNode,
    AssignStatement,
    BinaryOp,
    FunctionCall,
    FunctionDef,
    Identifier,
    LetStatement,
    ModelDefinition,
    NumberLiteral,
    PipeExpression,
    Program,
    TrainBlock,
)


@dataclass
class FusionOpportunity:
    """Records a detected opportunity for kernel fusion."""
    description: str
    line: int
    ops: list[str] = field(default_factory=list)


@dataclass
class MemoryEstimate:
    """Estimated GPU memory for a model."""
    model_name: str
    total_params: int
    param_memory_mb: float
    training_memory_mb: float  # includes gradients + optimizer states
    fp16_memory_mb: float


@dataclass
class OptimizationReport:
    """Summary of all optimizations detected."""
    fusions: list[FusionOpportunity] = field(default_factory=list)
    memory_estimates: list[MemoryEstimate] = field(default_factory=list)
    dead_imports: list[str] = field(default_factory=list)
    precision_hints: dict[str, str] = field(default_factory=dict)


class ASTOptimizer:
    """Performs optimization analysis on the TensorLoom AST."""

    def __init__(self) -> None:
        self.report = OptimizationReport()

    def optimize(self, program: Program) -> OptimizationReport:
        """Analyze the program and return an optimization report."""
        for stmt in program.statements:
            self._analyze_node(stmt)
        return self.report

    def _analyze_node(self, node: ASTNode) -> None:
        """Recursively analyze a node for optimization opportunities."""
        if isinstance(node, ModelDefinition):
            self._analyze_model(node)

        elif isinstance(node, TrainBlock):
            self._analyze_train(node)

        elif isinstance(node, PipeExpression):
            self._analyze_pipe(node)

        elif isinstance(node, FunctionDef):
            for stmt in node.body:
                self._analyze_node(stmt)

        elif isinstance(node, LetStatement):
            if isinstance(node.value, BinaryOp):
                self._detect_arithmetic_fusion(node.value, node.line)
            self._analyze_node(node.value)

        elif isinstance(node, AssignStatement):
            if isinstance(node.value, PipeExpression):
                self._analyze_pipe(node.value)
            elif isinstance(node.value, BinaryOp):
                self._detect_arithmetic_fusion(node.value, node.line)
            self._analyze_node(node.value)

    def _analyze_model(self, model: ModelDefinition) -> None:
        """Estimate model parameters and memory."""
        total_params = 0

        for layer in model.layers:
            if isinstance(layer.layer_type, FunctionCall):
                call = layer.layer_type
                if isinstance(call.callee, Identifier) and call.callee.name == "Linear":
                    dims = [a for a in call.args if isinstance(a, NumberLiteral)]
                    if len(dims) >= 2:
                        in_dim = int(dims[0].value)
                        out_dim = int(dims[1].value)
                        params = in_dim * out_dim + out_dim  # weights + bias
                        total_params += params

        if total_params > 0:
            param_mb = (total_params * 4) / (1024 * 1024)
            training_mb = param_mb * 3  # params + grads + optimizer
            self.report.memory_estimates.append(MemoryEstimate(
                model_name=model.name,
                total_params=total_params,
                param_memory_mb=param_mb,
                training_memory_mb=training_mb,
                fp16_memory_mb=training_mb / 2,
            ))

        # Check methods for pipe fusion
        for method in model.methods:
            for stmt in method.body:
                self._analyze_node(stmt)

    def _analyze_train(self, train: TrainBlock) -> None:
        """Analyze training block for precision hints."""
        if "precision" in train.params:
            p = train.params["precision"]
            if isinstance(p, Identifier):
                self.report.precision_hints[train.model_name] = p.name

    def _analyze_pipe(self, pipe: PipeExpression) -> None:
        """Detect fusion opportunities in pipe chains."""
        if len(pipe.stages) >= 2:
            ops = [s.func_name for s in pipe.stages]
            self.report.fusions.append(FusionOpportunity(
                description=f"Pipe chain of {len(pipe.stages)} ops can be fused: {' |> '.join(ops)}",
                line=pipe.line,
                ops=ops,
            ))

    def _detect_arithmetic_fusion(self, node: BinaryOp, line: int) -> None:
        """Detect fusible arithmetic like x @ y + bias."""
        if node.op == "+" and isinstance(node.left, BinaryOp) and node.left.op == "@":
            self.report.fusions.append(FusionOpportunity(
                description="matmul + bias can be fused into a single kernel",
                line=line,
                ops=["matmul", "add"],
            ))
