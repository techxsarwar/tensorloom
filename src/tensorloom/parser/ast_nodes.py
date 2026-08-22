"""
TensorLoom AST Node Definitions.

Every construct in TensorLoom source code is represented as one of these
dataclass nodes after parsing.  The nodes form a tree rooted at `Program`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ═══════════════════════════════════════════════════════════════
#  Base
# ═══════════════════════════════════════════════════════════════

@dataclass
class ASTNode:
    """Base class for all AST nodes.  Carries source location."""
    line: int = 0
    column: int = 0


# ═══════════════════════════════════════════════════════════════
#  Top-Level
# ═══════════════════════════════════════════════════════════════

@dataclass
class Program(ASTNode):
    """Root node — a list of top-level statements."""
    statements: list[ASTNode] = field(default_factory=list)


@dataclass
class ImportStatement(ASTNode):
    """import std.io  →  module_path = ["std", "io"]"""
    module_path: list[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════
#  Literals
# ═══════════════════════════════════════════════════════════════

@dataclass
class NumberLiteral(ASTNode):
    """Numeric literal — integer or float."""
    value: float | int = 0
    is_integer: bool = False


@dataclass
class StringLiteral(ASTNode):
    """A plain string literal."""
    value: str = ""


@dataclass
class FStringLiteral(ASTNode):
    """An f-string like f"loss={val:.4f}".  `parts` alternates between
    StringLiteral (plain text) and arbitrary expressions."""
    parts: list[ASTNode] = field(default_factory=list)


@dataclass
class BooleanLiteral(ASTNode):
    value: bool = False


@dataclass
class NoneLiteral(ASTNode):
    pass


@dataclass
class ListLiteral(ASTNode):
    """A list literal: [1, 2, 3]"""
    elements: list[ASTNode] = field(default_factory=list)


@dataclass
class TensorLiteral(ASTNode):
    """tensor([1.0, 2.0], dtype=float32, device=gpu)"""
    elements: list[ASTNode] = field(default_factory=list)
    kwargs: dict[str, ASTNode] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════
#  Expressions
# ═══════════════════════════════════════════════════════════════

@dataclass
class Identifier(ASTNode):
    """A variable or name reference."""
    name: str = ""


@dataclass
class BinaryOp(ASTNode):
    """left op right — covers +, -, *, /, @, ==, !=, <, >, <=, >=, and, or"""
    op: str = ""
    left: ASTNode = field(default_factory=ASTNode)
    right: ASTNode = field(default_factory=ASTNode)


@dataclass
class UnaryOp(ASTNode):
    """-x, not x"""
    op: str = ""
    operand: ASTNode = field(default_factory=ASTNode)


@dataclass
class PipeExpression(ASTNode):
    """value |> func1 |> func2(args)
    
    Desugars to:  func2(func1(value), args)
    Each stage is a PipeStage capturing the function and its extra args.
    """
    value: ASTNode = field(default_factory=ASTNode)
    stages: list["PipeStage"] = field(default_factory=list)


@dataclass
class PipeStage(ASTNode):
    """A single stage in a pipe chain:  |> func(extra_args)"""
    func_name: str = ""
    args: list[ASTNode] = field(default_factory=list)
    kwargs: dict[str, ASTNode] = field(default_factory=dict)


@dataclass
class FunctionCall(ASTNode):
    """callee(arg1, arg2, key=val)"""
    callee: ASTNode = field(default_factory=ASTNode)
    args: list[ASTNode] = field(default_factory=list)
    kwargs: dict[str, ASTNode] = field(default_factory=dict)


@dataclass
class MemberAccess(ASTNode):
    """object.member"""
    object: ASTNode = field(default_factory=ASTNode)
    member: str = ""


@dataclass
class IndexAccess(ASTNode):
    """object[index]"""
    object: ASTNode = field(default_factory=ASTNode)
    index: ASTNode = field(default_factory=ASTNode)


# ═══════════════════════════════════════════════════════════════
#  Statements
# ═══════════════════════════════════════════════════════════════

@dataclass
class LetStatement(ASTNode):
    """let name = expr"""
    name: str = ""
    value: ASTNode = field(default_factory=ASTNode)
    type_hint: Optional[str] = None


@dataclass
class AssignStatement(ASTNode):
    """target = value  (reassignment, no 'let')"""
    target: ASTNode = field(default_factory=ASTNode)
    value: ASTNode = field(default_factory=ASTNode)


@dataclass
class ReturnStatement(ASTNode):
    """return expr"""
    value: Optional[ASTNode] = None


@dataclass
class ExpressionStatement(ASTNode):
    """A bare expression used as a statement, e.g. print(x)"""
    expression: ASTNode = field(default_factory=ASTNode)


@dataclass
class IfStatement(ASTNode):
    condition: ASTNode = field(default_factory=ASTNode)
    body: list[ASTNode] = field(default_factory=list)
    elif_clauses: list[tuple[ASTNode, list[ASTNode]]] = field(default_factory=list)
    else_body: list[ASTNode] = field(default_factory=list)


@dataclass
class ForLoop(ASTNode):
    variable: str = ""
    iterable: ASTNode = field(default_factory=ASTNode)
    body: list[ASTNode] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════
#  Functions & Parameters
# ═══════════════════════════════════════════════════════════════

@dataclass
class Parameter(ASTNode):
    """A function parameter:  name: TypeHint = default"""
    name: str = ""
    type_hint: Optional[str] = None
    default: Optional[ASTNode] = None


@dataclass
class FunctionDef(ASTNode):
    """fn name(params) -> ReturnType: body"""
    name: str = ""
    params: list[Parameter] = field(default_factory=list)
    return_type: Optional[str] = None
    body: list[ASTNode] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════
#  Model Definition
# ═══════════════════════════════════════════════════════════════

@dataclass
class LayerDeclaration(ASTNode):
    """layer name = LayerType(args)"""
    name: str = ""
    layer_type: ASTNode = field(default_factory=ASTNode)  # FunctionCall


@dataclass
class ModelDefinition(ASTNode):
    """model Name:
        layer ...
        fn forward ...
    """
    name: str = ""
    layers: list[LayerDeclaration] = field(default_factory=list)
    methods: list[FunctionDef] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════
#  Training Block
# ═══════════════════════════════════════════════════════════════

@dataclass
class TrainBlock(ASTNode):
    """train model_name on data_name:
        epochs = 10
        optimizer = Adam(lr=0.001)
        ...
    """
    model_name: str = ""
    data_name: str = ""
    params: dict[str, ASTNode] = field(default_factory=dict)
    callbacks: list["TrainCallback"] = field(default_factory=list)
    checkpoint_config: Optional["CheckpointConfig"] = None


@dataclass
class TrainCallback(ASTNode):
    """on epoch_end(metrics): body"""
    event: str = ""
    param_name: str = ""
    body: list[ASTNode] = field(default_factory=list)


@dataclass
class CheckpointConfig(ASTNode):
    """checkpoint every 2 epochs"""
    frequency: ASTNode = field(default_factory=ASTNode)
    unit: str = "epochs"


# ═══════════════════════════════════════════════════════════════
#  NML-Specific Nodes
# ═══════════════════════════════════════════════════════════════

@dataclass
class NMLModel(ASTNode):
    """@model Name:  (top-level NML definition)"""
    name: str = ""
    config: dict[str, ASTNode] = field(default_factory=dict)
    layers: list[LayerDeclaration] = field(default_factory=list)
    forward_body: list[ASTNode] = field(default_factory=list)
