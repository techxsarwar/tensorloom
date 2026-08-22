"""
TensorLoom Semantic Analyzer — Type checking and validation.

Performs basic semantic analysis on the AST:
  - Scope tracking (variable declarations / usage)
  - Type hint validation
  - Model/layer consistency checks
  - Training block config validation
"""
from __future__ import annotations

from dataclasses import dataclass, field

from tensorloom.parser.ast_nodes import (
    ASTNode,
    AssignStatement,
    ExpressionStatement,
    ForLoop,
    FunctionCall,
    FunctionDef,
    Identifier,
    IfStatement,
    ImportStatement,
    KernelDef,
    LetStatement,
    ModelDefinition,
    Program,
    ReturnStatement,
    TrainBlock,
)


@dataclass
class Symbol:
    """A symbol in the scope table."""
    name: str
    kind: str  # "variable", "model", "function", "layer", "import", "kernel"
    line: int = 0


@dataclass
class Scope:
    """A lexical scope with a symbol table."""
    symbols: dict[str, Symbol] = field(default_factory=dict)
    parent: "Scope | None" = None

    def define(self, name: str, kind: str, line: int = 0) -> None:
        self.symbols[name] = Symbol(name=name, kind=kind, line=line)

    def lookup(self, name: str) -> Symbol | None:
        if name in self.symbols:
            return self.symbols[name]
        if self.parent:
            return self.parent.lookup(name)
        return None


@dataclass
class AnalysisWarning:
    message: str
    line: int


@dataclass
class AnalysisError:
    message: str
    line: int


class TypeChecker:
    """Performs semantic analysis on a TensorLoom AST."""

    def __init__(self) -> None:
        self.errors: list[AnalysisError] = []
        self.warnings: list[AnalysisWarning] = []
        self.global_scope = Scope()
        self.current_scope = self.global_scope

        # Pre-define builtins
        for builtin in ("print", "len", "range", "tensor", "type", "list",
                         "int", "float", "str", "bool", "abs", "min", "max"):
            self.global_scope.define(builtin, "builtin")

    def analyze(self, program: Program) -> tuple[list[AnalysisError], list[AnalysisWarning]]:
        """Run analysis on the entire program. Returns (errors, warnings)."""
        for stmt in program.statements:
            self._analyze_statement(stmt)
        return self.errors, self.warnings

    def _analyze_statement(self, node: ASTNode) -> None:
        if isinstance(node, ImportStatement):
            module = ".".join(node.module_path)
            self.current_scope.define(node.module_path[-1], "import", node.line)

        elif isinstance(node, LetStatement):
            # Analyze the value expression first
            self._analyze_expression(node.value)
            self.current_scope.define(node.name, "variable", node.line)

        elif isinstance(node, AssignStatement):
            self._analyze_expression(node.value)
            if isinstance(node.target, Identifier):
                if not self.current_scope.lookup(node.target.name):
                    self.warnings.append(AnalysisWarning(
                        f"Assignment to undeclared variable '{node.target.name}' — use 'let' for first declaration",
                        node.line,
                    ))

        elif isinstance(node, ModelDefinition):
            self.current_scope.define(node.name, "model", node.line)
            # Create a scope for the model body
            model_scope = Scope(parent=self.current_scope)
            model_scope.define("self", "variable")
            prev_scope = self.current_scope
            self.current_scope = model_scope

            for layer in node.layers:
                model_scope.define(layer.name, "layer", layer.line)

            for method in node.methods:
                self._analyze_function_def(method)

            self.current_scope = prev_scope

        elif isinstance(node, TrainBlock):
            # Verify model exists
            if not self.current_scope.lookup(node.model_name):
                self.errors.append(AnalysisError(
                    f"Undefined model '{node.model_name}' in train block",
                    node.line,
                ))
            # Verify data exists
            if not self.current_scope.lookup(node.data_name):
                self.errors.append(AnalysisError(
                    f"Undefined data variable '{node.data_name}' in train block",
                    node.line,
                ))
            # Validate known params
            valid_params = {"epochs", "optimizer", "loss", "precision", "lr",
                            "batch_size", "shuffle", "checkpoint", "distributed"}
            for key in node.params:
                if key not in valid_params:
                    self.warnings.append(AnalysisWarning(
                        f"Unknown training parameter '{key}'",
                        node.line,
                    ))

        elif isinstance(node, KernelDef):
            # Register the kernel as a callable in the enclosing scope
            self.current_scope.define(node.name, "kernel", node.line)
            # Kernel bodies use pointer arithmetic — create a permissive scope
            kernel_scope = Scope(parent=self.current_scope)
            for param in node.params:
                kernel_scope.define(param.name, "variable", node.line)
            # Pre-define tl namespace as a known import
            kernel_scope.define("tl", "import", node.line)
            prev = self.current_scope
            self.current_scope = kernel_scope
            for stmt in node.body:
                self._analyze_statement(stmt)
            self.current_scope = prev

        elif isinstance(node, FunctionDef):
            self._analyze_function_def(node)

        elif isinstance(node, ReturnStatement):
            if node.value:
                self._analyze_expression(node.value)

        elif isinstance(node, IfStatement):
            self._analyze_expression(node.condition)
            for stmt in node.body:
                self._analyze_statement(stmt)
            for cond, body in node.elif_clauses:
                self._analyze_expression(cond)
                for stmt in body:
                    self._analyze_statement(stmt)
            for stmt in node.else_body:
                self._analyze_statement(stmt)

        elif isinstance(node, ForLoop):
            self._analyze_expression(node.iterable)
            loop_scope = Scope(parent=self.current_scope)
            loop_scope.define(node.variable, "variable", node.line)
            prev = self.current_scope
            self.current_scope = loop_scope
            for stmt in node.body:
                self._analyze_statement(stmt)
            self.current_scope = prev

        elif isinstance(node, ExpressionStatement):
            self._analyze_expression(node.expression)

    def _analyze_function_def(self, node: FunctionDef) -> None:
        self.current_scope.define(node.name, "function", node.line)
        func_scope = Scope(parent=self.current_scope)
        for param in node.params:
            func_scope.define(param.name, "variable", node.line)
        prev = self.current_scope
        self.current_scope = func_scope
        for stmt in node.body:
            self._analyze_statement(stmt)
        self.current_scope = prev

    def _analyze_expression(self, node: ASTNode) -> None:
        """Basic expression analysis — check variable references exist."""
        if isinstance(node, Identifier):
            if node.name != "self" and not self.current_scope.lookup(node.name):
                # Don't error on common builtins or type names
                known_names = {"relu", "sigmoid", "tanh", "softmax", "gelu", "silu",
                               "dropout", "flatten", "Adam", "AdamW", "SGD",
                               "CrossEntropy", "MSE", "Linear", "Conv2d",
                               "float32", "float16", "gpu", "cpu", "true", "false",
                               "fp16", "bf16", "mnist", "MNIST"}
                if node.name not in known_names:
                    pass  # Soft check — don't error in Phase 1

        elif isinstance(node, FunctionCall):
            self._analyze_expression(node.callee)
            for arg in node.args:
                self._analyze_expression(arg)
            for val in node.kwargs.values():
                self._analyze_expression(val)
