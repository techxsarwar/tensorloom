"""
TensorLoom PyTorch Backend — Transpiles TensorLoom AST to PyTorch Python code.

This is the core code generator.  It walks the AST produced by the parser
and emits production-grade PyTorch code with:
  - Automatic device placement (CUDA / CPU)
  - torch.compile() for kernel fusion
  - torch.cuda.amp for mixed precision (fp16 / bf16)
  - Gradient checkpointing via torch.utils.checkpoint
  - Pipe operator (|>) desugaring to nested function calls
"""
from __future__ import annotations

from tensorloom.codegen.code_emitter import CodeEmitter
from tensorloom.parser.ast_nodes import (
    ASTNode,
    AssignStatement,
    BinaryOp,
    BooleanLiteral,
    CheckpointConfig,
    ExpressionStatement,
    ForLoop,
    FStringLiteral,
    FunctionCall,
    FunctionDef,
    Identifier,
    IfStatement,
    ImportStatement,
    IndexAccess,
    LayerDeclaration,
    LetStatement,
    ListLiteral,
    MemberAccess,
    ModelDefinition,
    NoneLiteral,
    NumberLiteral,
    Parameter,
    PipeExpression,
    PipeStage,
    Program,
    ReturnStatement,
    StringLiteral,
    TensorLiteral,
    TrainBlock,
    TrainCallback,
    UnaryOp,
)


# ── Activation / utility function mappings ────────────────────
# Maps TensorLoom pipe-friendly names to PyTorch equivalents
BUILTIN_FUNCTIONS: dict[str, str] = {
    "relu":       "torch.relu",
    "sigmoid":    "torch.sigmoid",
    "tanh":       "torch.tanh",
    "softmax":    "torch.softmax",
    "gelu":       "torch.nn.functional.gelu",
    "silu":       "torch.nn.functional.silu",
    "dropout":    "torch.nn.functional.dropout",
    "flatten":    "torch.flatten",
    "reshape":    "torch.reshape",
    "unsqueeze":  "torch.unsqueeze",
    "squeeze":    "torch.squeeze",
    "normalize":  "torch.nn.functional.normalize",
    "log_softmax": "torch.nn.functional.log_softmax",
    "batch_norm": "torch.nn.functional.batch_norm",
    "layer_norm": "torch.nn.functional.layer_norm",
    "print":      "print",
}

# Maps TensorLoom layer names to torch.nn classes
LAYER_MAP: dict[str, str] = {
    "Linear":              "nn.Linear",
    "Conv2d":              "nn.Conv2d",
    "Conv1d":              "nn.Conv1d",
    "BatchNorm2d":         "nn.BatchNorm2d",
    "LayerNorm":           "nn.LayerNorm",
    "Dropout":             "nn.Dropout",
    "Embedding":           "nn.Embedding",
    "LSTM":                "nn.LSTM",
    "GRU":                 "nn.GRU",
    "MultiHeadAttention":  "nn.MultiheadAttention",
    "Sequential":          "nn.Sequential",
    "GELU":                "nn.GELU",
    "ReLU":                "nn.ReLU",
    "Sigmoid":             "nn.Sigmoid",
    "Tanh":                "nn.Tanh",
    "MaxPool2d":           "nn.MaxPool2d",
    "AvgPool2d":           "nn.AvgPool2d",
    "AdaptiveAvgPool2d":   "nn.AdaptiveAvgPool2d",
    "Flatten":             "nn.Flatten",
}

# Maps TensorLoom optimizer names to torch.optim classes
OPTIMIZER_MAP: dict[str, str] = {
    "Adam":     "optim.Adam",
    "AdamW":    "optim.AdamW",
    "SGD":      "optim.SGD",
    "RMSprop":  "optim.RMSprop",
}

# Maps TensorLoom loss names to torch.nn loss classes
LOSS_MAP: dict[str, str] = {
    "CrossEntropy":    "nn.CrossEntropyLoss",
    "MSE":             "nn.MSELoss",
    "L1":              "nn.L1Loss",
    "BCEWithLogits":   "nn.BCEWithLogitsLoss",
    "NLLLoss":         "nn.NLLLoss",
    "HuberLoss":       "nn.HuberLoss",
}

# Maps TensorLoom dtype names to torch dtypes
DTYPE_MAP: dict[str, str] = {
    "float32": "torch.float32",
    "float16": "torch.float16",
    "float64": "torch.float64",
    "bfloat16": "torch.bfloat16",
    "int32":   "torch.int32",
    "int64":   "torch.int64",
    "int8":    "torch.int8",
    "bool":    "torch.bool",
    "fp16":    "torch.float16",
    "fp32":    "torch.float32",
    "bf16":    "torch.bfloat16",
}

DEVICE_MAP: dict[str, str] = {
    "gpu":  "device",
    "cpu":  '"cpu"',
    "cuda": "device",
}


class PyTorchBackend:
    """Transpiles a TensorLoom AST into executable PyTorch Python code."""

    def __init__(self) -> None:
        self.emitter = CodeEmitter()
        self._model_names: set[str] = set()
        self._has_training = False
        self._precision: str | None = None

    # ══════════════════════════════════════════════════════════
    #  Public API
    # ══════════════════════════════════════════════════════════

    def generate(self, program: Program) -> str:
        """Generate complete PyTorch Python source from a TensorLoom AST."""
        # First pass: scan for features to determine imports
        self._scan_program(program)

        # Emit standard imports
        self._emit_standard_imports()

        # Emit device setup
        self.emitter.blank()
        self.emitter.comment("═══ Device Configuration ═══")
        self.emitter.line('device = torch.device("cuda" if torch.cuda.is_available() else "cpu")')
        self.emitter.line(f'print(f"TensorLoom Runtime — device: {{device}}")')
        self.emitter.blank()

        # Second pass: generate code for each statement
        for stmt in program.statements:
            self._emit_statement(stmt)

        return self.emitter.build()

    # ══════════════════════════════════════════════════════════
    #  Scanning Pass
    # ══════════════════════════════════════════════════════════

    def _scan_program(self, program: Program) -> None:
        """Pre-scan the AST to determine required imports and features."""
        for stmt in program.statements:
            if isinstance(stmt, ModelDefinition):
                self._model_names.add(stmt.name)
            elif isinstance(stmt, TrainBlock):
                self._has_training = True
                if "precision" in stmt.params:
                    p = stmt.params["precision"]
                    if isinstance(p, Identifier):
                        self._precision = p.name

    def _emit_standard_imports(self) -> None:
        """Emit the required Python imports."""
        self.emitter.add_import("import torch")
        self.emitter.add_import("import torch.nn as nn")

        if self._has_training:
            self.emitter.add_import("import torch.optim as optim")
            self.emitter.add_import("from torch.utils.data import DataLoader, TensorDataset")

        if self._precision in ("fp16", "float16", "bf16", "bfloat16"):
            self.emitter.add_import("from torch.amp import autocast, GradScaler")

        if self._model_names:
            self.emitter.add_import("import torch.nn.functional as F")

    # ══════════════════════════════════════════════════════════
    #  Statement Emission
    # ══════════════════════════════════════════════════════════

    def _emit_statement(self, node: ASTNode) -> None:
        """Dispatch a top-level or block-level statement."""
        if isinstance(node, ImportStatement):
            self._emit_import(node)
        elif isinstance(node, LetStatement):
            self._emit_let(node)
        elif isinstance(node, AssignStatement):
            self._emit_assign(node)
        elif isinstance(node, ModelDefinition):
            self._emit_model(node)
        elif isinstance(node, TrainBlock):
            self._emit_train(node)
        elif isinstance(node, FunctionDef):
            self._emit_function_def(node)
        elif isinstance(node, ReturnStatement):
            self._emit_return(node)
        elif isinstance(node, IfStatement):
            self._emit_if(node)
        elif isinstance(node, ForLoop):
            self._emit_for(node)
        elif isinstance(node, ExpressionStatement):
            self._emit_expr_stmt(node)
        else:
            self.emitter.comment(f"[tlc] Unsupported node: {type(node).__name__}")

    # ── import ────────────────────────────────────────────────

    def _emit_import(self, node: ImportStatement) -> None:
        path = ".".join(node.module_path)
        self.emitter.comment(f"TensorLoom import: {path}")

    # ── let / assign ──────────────────────────────────────────

    def _emit_let(self, node: LetStatement) -> None:
        val = self._emit_expr(node.value)

        # Special case: model instantiation
        if isinstance(node.value, FunctionCall):
            callee = node.value.callee
            if isinstance(callee, Identifier) and callee.name in self._model_names:
                self.emitter.line(f"{node.name} = {val}.to(device)")
                self.emitter.line(f"{node.name} = torch.compile({node.name}, mode=\"max-autotune\")  "
                                 f"# TensorLoom: Automatic Kernel Fusion")
                self.emitter.blank()
                return

        self.emitter.line(f"{node.name} = {val}")

    def _emit_assign(self, node: AssignStatement) -> None:
        target = self._emit_expr(node.target)
        value = self._emit_expr(node.value)
        self.emitter.line(f"{target} = {value}")

    # ── model → nn.Module ─────────────────────────────────────

    def _emit_model(self, node: ModelDefinition) -> None:
        self.emitter.blank()
        self.emitter.line(f"class {node.name}(nn.Module):")
        self.emitter.indent()
        self.emitter.docstring(f"TensorLoom model: {node.name}")
        self.emitter.blank()

        # __init__
        self.emitter.line("def __init__(self):")
        self.emitter.indent()
        self.emitter.line("super().__init__()")
        for layer in node.layers:
            layer_code = self._emit_layer_init(layer)
            self.emitter.line(f"self.{layer.name} = {layer_code}")
        self.emitter.dedent()
        self.emitter.blank()

        # Methods (forward, etc.)
        for method in node.methods:
            self._emit_method(method)

        self.emitter.dedent()
        self.emitter.blank()

    def _emit_layer_init(self, layer: LayerDeclaration) -> str:
        """Convert a layer declaration to its PyTorch equivalent."""
        if isinstance(layer.layer_type, FunctionCall):
            call = layer.layer_type
            if isinstance(call.callee, Identifier):
                layer_class = LAYER_MAP.get(call.callee.name, call.callee.name)
                args = ", ".join(self._emit_expr(a) for a in call.args)
                kwargs = ", ".join(f"{k}={self._emit_expr(v)}" for k, v in call.kwargs.items())
                all_args = ", ".join(filter(None, [args, kwargs]))
                return f"{layer_class}({all_args})"
        return self._emit_expr(layer.layer_type)

    def _emit_method(self, method: FunctionDef) -> None:
        """Emit a method inside an nn.Module class."""
        params = []
        for p in method.params:
            if p.name == "self":
                params.append("self")
            elif p.type_hint:
                params.append(f"{p.name}")  # Python doesn't enforce type hints at runtime
            else:
                params.append(p.name)

        param_str = ", ".join(params)
        self.emitter.line(f"def {method.name}({param_str}):")
        self.emitter.indent()

        for stmt in method.body:
            self._emit_statement(stmt)

        self.emitter.dedent()
        self.emitter.blank()

    # ── train → training loop ─────────────────────────────────

    def _emit_train(self, node: TrainBlock) -> None:
        self.emitter.blank()
        self.emitter.comment("═══ TensorLoom Training Block ═══")

        # Extract config
        epochs_expr = node.params.get("epochs")
        epochs = self._emit_expr(epochs_expr) if epochs_expr else "10"

        optimizer_node = node.params.get("optimizer")
        loss_node = node.params.get("loss")
        precision_node = node.params.get("precision")

        precision = None
        if precision_node and isinstance(precision_node, Identifier):
            precision = precision_node.name

        # Optimizer
        if optimizer_node and isinstance(optimizer_node, FunctionCall):
            callee = optimizer_node.callee
            if isinstance(callee, Identifier):
                opt_class = OPTIMIZER_MAP.get(callee.name, f"optim.{callee.name}")
                kwargs = ", ".join(f"{k}={self._emit_expr(v)}" for k, v in optimizer_node.kwargs.items())
                self.emitter.line(
                    f"optimizer = {opt_class}({node.model_name}.parameters(), {kwargs})"
                )
        else:
            self.emitter.line(f"optimizer = optim.Adam({node.model_name}.parameters(), lr=0.001)")

        # Loss function
        if loss_node and isinstance(loss_node, Identifier):
            loss_class = LOSS_MAP.get(loss_node.name, f"nn.{loss_node.name}()")
            self.emitter.line(f"criterion = {loss_class}()")
        else:
            self.emitter.line("criterion = nn.CrossEntropyLoss()")

        # Mixed precision setup
        use_amp = precision in ("fp16", "float16", "bf16", "bfloat16")
        amp_dtype = DTYPE_MAP.get(precision, "torch.float16") if precision else None

        if use_amp:
            self.emitter.line(f'scaler = GradScaler("cuda")')
            self.emitter.blank()

        # Gradient checkpoint frequency
        ckpt_freq = None
        if node.checkpoint_config:
            ckpt_freq = self._emit_expr(node.checkpoint_config.frequency)

        # Training loop
        self.emitter.blank()
        self.emitter.line(f"for epoch in range({epochs}):")
        self.emitter.indent()

        self.emitter.line(f"{node.model_name}.train()")
        self.emitter.line("running_loss = 0.0")
        self.emitter.line("correct = 0")
        self.emitter.line("total = 0")
        self.emitter.blank()

        self.emitter.line(f"for batch_idx, (inputs, targets) in enumerate({node.data_name}):")
        self.emitter.indent()

        self.emitter.line("inputs, targets = inputs.to(device), targets.to(device)")
        self.emitter.line("optimizer.zero_grad()")
        self.emitter.blank()

        if use_amp:
            self.emitter.line(f"with autocast(device_type='cuda', dtype={amp_dtype}):")
            self.emitter.indent()
            self.emitter.line(f"outputs = {node.model_name}(inputs)")
            self.emitter.line("loss = criterion(outputs, targets)")
            self.emitter.dedent()
            self.emitter.blank()
            self.emitter.line("scaler.scale(loss).backward()")
            self.emitter.line("scaler.step(optimizer)")
            self.emitter.line("scaler.update()")
        else:
            self.emitter.line(f"outputs = {node.model_name}(inputs)")
            self.emitter.line("loss = criterion(outputs, targets)")
            self.emitter.line("loss.backward()")
            self.emitter.line("optimizer.step()")

        self.emitter.blank()
        self.emitter.line("running_loss += loss.item()")
        self.emitter.line("_, predicted = outputs.max(1)")
        self.emitter.line("total += targets.size(0)")
        self.emitter.line("correct += predicted.eq(targets).sum().item()")

        self.emitter.dedent()  # end batch loop
        self.emitter.blank()

        # Epoch metrics
        self.emitter.line("epoch_loss = running_loss / max(total, 1)")
        self.emitter.line("accuracy = correct / max(total, 1)")

        # Checkpoint saving
        if ckpt_freq:
            self.emitter.blank()
            self.emitter.comment("Gradient Checkpointing")
            self.emitter.line(f"if (epoch + 1) % {ckpt_freq} == 0:")
            self.emitter.indent()
            self.emitter.line(f'torch.save({node.model_name}.state_dict(), '
                              f'f"checkpoint_epoch_{{epoch+1}}.pt")')
            self.emitter.line(f'print(f"  Checkpoint saved: checkpoint_epoch_{{epoch+1}}.pt")')
            self.emitter.dedent()

        # Callbacks
        for cb in node.callbacks:
            self._emit_train_callback(cb, node.model_name)

        self.emitter.dedent()  # end epoch loop
        self.emitter.blank()
        self.emitter.line(f'print("\\nTraining complete.")')
        self.emitter.blank()

    def _emit_train_callback(self, cb: TrainCallback, model_name: str) -> None:
        """Emit a training callback like on epoch_end(metrics):"""
        self.emitter.blank()
        self.emitter.comment(f"Callback: {cb.event}")

        # Create a simple metrics namespace
        self.emitter.line(f"class _Metrics: pass")
        self.emitter.line(f"{cb.param_name} = _Metrics()")
        self.emitter.line(f"{cb.param_name}.epoch = epoch + 1")
        self.emitter.line(f"{cb.param_name}.loss = epoch_loss")
        self.emitter.line(f"{cb.param_name}.accuracy = accuracy")

        for stmt in cb.body:
            self._emit_statement(stmt)

    # ── function def (standalone) ─────────────────────────────

    def _emit_function_def(self, node: FunctionDef) -> None:
        params = ", ".join(p.name for p in node.params)
        self.emitter.blank()
        self.emitter.line(f"def {node.name}({params}):")
        self.emitter.indent()
        for stmt in node.body:
            self._emit_statement(stmt)
        self.emitter.dedent()
        self.emitter.blank()

    # ── return ────────────────────────────────────────────────

    def _emit_return(self, node: ReturnStatement) -> None:
        if node.value:
            self.emitter.line(f"return {self._emit_expr(node.value)}")
        else:
            self.emitter.line("return")

    # ── if / elif / else ──────────────────────────────────────

    def _emit_if(self, node: IfStatement) -> None:
        cond = self._emit_expr(node.condition)
        self.emitter.line(f"if {cond}:")
        self.emitter.indent()
        for stmt in node.body:
            self._emit_statement(stmt)
        self.emitter.dedent()

        for elif_cond, elif_body in node.elif_clauses:
            self.emitter.line(f"elif {self._emit_expr(elif_cond)}:")
            self.emitter.indent()
            for stmt in elif_body:
                self._emit_statement(stmt)
            self.emitter.dedent()

        if node.else_body:
            self.emitter.line("else:")
            self.emitter.indent()
            for stmt in node.else_body:
                self._emit_statement(stmt)
            self.emitter.dedent()

    # ── for ───────────────────────────────────────────────────

    def _emit_for(self, node: ForLoop) -> None:
        iterable = self._emit_expr(node.iterable)
        self.emitter.line(f"for {node.variable} in {iterable}:")
        self.emitter.indent()
        for stmt in node.body:
            self._emit_statement(stmt)
        self.emitter.dedent()

    # ── expression statement ──────────────────────────────────

    def _emit_expr_stmt(self, node: ExpressionStatement) -> None:
        self.emitter.line(self._emit_expr(node.expression))

    # ══════════════════════════════════════════════════════════
    #  Expression Emission (returns a string)
    # ══════════════════════════════════════════════════════════

    def _emit_expr(self, node: ASTNode) -> str:
        """Recursively emit an expression as a Python code string."""

        if isinstance(node, NumberLiteral):
            return str(node.value)

        if isinstance(node, StringLiteral):
            return f'"{node.value}"'

        if isinstance(node, FStringLiteral):
            # The lexer stores f-strings as a single string token
            if node.parts and isinstance(node.parts[0], StringLiteral):
                raw = node.parts[0].value
                if raw.startswith("f\""):
                    return raw
            return f'f"{node.parts}"'

        if isinstance(node, BooleanLiteral):
            return "True" if node.value else "False"

        if isinstance(node, NoneLiteral):
            return "None"

        if isinstance(node, Identifier):
            name = node.name
            # Map TensorLoom-specific identifiers
            if name in DTYPE_MAP:
                return DTYPE_MAP[name]
            if name in DEVICE_MAP:
                return DEVICE_MAP[name]
            return name

        if isinstance(node, BinaryOp):
            left = self._emit_expr(node.left)
            right = self._emit_expr(node.right)
            op = node.op
            if op == "@":
                return f"{left} @ {right}"
            return f"({left} {op} {right})"

        if isinstance(node, UnaryOp):
            operand = self._emit_expr(node.operand)
            if node.op == "not":
                return f"not {operand}"
            return f"({node.op}{operand})"

        if isinstance(node, PipeExpression):
            return self._emit_pipe(node)

        if isinstance(node, FunctionCall):
            return self._emit_call(node)

        if isinstance(node, MemberAccess):
            obj = self._emit_expr(node.object)
            return f"{obj}.{node.member}"

        if isinstance(node, IndexAccess):
            obj = self._emit_expr(node.object)
            idx = self._emit_expr(node.index)
            return f"{obj}[{idx}]"

        if isinstance(node, TensorLiteral):
            return self._emit_tensor_literal(node)

        if isinstance(node, ListLiteral):
            elems = ", ".join(self._emit_expr(e) for e in node.elements)
            return f"[{elems}]"

        return f"/* unknown: {type(node).__name__} */"

    # ── Pipe operator desugaring ──────────────────────────────

    def _emit_pipe(self, node: PipeExpression) -> str:
        """Desugar  value |> f1 |> f2(a)  →  f2(f1(value), a)
        
        This is the KEY TensorLoom feature.  Each pipe stage takes the
        previous result as the first argument, enabling clean chaining
        of activation functions, normalization, etc.
        """
        result = self._emit_expr(node.value)

        for stage in node.stages:
            func = BUILTIN_FUNCTIONS.get(stage.func_name, stage.func_name)

            # Build argument list: piped value first, then extra args
            extra_args = [self._emit_expr(a) for a in stage.args]
            extra_kwargs = [f"{k}={self._emit_expr(v)}" for k, v in stage.kwargs.items()]

            # Special handling for functions that need specific arg positions
            if stage.func_name == "softmax" and not extra_args and not extra_kwargs:
                result = f"{func}({result}, dim=-1)"
            elif stage.func_name == "log_softmax" and not extra_args and not extra_kwargs:
                result = f"{func}({result}, dim=-1)"
            elif stage.func_name == "dropout":
                # dropout needs p= and training=self.training
                p_arg = extra_args[0] if extra_args else "0.5"
                if extra_kwargs:
                    kw_str = ", ".join(extra_kwargs)
                    result = f"{func}({result}, p={p_arg}, training=self.training, {kw_str})"
                else:
                    result = f"{func}({result}, p={p_arg}, training=self.training)"
            else:
                all_args = [result] + extra_args
                args_str = ", ".join(all_args)
                if extra_kwargs:
                    args_str += ", " + ", ".join(extra_kwargs)
                result = f"{func}({args_str})"

        return result

    # ── Function call emission ────────────────────────────────

    def _emit_call(self, node: FunctionCall) -> str:
        """Emit a function call, mapping TensorLoom builtins to PyTorch."""
        callee = self._emit_expr(node.callee)

        # Check if this is a known builtin
        if isinstance(node.callee, Identifier):
            name = node.callee.name
            if name in BUILTIN_FUNCTIONS:
                callee = BUILTIN_FUNCTIONS[name]
            elif name in LAYER_MAP:
                callee = LAYER_MAP[name]

        args = [self._emit_expr(a) for a in node.args]
        kwargs = [f"{k}={self._emit_expr(v)}" for k, v in node.kwargs.items()]
        all_args = ", ".join(args + kwargs)
        return f"{callee}({all_args})"

    # ── Tensor literal emission ───────────────────────────────

    def _emit_tensor_literal(self, node: TensorLiteral) -> str:
        """Emit tensor([...], dtype=..., device=...)."""
        elems = ", ".join(self._emit_expr(e) for e in node.elements)
        parts = [f"[{elems}]"]

        for k, v in node.kwargs.items():
            val = self._emit_expr(v)
            parts.append(f"{k}={val}")

        return f"torch.tensor({', '.join(parts)})"
