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
    KernelDef,
    KernelParam,
    LayerDeclaration,
    LetStatement,
    ListLiteral,
    MemberAccess,
    ModelDefinition,
    NMLModel,
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

    def __init__(self, source_dir: str = ".") -> None:
        self.emitter = CodeEmitter()
        self.source_dir = source_dir  # directory for resolving .nml imports
        self._model_names: set[str] = set()
        self._has_training = False
        self._precision: str | None = None
        self._checkpointed_models: set[str] = set()  # models needing gradient checkpointing
        self._has_checkpoint = False
        self._is_distributed = False  # DDP mode
        self._distributed_models: set[str] = set()  # instance names that need DDP wrapping
        self._instance_to_class: dict[str, str] = {}  # { "net": "Net" }
        self._has_kernels = False  # Triton kernel mode
        self._kernel_names: set[str] = set()  # kernel function names
        self._nml_imports: dict[str, str] = {}  # { alias: nml_file_path }

    # ══════════════════════════════════════════════════════════
    #  Public API
    # ══════════════════════════════════════════════════════════

    def generate(self, program: Program) -> str:
        """Generate complete PyTorch Python source from a TensorLoom AST."""
        # First pass: scan for features to determine imports
        self._scan_program(program)

        # Emit standard imports
        self._emit_standard_imports()

        if self._is_distributed:
            return self._generate_distributed(program)

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

    def _generate_distributed(self, program: Program) -> str:
        """Generate DDP-wrapped PyTorch code with setup/cleanup/torchrun entry."""
        # ── DDP utility functions ──
        self.emitter.blank()
        self.emitter.blank()
        self.emitter.comment("═══ TensorLoom Distributed Data Parallel (DDP) ═══")
        self.emitter.blank()

        # setup_ddp()
        self.emitter.line("def setup_ddp():")
        self.emitter.indent()
        self.emitter.docstring("Initialize distributed process group. Auto-injected by TensorLoom.")
        self.emitter.line('dist.init_process_group(backend="nccl", init_method="env://")')
        self.emitter.line('local_rank = int(os.environ["LOCAL_RANK"])')
        self.emitter.line("torch.cuda.set_device(local_rank)")
        self.emitter.line("return local_rank")
        self.emitter.dedent()
        self.emitter.blank()

        # cleanup_ddp()
        self.emitter.line("def cleanup_ddp():")
        self.emitter.indent()
        self.emitter.docstring("Tear down distributed process group.")
        self.emitter.line("dist.destroy_process_group()")
        self.emitter.dedent()
        self.emitter.blank()

        # Emit model definitions at module level (outside train_distributed)
        for stmt in program.statements:
            if isinstance(stmt, ModelDefinition):
                self._emit_statement(stmt)
            elif isinstance(stmt, ImportStatement):
                self._emit_statement(stmt)

        # train_distributed() function
        self.emitter.blank()
        self.emitter.line("def train_distributed():")
        self.emitter.indent()
        self.emitter.docstring("Main distributed training entry point. Launch with torchrun.")
        self.emitter.line("local_rank = setup_ddp()")
        self.emitter.line(f'print(f"TensorLoom DDP -- rank={{dist.get_rank()}}/{{dist.get_world_size()}} on GPU {{local_rank}}")')
        self.emitter.blank()

        # Emit non-model, non-train statements inside the function
        for stmt in program.statements:
            if isinstance(stmt, (ModelDefinition, ImportStatement)):
                continue  # already emitted at module level
            elif isinstance(stmt, TrainBlock):
                self._emit_train_distributed(stmt)
            elif isinstance(stmt, LetStatement):
                # Model instantiation needs DDP wrapping
                if isinstance(stmt.value, FunctionCall):
                    callee = stmt.value.callee
                    if isinstance(callee, Identifier) and callee.name in self._model_names:
                        self._emit_let_distributed(stmt)
                        continue
                self._emit_statement(stmt)
            else:
                self._emit_statement(stmt)

        self.emitter.blank()
        self.emitter.line("cleanup_ddp()")
        self.emitter.dedent()
        self.emitter.blank()

        # torchrun entry point
        self.emitter.blank()
        self.emitter.comment("═══ Entry Point ═══")
        self.emitter.comment('Launch: torchrun --nproc_per_node=NUM_GPUS script.py')
        self.emitter.line('if __name__ == "__main__":')
        self.emitter.indent()
        self.emitter.line('if "LOCAL_RANK" in os.environ:')
        self.emitter.indent()
        self.emitter.line("train_distributed()")
        self.emitter.dedent()
        self.emitter.line("else:")
        self.emitter.indent()
        self.emitter.line('print("[TensorLoom] Launch with: torchrun --nproc_per_node=<NUM_GPUS> " + __file__)')
        self.emitter.line('print("             Falling back to single-GPU mode.")')
        self.emitter.line('# Single-GPU fallback')
        self.emitter.line('os.environ["LOCAL_RANK"] = "0"')
        self.emitter.line('os.environ["RANK"] = "0"')
        self.emitter.line('os.environ["WORLD_SIZE"] = "1"')
        self.emitter.line('os.environ["MASTER_ADDR"] = "localhost"')
        self.emitter.line('os.environ["MASTER_PORT"] = "29500"')
        self.emitter.line("train_distributed()")
        self.emitter.dedent()
        self.emitter.dedent()
        self.emitter.blank()

        return self.emitter.build()

    def _emit_let_distributed(self, node: LetStatement) -> None:
        """Emit model instantiation with DDP wrapping."""
        val = self._emit_expr(node.value)
        self.emitter.comment(f"Model instantiation with DDP wrapping")
        self.emitter.line(f"{node.name} = {val}.to(local_rank)")
        self.emitter.line(f"{node.name} = torch.compile({node.name}, mode=\"max-autotune\")  "
                          f"# TensorLoom: Automatic Kernel Fusion")
        self.emitter.line(f"{node.name} = DDP({node.name}, device_ids=[local_rank])")
        self.emitter.blank()

    def _emit_train_distributed(self, node: TrainBlock) -> None:
        """Emit a DDP-aware training loop with DistributedSampler."""
        self.emitter.blank()
        self.emitter.comment("═══ Distributed Training Loop ═══")

        # Extract config
        epochs_expr = node.params.get("epochs")
        epochs = self._emit_expr(epochs_expr) if epochs_expr else "10"

        optimizer_node = node.params.get("optimizer")
        loss_node = node.params.get("loss")
        precision_node = node.params.get("precision")
        batch_size_node = node.params.get("batch_size")

        batch_size = "64"
        if batch_size_node:
            batch_size = self._emit_expr(batch_size_node)

        precision = None
        if precision_node and isinstance(precision_node, Identifier):
            precision = precision_node.name

        # DistributedSampler setup
        self.emitter.comment("Distributed sampler — ensures each GPU gets unique data slices")
        self.emitter.line(f"sampler = DistributedSampler(")
        self.emitter.indent()
        self.emitter.line(f"{node.data_name},")
        self.emitter.line(f"num_replicas=dist.get_world_size(),")
        self.emitter.line(f"rank=dist.get_rank(),")
        self.emitter.line(f"shuffle=True,")
        self.emitter.dedent()
        self.emitter.line(")")
        self.emitter.line(f"loader = DataLoader({node.data_name}, batch_size={batch_size}, sampler=sampler)")
        self.emitter.blank()

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

        # Loss
        if loss_node and isinstance(loss_node, Identifier):
            loss_class = LOSS_MAP.get(loss_node.name, f"nn.{loss_node.name}()")
            self.emitter.line(f"criterion = {loss_class}()")
        else:
            self.emitter.line("criterion = nn.CrossEntropyLoss()")

        # Mixed precision
        use_amp = precision in ("fp16", "float16", "bf16", "bfloat16")
        amp_dtype = DTYPE_MAP.get(precision, "torch.float16") if precision else None

        if use_amp:
            self.emitter.line(f'scaler = GradScaler("cuda")')

        # Checkpoint frequency
        ckpt_freq = None
        if node.checkpoint_config:
            ckpt_freq = self._emit_expr(node.checkpoint_config.frequency)

        # Training loop
        self.emitter.blank()
        self.emitter.line(f"for epoch in range({epochs}):")
        self.emitter.indent()

        # CRITICAL: set_epoch on sampler to reshuffle data per epoch across ranks
        self.emitter.comment("Critical: re-shuffle data across ranks each epoch")
        self.emitter.line("sampler.set_epoch(epoch)")
        self.emitter.blank()

        self.emitter.line(f"{node.model_name}.train()")
        self.emitter.line("running_loss = 0.0")
        self.emitter.line("correct = 0")
        self.emitter.line("total = 0")
        self.emitter.blank()

        self.emitter.line("for batch_idx, (inputs, targets) in enumerate(loader):")
        self.emitter.indent()

        self.emitter.line("inputs, targets = inputs.to(local_rank), targets.to(local_rank)")
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

        # Weight snapshot — only rank 0 saves to avoid file conflicts
        if ckpt_freq:
            self.emitter.blank()
            self.emitter.comment("Weight Snapshot — only rank 0 saves")
            self.emitter.line(f"if dist.get_rank() == 0 and (epoch + 1) % {ckpt_freq} == 0:")
            self.emitter.indent()
            self.emitter.line(f'torch.save({node.model_name}.module.state_dict(), '
                              f'f"checkpoint_epoch_{{epoch+1}}.pt")')
            self.emitter.line(f'print(f"  [snapshot] Saved checkpoint_epoch_{{epoch+1}}.pt")')
            self.emitter.dedent()

        # Callbacks — only rank 0 prints to avoid duplicated output
        for cb in node.callbacks:
            self.emitter.blank()
            self.emitter.line("if dist.get_rank() == 0:")
            self.emitter.indent()
            self._emit_train_callback(cb, node.model_name)
            self.emitter.dedent()

        self.emitter.dedent()  # end epoch loop
        self.emitter.blank()

        self.emitter.line("if dist.get_rank() == 0:")
        self.emitter.indent()
        self.emitter.line('print("\\nDistributed training complete.")')
        self.emitter.dedent()

    # ══════════════════════════════════════════════════════════
    #  Scanning Pass
    # ══════════════════════════════════════════════════════════

    def _scan_program(self, program: Program) -> None:
        """Pre-scan the AST to determine required imports and features."""
        for stmt in program.statements:
            if isinstance(stmt, KernelDef):
                self._has_kernels = True
                self._kernel_names.add(stmt.name)
            elif isinstance(stmt, NMLModel):
                self._model_names.add(stmt.name)
            elif isinstance(stmt, ModelDefinition):
                self._model_names.add(stmt.name)
            elif isinstance(stmt, ImportStatement) and stmt.is_nml and stmt.alias:
                # Cross-file NML import: register alias as a model name
                nml_filename = ".".join(stmt.module_path[:-1]) + ".nml"
                self._nml_imports[stmt.alias] = nml_filename
                self._model_names.add(stmt.alias)
            elif isinstance(stmt, LetStatement):
                # Detect model instantiation: let net = Net()
                if isinstance(stmt.value, FunctionCall):
                    callee = stmt.value.callee
                    if isinstance(callee, Identifier) and callee.name in self._model_names:
                        self._instance_to_class[stmt.name] = callee.name
            elif isinstance(stmt, TrainBlock):
                self._has_training = True
                if "precision" in stmt.params:
                    p = stmt.params["precision"]
                    if isinstance(p, Identifier):
                        self._precision = p.name
                # Detect gradient checkpointing — resolve instance to class
                if stmt.checkpoint_config is not None:
                    self._has_checkpoint = True
                    class_name = self._instance_to_class.get(
                        stmt.model_name, stmt.model_name
                    )
                    self._checkpointed_models.add(class_name)
                # Detect distributed training
                dist_node = stmt.params.get("distributed")
                if isinstance(dist_node, BooleanLiteral) and dist_node.value:
                    self._is_distributed = True
                    self._distributed_models.add(stmt.model_name)

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

        if self._has_checkpoint:
            self.emitter.add_import("from torch.utils.checkpoint import checkpoint as activation_checkpoint")

        if self._is_distributed:
            self.emitter.add_import("import os")
            self.emitter.add_import("import torch.distributed as dist")
            self.emitter.add_import("from torch.nn.parallel import DistributedDataParallel as DDP")
            self.emitter.add_import("from torch.utils.data.distributed import DistributedSampler")

        if self._has_kernels:
            self.emitter.add_import("import triton")
            self.emitter.add_import("import triton.language as tl")

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
        elif isinstance(node, KernelDef):
            self._emit_kernel(node)
        elif isinstance(node, NMLModel):
            self._emit_nml_model(node)
        else:
            self.emitter.comment(f"[tlc] Unsupported node: {type(node).__name__}")

    # ── import ────────────────────────────────────────────────

    def _emit_import(self, node: ImportStatement) -> None:
        if node.is_nml and node.alias:
            # Cross-file NML import: sub-compile the .nml file
            self._emit_nml_import(node)
        else:
            path = ".".join(node.module_path)
            self.emitter.comment(f"TensorLoom import: {path}")

    def _emit_nml_import(self, node: ImportStatement) -> None:
        """Sub-compile a .nml file and inject its generated class inline."""
        import os
        nml_filename = ".".join(node.module_path[:-1]) + ".nml"
        nml_path = os.path.join(self.source_dir, nml_filename)

        if not os.path.exists(nml_path):
            self.emitter.comment(
                f"[ERROR] NML file not found: {nml_path}"
            )
            self.emitter.comment(
                f"  import {'.'.join(node.module_path)} as {node.alias}"
            )
            return

        # Read and sub-compile the .nml file
        with open(nml_path, "r", encoding="utf-8") as f:
            nml_source = f.read()

        from tensorloom.lexer.lexer import Lexer
        from tensorloom.parser.parser import Parser

        nml_tokens = Lexer(nml_source, filename=nml_filename).tokenize()
        nml_ast = Parser(nml_tokens).parse()

        # Find the NMLModel node(s) and emit them
        self.emitter.blank()
        self.emitter.comment(f"=== NML Import: {nml_filename} as {node.alias} ===")

        for stmt in nml_ast.statements:
            if isinstance(stmt, NMLModel):
                # If there's an alias, rename the class
                original_name = stmt.name
                if node.alias and node.alias != original_name:
                    stmt.name = node.alias
                self._emit_nml_model(stmt)
                # Restore original name (don't mutate permanently)
                stmt.name = original_name
                break  # Only import the first model definition

        self.emitter.blank()

    # ── @kernel → Triton JIT kernel + launcher ────────────────

    def _emit_kernel(self, node: KernelDef) -> None:
        """Emit a @triton.jit kernel and an auto-generated launcher wrapper."""
        self.emitter.blank()
        self.emitter.comment(f"═══ TensorLoom Triton Kernel: {node.name} ═══")
        self.emitter.blank()

        # ── 1. Emit @triton.jit decorated kernel function ──
        self.emitter.line("@triton.jit")

        # Build parameter list with tl.constexpr annotations
        param_strs: list[str] = []
        constexpr_params: list[str] = []
        pointer_params: list[str] = []
        scalar_params: list[str] = []

        for p in node.params:
            if p.is_constexpr:
                param_strs.append(f"{p.name}: tl.constexpr")
                constexpr_params.append(p.name)
            elif p.name.endswith("_ptr"):
                param_strs.append(p.name)
                pointer_params.append(p.name)
            else:
                param_strs.append(p.name)
                scalar_params.append(p.name)

        self.emitter.line(f"def {node.name}({', '.join(param_strs)}):")
        self.emitter.indent()
        self.emitter.docstring(f"TensorLoom Triton kernel: {node.name}")

        # Emit kernel body — transpile let statements to plain assignments
        for stmt in node.body:
            self._emit_kernel_statement(stmt)

        self.emitter.dedent()
        self.emitter.blank()

        # ── 2. Auto-generate launcher wrapper ──
        # The launcher handles: grid calculation, pointer extraction, kernel dispatch
        launcher_name = f"{node.name}_launcher"

        # Build launcher params: replace _ptr params with tensor names
        launcher_params: list[str] = []
        for p in node.params:
            if p.is_constexpr:
                launcher_params.append(f"{p.name}=1024")  # sensible default
            elif p.name.endswith("_ptr"):
                # Convert x_ptr → x (tensor argument)
                tensor_name = p.name.removesuffix("_ptr")
                launcher_params.append(tensor_name)
            else:
                launcher_params.append(p.name)

        self.emitter.line(f"def {launcher_name}({', '.join(launcher_params)}):")
        self.emitter.indent()
        self.emitter.docstring(
            f"Auto-generated launcher for Triton kernel '{node.name}'.\n"
            f"    Computes grid dimensions and dispatches the kernel."
        )

        # Determine the size variable — use first scalar param or infer from first pointer
        size_var = None
        for p in node.params:
            if not p.is_constexpr and not p.name.endswith("_ptr"):
                size_var = p.name
                break

        # If there's a constexpr BLOCK_SIZE, generate grid with cdiv
        block_param = None
        for p in constexpr_params:
            if "BLOCK" in p.upper() or "SIZE" in p.upper():
                block_param = p
                break

        if size_var and block_param:
            self.emitter.line(
                f"grid = lambda meta: (triton.cdiv({size_var}, meta['{block_param}']),)"
            )
        elif size_var:
            self.emitter.line(
                f"grid = ({size_var},)"
            )
        else:
            self.emitter.comment("Grid: user must specify grid dimensions")
            self.emitter.line("grid = (1,)")

        # Output tensor allocation if there are output pointers
        output_ptrs = [p for p in pointer_params if p.startswith(("z_ptr", "out_ptr", "output_ptr", "result_ptr", "c_ptr"))]
        if output_ptrs:
            # Infer output shape from first input pointer
            first_input = [p for p in pointer_params if p not in output_ptrs]
            if first_input:
                first_tensor = first_input[0].removesuffix("_ptr")
                for op in output_ptrs:
                    out_tensor = op.removesuffix("_ptr")
                    self.emitter.line(
                        f"{out_tensor} = torch.empty_like({first_tensor})"
                    )

        # Build kernel call arguments
        kernel_args: list[str] = []
        for p in node.params:
            if p.is_constexpr:
                kernel_args.append(f"{p.name}={p.name}")
            elif p.name.endswith("_ptr"):
                tensor_name = p.name.removesuffix("_ptr")
                kernel_args.append(tensor_name)
            else:
                kernel_args.append(p.name)

        self.emitter.line(f"{node.name}[grid]({', '.join(kernel_args)})")

        # Return output tensors if any
        if output_ptrs:
            out_names = [p.removesuffix("_ptr") for p in output_ptrs]
            self.emitter.line(f"return {', '.join(out_names)}")
        else:
            # In-place kernel, return first input
            if pointer_params:
                first = pointer_params[0].removesuffix("_ptr")
                self.emitter.line(f"return {first}")

        self.emitter.dedent()
        self.emitter.blank()

    def _emit_kernel_statement(self, node: ASTNode) -> None:
        """Emit a statement inside a Triton kernel body.
        
        Transpiles TensorLoom let-statements to standard Python assignments
        and preserves tl.* calls as-is for Triton.
        """
        if isinstance(node, LetStatement):
            val = self._emit_expr(node.value)
            self.emitter.line(f"{node.name} = {val}")
        elif isinstance(node, AssignStatement):
            target = self._emit_expr(node.target)
            val = self._emit_expr(node.value)
            self.emitter.line(f"{target} = {val}")
        elif isinstance(node, ExpressionStatement):
            expr = self._emit_expr(node.expression)
            self.emitter.line(expr)
        elif isinstance(node, IfStatement):
            self._emit_if(node)
        elif isinstance(node, ForLoop):
            self._emit_for(node)
        elif isinstance(node, ReturnStatement):
            self._emit_return(node)
        else:
            self.emitter.comment(f"[kernel] Unsupported: {type(node).__name__}")

    # ── @model (NML) → nn.Module with configurable __init__ ──

    def _emit_nml_model(self, node: NMLModel) -> None:
        """Emit a PyTorch nn.Module from an NML @model declaration.
        
        @config values become __init__ keyword arguments with defaults,
        allowing callers to override them at instantiation time.
        """
        self.emitter.blank()
        self.emitter.comment(f"=== NML Model: {node.name} ===")
        self.emitter.blank()

        self.emitter.line(f"class {node.name}(nn.Module):")
        self.emitter.indent()
        self.emitter.docstring(
            f"NML declarative model: {node.name}\n"
            f"    Auto-generated from .nml specification."
        )
        self.emitter.blank()

        # ── __init__ with @config as keyword args ──
        # Build __init__ signature from config keys
        config_params: list[str] = []
        for key, default_val in node.config.items():
            default_str = self._emit_expr(default_val)
            config_params.append(f"{key}={default_str}")

        if config_params:
            init_sig = f"def __init__(self, {', '.join(config_params)}):"
        else:
            init_sig = "def __init__(self):"

        self.emitter.line(init_sig)
        self.emitter.indent()
        self.emitter.line("super().__init__()")

        # Store config values as instance attributes
        for key in node.config:
            self.emitter.line(f"self.{key} = {key}")

        self.emitter.blank()

        # Emit layer declarations — may reference config variables
        for layer in node.layers:
            layer_code = self._emit_nml_layer(layer, node.config)
            self.emitter.line(f"self.{layer.name} = {layer_code}")

        self.emitter.dedent()
        self.emitter.blank()

        # ── forward method from @forward ──
        if node.forward_body:
            fwd_params = ", ".join(["self"] + node.forward_params)
            self.emitter.line(f"def forward({fwd_params}):")
            self.emitter.indent()

            for stmt in node.forward_body:
                self._emit_nml_forward_statement(stmt, node)

            self.emitter.dedent()
            self.emitter.blank()

        self.emitter.dedent()
        self.emitter.blank()

        # Register as model name so train blocks can reference it
        self._model_names.add(node.name)

    def _emit_nml_layer(self, layer: LayerDeclaration,
                         config: dict[str, ASTNode]) -> str:
        """Emit a layer declaration, resolving config references to self.X."""
        if isinstance(layer.layer_type, FunctionCall):
            call = layer.layer_type
            if isinstance(call.callee, Identifier):
                layer_class = LAYER_MAP.get(call.callee.name, call.callee.name)
                args = []
                for a in call.args:
                    expr = self._emit_expr(a)
                    # If arg references a config key, prefix with self.
                    if isinstance(a, Identifier) and a.name in config:
                        expr = f"self.{a.name}"
                    args.append(expr)
                kwargs = []
                for k, v in call.kwargs.items():
                    val = self._emit_expr(v)
                    if isinstance(v, Identifier) and v.name in config:
                        val = f"self.{v.name}"
                    kwargs.append(f"{k}={val}")
                all_args = ", ".join(args + kwargs)
                return f"{layer_class}({all_args})"
        return self._emit_expr(layer.layer_type)

    def _emit_nml_forward_statement(self, node: ASTNode,
                                      nml: NMLModel) -> None:
        """Emit a statement in an NML forward body.
        
        Layer references (e.g., 'attention(x)') are rewritten to
        'self.attention(x)' to match the nn.Module pattern.
        """
        if isinstance(node, AssignStatement):
            target = self._emit_expr(node.target)
            val = self._emit_nml_forward_expr(node.value, nml)
            self.emitter.line(f"{target} = {val}")
        elif isinstance(node, LetStatement):
            val = self._emit_nml_forward_expr(node.value, nml)
            self.emitter.line(f"{node.name} = {val}")
        elif isinstance(node, ReturnStatement):
            if node.value:
                val = self._emit_nml_forward_expr(node.value, nml)
                self.emitter.line(f"return {val}")
            else:
                self.emitter.line("return")
        elif isinstance(node, ExpressionStatement):
            expr = self._emit_nml_forward_expr(node.expression, nml)
            self.emitter.line(expr)
        else:
            self._emit_statement(node)

    def _emit_nml_forward_expr(self, node: ASTNode, nml: NMLModel) -> str:
        """Emit an expression in NML forward, prefixing layer names with self."""
        layer_names = {l.name for l in nml.layers}

        if isinstance(node, FunctionCall):
            if isinstance(node.callee, Identifier) and node.callee.name in layer_names:
                # Rewrite layer(x) → self.layer(x)
                args = ", ".join(
                    self._emit_nml_forward_expr(a, nml) for a in node.args
                )
                kwargs = ", ".join(
                    f"{k}={self._emit_nml_forward_expr(v, nml)}"
                    for k, v in node.kwargs.items()
                )
                all_args = ", ".join(filter(None, [args, kwargs]))
                return f"self.{node.callee.name}({all_args})"
            else:
                # Standard function call
                callee = self._emit_expr(node.callee)
                args = ", ".join(
                    self._emit_nml_forward_expr(a, nml) for a in node.args
                )
                kwargs = ", ".join(
                    f"{k}={self._emit_nml_forward_expr(v, nml)}"
                    for k, v in node.kwargs.items()
                )
                all_args = ", ".join(filter(None, [args, kwargs]))
                return f"{callee}({all_args})"

        if isinstance(node, BinaryOp):
            left = self._emit_nml_forward_expr(node.left, nml)
            right = self._emit_nml_forward_expr(node.right, nml)
            return f"({left} {node.op} {right})"

        # Fallback to standard expression emission
        return self._emit_expr(node)

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
            # If model needs gradient checkpointing and this is forward(),
            # generate a _forward_body + checkpointed forward wrapper
            if method.name == "forward" and node.name in self._checkpointed_models:
                self._emit_checkpointed_forward(method)
            else:
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

    def _emit_checkpointed_forward(self, method: FunctionDef) -> None:
        """Emit a forward() that wraps computation in torch.utils.checkpoint.

        This enables *activation checkpointing* — intermediate activations are
        discarded during the forward pass and recomputed during backward,
        trading ~30 %% extra compute for up to 60 %% less activation memory.

        Generated pattern:
            def _forward_body(self, x):
                # original forward body
            def forward(self, x):
                return activation_checkpoint(self._forward_body, x, use_reentrant=False)
        """
        # 1) _forward_body with the real logic
        params = []
        for p in method.params:
            params.append("self" if p.name == "self" else p.name)
        param_str = ", ".join(params)

        self.emitter.line(f"def _forward_body({param_str}):")
        self.emitter.indent()
        self.emitter.comment("Actual computation (activations are recomputed during backward)")
        for stmt in method.body:
            self._emit_statement(stmt)
        self.emitter.dedent()
        self.emitter.blank()

        # 2) public forward wrapping _forward_body in activation_checkpoint
        non_self_params = [p.name for p in method.params if p.name != "self"]
        args_str = ", ".join(non_self_params)

        self.emitter.line(f"def forward({param_str}):")
        self.emitter.indent()
        self.emitter.docstring(
            "Gradient-checkpointed forward — trades compute for ~60%% less activation memory."
        )
        self.emitter.line(
            f"return activation_checkpoint(self._forward_body, {args_str}, use_reentrant=False)"
        )
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

        # Weight snapshot saving (periodic model state backups)
        if ckpt_freq:
            self.emitter.blank()
            self.emitter.comment("Weight Snapshot — periodic model backup")
            self.emitter.line(f"if (epoch + 1) % {ckpt_freq} == 0:")
            self.emitter.indent()
            self.emitter.line(f'torch.save({node.model_name}.state_dict(), '
                              f'f"checkpoint_epoch_{{epoch+1}}.pt")')
            self.emitter.line(f'print(f"  [snapshot] Saved checkpoint_epoch_{{epoch+1}}.pt")')
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
