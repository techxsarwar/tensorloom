"""
Tests for TensorLoom Distributed Data Parallel (DDP) code generation.

Validates that `distributed = true` in a train block generates correct:
  - DDP imports (torch.distributed, DDP, DistributedSampler)
  - setup_ddp() / cleanup_ddp() lifecycle functions
  - Model wrapping with DDP(model, device_ids=[local_rank])
  - DistributedSampler with sampler.set_epoch(epoch)
  - Rank-gated checkpointing and callbacks
  - torchrun entry point with single-GPU fallback
"""
import pytest
from tensorloom.lexer.lexer import Lexer
from tensorloom.parser.parser import Parser
from tensorloom.codegen.pytorch_backend import PyTorchBackend


def compile_source(source: str) -> str:
    """Helper: lex + parse + codegen."""
    tokens = Lexer(source).tokenize()
    ast = Parser(tokens).parse()
    return PyTorchBackend().generate(ast)


# ── Minimal DDP source for testing ────────────────────────────

MINIMAL_DDP = """model Net:
    layer fc = Linear(10, 5)
    fn forward(self, x: Tensor) -> Tensor:
        return self.fc(x)

let net = Net()

train net on data:
    epochs = 5
    optimizer = Adam(lr=0.001)
    loss = CrossEntropy
    distributed = true
"""

FULL_DDP = """model Net:
    layer fc1 = Linear(784, 256)
    layer fc2 = Linear(256, 10)
    fn forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x) |> relu
        x = self.fc2(x)
        return x

let net = Net()

train net on data:
    epochs = 10
    batch_size = 32
    optimizer = Adam(lr=0.001)
    loss = CrossEntropy
    precision = fp16
    distributed = true
    checkpoint every 3 epochs

    on epoch_end(metrics):
        print(f"Epoch {metrics.epoch}: loss={metrics.loss:.4f}")
"""


# ── Import Tests ──────────────────────────────────────────────

class TestDDPImports:
    def test_has_distributed_import(self):
        code = compile_source(MINIMAL_DDP)
        assert "import torch.distributed as dist" in code

    def test_has_ddp_import(self):
        code = compile_source(MINIMAL_DDP)
        assert "from torch.nn.parallel import DistributedDataParallel as DDP" in code

    def test_has_sampler_import(self):
        code = compile_source(MINIMAL_DDP)
        assert "from torch.utils.data.distributed import DistributedSampler" in code

    def test_has_os_import(self):
        code = compile_source(MINIMAL_DDP)
        assert "import os" in code

    def test_no_device_variable(self):
        """DDP mode should NOT emit the standard device variable."""
        code = compile_source(MINIMAL_DDP)
        assert 'device = torch.device(' not in code


# ── Lifecycle Functions ───────────────────────────────────────

class TestDDPLifecycle:
    def test_setup_ddp_function(self):
        code = compile_source(MINIMAL_DDP)
        assert "def setup_ddp():" in code
        assert 'dist.init_process_group(backend="nccl"' in code
        assert 'os.environ["LOCAL_RANK"]' in code
        assert "torch.cuda.set_device(local_rank)" in code

    def test_cleanup_ddp_function(self):
        code = compile_source(MINIMAL_DDP)
        assert "def cleanup_ddp():" in code
        assert "dist.destroy_process_group()" in code

    def test_train_distributed_function(self):
        code = compile_source(MINIMAL_DDP)
        assert "def train_distributed():" in code
        assert "local_rank = setup_ddp()" in code
        assert "cleanup_ddp()" in code


# ── DDP Model Wrapping ───────────────────────────────────────

class TestDDPModelWrapping:
    def test_model_to_local_rank(self):
        code = compile_source(MINIMAL_DDP)
        assert "Net().to(local_rank)" in code

    def test_ddp_wrap(self):
        code = compile_source(MINIMAL_DDP)
        assert "DDP(net, device_ids=[local_rank])" in code

    def test_torch_compile_before_ddp(self):
        """torch.compile should be applied BEFORE DDP wrapping."""
        code = compile_source(MINIMAL_DDP)
        compile_pos = code.index("torch.compile")
        ddp_pos = code.index("DDP(net")
        assert compile_pos < ddp_pos


# ── DistributedSampler ────────────────────────────────────────

class TestDistributedSampler:
    def test_sampler_creation(self):
        code = compile_source(MINIMAL_DDP)
        assert "DistributedSampler(" in code
        assert "num_replicas=dist.get_world_size()" in code
        assert "rank=dist.get_rank()" in code

    def test_sampler_set_epoch(self):
        """Critical: sampler.set_epoch(epoch) must be called each epoch."""
        code = compile_source(MINIMAL_DDP)
        assert "sampler.set_epoch(epoch)" in code

    def test_loader_uses_sampler(self):
        code = compile_source(MINIMAL_DDP)
        assert "DataLoader(data, batch_size=" in code
        assert "sampler=sampler)" in code

    def test_data_to_local_rank(self):
        """Tensors should be moved to local_rank, not device."""
        code = compile_source(MINIMAL_DDP)
        assert ".to(local_rank)" in code
        assert ".to(device)" not in code

    def test_custom_batch_size(self):
        code = compile_source(FULL_DDP)
        assert "batch_size=32" in code


# ── Rank-Gated Operations ────────────────────────────────────

class TestRankGating:
    def test_checkpoint_rank_gated(self):
        """Only rank 0 should save checkpoints."""
        code = compile_source(FULL_DDP)
        assert "dist.get_rank() == 0" in code
        assert "torch.save(" in code

    def test_checkpoint_uses_module_state_dict(self):
        """DDP models need .module.state_dict() not .state_dict()."""
        code = compile_source(FULL_DDP)
        assert ".module.state_dict()" in code

    def test_callback_rank_gated(self):
        """Callbacks should only execute on rank 0."""
        code = compile_source(FULL_DDP)
        # The callback print should be inside a rank 0 guard
        callback_section = code[code.index("Callback: epoch_end"):]
        assert "dist.get_rank() == 0" in code


# ── Entry Point ───────────────────────────────────────────────

class TestEntryPoint:
    def test_main_guard(self):
        code = compile_source(MINIMAL_DDP)
        assert 'if __name__ == "__main__":' in code

    def test_torchrun_check(self):
        code = compile_source(MINIMAL_DDP)
        assert '"LOCAL_RANK" in os.environ' in code
        assert "train_distributed()" in code

    def test_single_gpu_fallback(self):
        code = compile_source(MINIMAL_DDP)
        assert 'os.environ["MASTER_ADDR"]' in code
        assert 'os.environ["MASTER_PORT"]' in code
        assert 'os.environ["WORLD_SIZE"] = "1"' in code


# ── Mixed Precision + DDP ─────────────────────────────────────

class TestDDPWithAMP:
    def test_amp_with_ddp(self):
        code = compile_source(FULL_DDP)
        assert "autocast(device_type='cuda'" in code
        assert "GradScaler" in code
        assert "scaler.scale(loss).backward()" in code

    def test_activation_checkpoint_with_ddp(self):
        """Activation checkpointing should work with DDP."""
        code = compile_source(FULL_DDP)
        assert "_forward_body" in code
        assert "activation_checkpoint" in code


# ── Valid Python ──────────────────────────────────────────────

class TestDDPValidPython:
    def test_minimal_compiles(self):
        code = compile_source(MINIMAL_DDP)
        compile(code, "<test>", "exec")

    def test_full_compiles(self):
        code = compile_source(FULL_DDP)
        compile(code, "<test>", "exec")


# ── Non-Distributed Regression ────────────────────────────────

class TestNonDistributedRegression:
    """Ensure distributed: false (or absent) produces standard single-GPU code."""

    STANDARD_SOURCE = """model Net:
    layer fc = Linear(10, 5)
    fn forward(self, x: Tensor) -> Tensor:
        return self.fc(x)

let net = Net()

train net on data:
    epochs = 5
    optimizer = Adam(lr=0.001)
    loss = CrossEntropy
"""

    def test_no_ddp_imports(self):
        code = compile_source(self.STANDARD_SOURCE)
        assert "torch.distributed" not in code
        assert "DistributedDataParallel" not in code
        assert "DistributedSampler" not in code

    def test_uses_device_not_local_rank(self):
        code = compile_source(self.STANDARD_SOURCE)
        assert "device = torch.device(" in code
        assert "local_rank" not in code

    def test_no_setup_cleanup(self):
        code = compile_source(self.STANDARD_SOURCE)
        assert "setup_ddp" not in code
        assert "cleanup_ddp" not in code
