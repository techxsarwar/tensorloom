# 🌐 Module 05: Distributed Systems Architecture & DDP

This module details the systems-level mechanics of TensorLoom’s automated **Distributed Data Parallel (DDP)** code generation, collective communication primitives with NCCL, process group lifecycle management, and deterministic distributed sampling.

---

## 1. The Distributed Data Parallel (DDP) Architecture

In large-scale AI training, data parallelism replicates the neural network across $N$ discrete GPU devices (ranks). Each rank receives an independent shard of the batch:

```
                            [ Global Batch: Size B ]
                           /           |            \
                   Shard B/N       Shard B/N     Shard B/N
                      │                │             │
                      ▼                ▼             ▼
                 [ Rank 0 ]       [ Rank 1 ]    [ Rank N-1 ]
                 (Local Fwd)      (Local Fwd)   (Local Fwd)
                      │                │             │
                      ▼                ▼             ▼
                 (Local Loss)     (Local Loss)  (Local Loss)
                      │                │             │
                      ▼                ▼             ▼
                 (Local Back)     (Local Back)  (Local Back)
                      \                |            /
                       \───────────────┼───────────/
                                       ▼
                       [ NCCL Ring All-Reduce Collectives ]
                       Gradients averaged across all ranks:
                         G_global = (1/N) * sum(G_i)
                                       │
                                       ▼
                       [ Synchronized Optimizer Step ]
```

---

## 2. Low-Level Boilerplate Automated by TensorLoom

When `distributed = true` is declared in a `.tl` training block, the compiler automatically generates a 145-line distributed runtime scaffold:

### 2.1 Process Group Setup (`setup_ddp`)
```python
def setup_ddp():
    """Initialize distributed process group with NCCL backend."""
    dist.init_process_group(backend="nccl", init_method="env://")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
```

### 2.2 Model Wrapping & Gradient Bucketing
```python
local_rank = int(os.environ["LOCAL_RANK"])
net = Net().to(local_rank)
net = torch.nn.parallel.DistributedDataParallel(
    net, 
    device_ids=[local_rank],
    output_device=local_rank,
    broadcast_buffers=True
)
```

### 2.3 Deterministic Distributed Sampling
To prevent duplicate sample consumption across ranks and ensure uniform random permutations across epochs:
```python
sampler = DistributedSampler(dataset, shuffle=True)
dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

for epoch in range(epochs):
    sampler.set_epoch(epoch)  # Crucial for deterministic multi-GPU shuffling
```

### 2.4 Rank-0 Gated Telemetry & Barrier Teardown
```python
# Suppress redundant output on worker processes
if int(os.environ.get("LOCAL_RANK", 0)) == 0:
    print(f"Epoch {epoch} | Loss: {epoch_loss:.4f}")

dist.barrier()               # Ensure all ranks finish before shutdown
dist.destroy_process_group() # Release NCCL communicator contexts
```

---

## 3. Launching Distributed Jobs with `torchrun`

TensorLoom-compiled distributed scripts adhere to the standard PyTorch Distributed Elastic launcher contract:

```bash
# Multi-GPU single-node execution (4 GPUs)
torchrun --nproc_per_node=4 train_ddp_compiled.py

# Multi-Node execution across two 8-GPU nodes (16 GPUs total)
torchrun \
    --nnodes=2 \
    --nproc_per_node=8 \
    --rdzv_id=tensorloom_job_101 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=master-node.cluster.local:29500 \
    train_ddp_compiled.py
```
