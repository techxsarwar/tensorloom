# 🌐 Guide 07: Multi-GPU Distributed Data Parallel (DDP)

Scaling deep learning models across multiple GPUs is traditionally one of the most tedious and error-prone parts of PyTorch engineering.

TensorLoom reduces this to a single declarative flag: `distributed = true`.

---

## 1. Enabling DDP in TensorLoom

Simply add `distributed = true` to your `train` block:

```
import transformer.nml as Transformer

let net = Transformer(d_model=512, n_heads=8)
let data = load_dataset()

train net on data:
    epochs = 20
    optimizer = AdamW(lr=0.0001)
    loss = CrossEntropy
    precision = fp16
    distributed = true
```

---

## 2. What TensorLoom Synthesizes Automatically

When `distributed = true` is set, the compiler automatically injects:

1. **Process Group Initialization (`setup_ddp`)**:
   - Initializes NCCL backend with `dist.init_process_group("nccl", init_method="env://")`.
   - Maps each worker process to its local GPU via `torch.cuda.set_device(local_rank)`.
2. **Distributed Dataset Sharding**:
   - Wraps the dataset with `DistributedSampler(dataset, shuffle=True)`.
   - Calls `sampler.set_epoch(epoch)` before every epoch to ensure uniform data shuffling across nodes.
3. **Model Synchronization**:
   - Wraps the model with `torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])`.
4. **Rank-0 Gated Logging**:
   - Suppresses redundant console output on worker ranks so training logs remain clean.
5. **Process Group Teardown (`cleanup_ddp`)**:
   - Adds `dist.barrier()` and `dist.destroy_process_group()` for clean shutdown.
6. **Execution Entrypoint**:
   - Adds the `if __name__ == "__main__":` entrypoint required by multi-process spawn runners.

---

## 3. Running Distributed Training with `torchrun`

Once compiled, launch your script on any multi-GPU machine using PyTorch's `torchrun`:

```bash
# Compile to DDP PyTorch script
python -m tensorloom compile train_ddp.tl -o train_ddp_compiled.py

# Launch on 4 GPUs
torchrun --nproc_per_node=4 train_ddp_compiled.py

# Launch across 2 nodes with 8 GPUs each (16 GPUs total)
torchrun --nnodes=2 --nproc_per_node=8 --rdzv_backend=c10d --rdzv_endpoint=node1:29500 train_ddp_compiled.py
```
