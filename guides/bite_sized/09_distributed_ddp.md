# ⚡ Lesson 09: Multi-GPU Distributed Scaling (1 Min)

Scale training across clusters of GPUs with a single flag: `distributed = true`.

---

### 1. Enable DDP in Your Script
```
train net on data:
    epochs = 20
    optimizer = AdamW(lr=0.0001)
    loss = CrossEntropy
    precision = fp16
    distributed = true     // 🚀 Scales to all GPUs!
```

---

### 2. What the Compiler Injects (145 Lines of DDP Scaffold)
- **NCCL Backend Init**: `dist.init_process_group("nccl")`
- **Device Pinning**: `torch.cuda.set_device(local_rank)`
- **Model Synchronization**: `DistributedDataParallel(net)`
- **Data Sharding**: `DistributedSampler(dataset, shuffle=True)` with `sampler.set_epoch(epoch)`
- **Rank-0 Logging**: Clean console output on rank 0 only.
- **Teardown**: `dist.destroy_process_group()`

---

### 3. Launching with `torchrun`
```bash
# 1. Compile
python -m tensorloom compile train.tl -o train_ddp.py

# 2. Launch across 4 GPUs
torchrun --nproc_per_node=4 train_ddp.py
```

---

### 💡 Key Takeaway
One flag (`distributed = true`) eliminates 145 lines of error-prone PyTorch distributed networking boilerplate.
