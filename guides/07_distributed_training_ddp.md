# 🌐 Guide 07: The Superhero Team (Multi-GPU Distributed Training)

> *One superhero is awesome. But what if a giant monster appears, and you need the entire Avengers team to fight together? You give each superhero a walkie-talkie so they can coordinate their attacks in real time! That is what Distributed Data Parallel (DDP) does with GPUs.*

---

## 🦸 Why Train on Multiple GPUs?

When your AI model is huge (like a language model or a high-res video generator), training on a single GPU might take **3 whole months**.

If you connect **8 GPUs** together:
- GPU 1 reads Chapter 1 of the dataset.
- GPU 2 reads Chapter 2.
- GPU 3 reads Chapter 3... and so on!
- After each batch, all 8 GPUs quickly share their notes over a high-speed walkie-talkie network (NCCL).

Now your 3-month training job finishes in **a few days**!

```
                    [ 📚 Huge Training Dataset ]
                    /       |           |       \
               Chapter 1  Chapter 2  Chapter 3  Chapter 4
                  │         │           │         │
                  ▼         ▼           ▼         ▼
               [GPU 0]   [GPU 1]     [GPU 2]   [GPU 3]
                  \         │           │         /
                   \────────┴─────┬─────┴────────/
                                  ▼
                     [ 📻 High-Speed NCCL Sync ]
                     "All GPUs agree on new weights!"
```

---

## 😫 The Nightmare of Distributed Coding in PyTorch

In regular PyTorch, setting up multi-GPU training is terrifying:
1. You have to read environment variables like `LOCAL_RANK`, `RANK`, and `WORLD_SIZE`.
2. You have to initialize the NCCL communication backend.
3. You have to wrap the model in `DistributedDataParallel`.
4. You have to shard the dataset with `DistributedSampler`.
5. You have to call `sampler.set_epoch(epoch)` every single round.
6. You have to silence all GPUs except Rank 0 so your terminal doesn't print the same message 8 times.
7. You have to cleanly destroy the process group when finished.

If you miss any step, the GPUs lock up and freeze forever! 🧊

---

## 🪄 The TensorLoom Magic: Just Write `distributed = true`

In TensorLoom, all of that nightmare plumbing is replaced with **one word**:

```
import transformer.nml as Transformer

let net = Transformer(d_model=512, n_heads=8)
let data = load_dataset()

train net on data:
    epochs = 20
    optimizer = AdamW(lr=0.0001)
    loss = CrossEntropy
    precision = fp16
    distributed = true     // 🚀 BOOM! That's it!
```

---

## 🚀 How to Launch on 4 GPUs with `torchrun`

When you compile your script with `python -m tensorloom compile train.tl -o train_ddp.py`, TensorLoom writes all 145 lines of distributed sync code for you.

To launch your superhero team across 4 GPUs, just run:

```bash
torchrun --nproc_per_node=4 train_ddp.py
```

All 4 GPUs will fire up, divide the work evenly, sync their gradients, and train your model at maximum cluster throughput! 🏎️💨
