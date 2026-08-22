# 📚 TensorLoom Documentation & Guides

Welcome to the comprehensive guide library for **TensorLoom** — the high-performance, GPU-efficient domain-specific language for AI training.

Whether you're an AI researcher, ML systems engineer, or student, these guides will take you from the fundamentals to advanced bare-metal GPU accelerator programming.

---

## 🗺️ Guide Index

| Chapter | Guide | Topics Covered |
| :--- | :--- | :--- |
| **01** | [**Why TensorLoom?**](01_why_tensorloom.md) | Philosophy, comparison with raw PyTorch, memory efficiency, auto-fusion, and why DSLs matter |
| **02** | [**5-Minute Quickstart**](02_quickstart_tutorial.md) | Installing, CLI usage, writing your first `.tl` script, and transpiling to PyTorch |
| **03** | [**Declarative NML Architecture**](03_nml_declarative_models.md) | Neural Markup Language (`.nml`), `@model`, `@config`, `@layers`, `@forward`, and polymorphic rewrites |
| **04** | [**Imperative `.tl` Scripting**](04_imperative_tl_pipelines.md) | Pipeline syntax, functional pipe operator (`\|>`), `train` blocks, mixed precision, and checkpoints |
| **05** | [**Cross-File Modular Architecture**](05_cross_file_modules.md) | `import model.nml as Alias`, sub-compilation, symbol resolution, and constructor overrides |
| **06** | [**Inline Triton GPU Kernels**](06_custom_triton_kernels.md) | Bare-metal accelerator coding with `@kernel`, tiled memory access, pointer math, and auto-launchers |
| **07** | [**Multi-GPU Distributed Scaling (DDP)**](07_distributed_training_ddp.md) | 1-line cluster scaling with `distributed = true`, NCCL backend, rank-gating, and `torchrun` execution |
| **08** | [**Static Shape Inference & Profiling**](08_shape_inference_and_verification.md) | Catching dimension bugs at compile time (<3ms), parameter counting, and memory profiling with `tlc info` |

---

## ⚡ Quick CLI Tip

When running the compiler from the command line:

```bash
# Using the global binary entry point
tlc compile path/to/script.tl -o output.py

# Or directly using Python module syntax
python -m tensorloom compile path/to/script.tl -o output.py
```
