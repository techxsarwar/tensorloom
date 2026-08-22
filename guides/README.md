# 📚 The TensorLoom Documentation & Learning Center

Welcome to the official documentation hub for **TensorLoom** — the high-performance, GPU-efficient domain-specific language for AI training.

Choose your learning track below:

---

## 🧭 Choose Your Track

```
┌─────────────────────────────────────────────────────────────┐
│                 TensorLoom Learning Hub                     │
├──────────────────────────────┬──────────────────────────────┤
│  🚀 Beginner / Fun Track     │  🎓 Professional Track       │
│  (Explained with intuitive   │  (Rigorous math, systems,    │
│   analogies & 5-min guides)  │   compiler IR, and GPU arch) │
├──────────────────────────────┼──────────────────────────────┤
│  👉 [Start Beginner Track](#-beginner--kid-friendly-track)  │  👉 [Start Professional Track](professional/README.md) │
└──────────────────────────────┴──────────────────────────────┘
```

---

## 🚀 Beginner & Kid-Friendly Track

*Learn AI programming using intuitive analogies: LEGO builders, robotic elves, superhero teams, waterslides, and toddler shape-sorters!*

| Chapter | Guide Link | What You Will Discover |
| :--- | :--- | :--- |
| **01** | [**🚀 Why TensorLoom?**](01_why_tensorloom.md) | Why normal AI code is like building LEGO with oven mitts, and how TensorLoom gives you superpowers. |
| **02** | [**⚡ 5-Minute Quickstart**](02_quickstart_tutorial.md) | Build your very first number-reading robot brain in 60 seconds from scratch! |
| **03** | [**📐 The Secret Blueprint (`.nml`)**](03_nml_declarative_models.md) | Learn the 4 magic blocks of `.nml` and the 3 behind-the-scenes compiler transformations. |
| **04** | [**🛠️ The Training Gym (`.tl`)**](04_imperative_tl_pipelines.md) | Master variables with `let`, the fun Waterslide Operator (`\|>`), and the AI Personal Trainer (`train`). |
| **05** | [**📦 The LEGO Toy Box (Imports)**](05_cross_file_modules.md) | Plug blueprints from other files into your training scripts without copying and pasting! |
| **06** | [**⚡ The Nitro Turbo Button (Triton)**](06_custom_triton_kernels.md) | Command 10,000 GPU elves simultaneously using bare-metal inline `@kernel` blocks. |
| **07** | [**🌐 The Superhero Team (Multi-GPU DDP)**](07_distributed_training_ddp.md) | Train across 8 GPUs at once just by typing `distributed = true`. |
| **08** | [**🔍 The Shape-Sorting Toy (Verification)**](08_shape_inference_and_verification.md) | How the compiler catches "square peg in round hole" bugs in 2 milliseconds before they crash. |

---

## 🎓 Professional & Systems Engineering Track

*For ML researchers, systems engineers, and compiler architects seeking a formal, deep-dive curriculum.*

| Module | Module Link | Technical Focus |
| :--- | :--- | :--- |
| **Index** | [**🎓 Professional Track Master Index**](professional/README.md) | Full curriculum overview, dependency tree, and learning pathways. |
| **M01** | [**Language Fundamentals & Grammar**](professional/01_language_fundamentals.md) | Lexical scanner, token hierarchy, EBNF statement grammar, and AST dataclasses. |
| **M02** | [**Formal NML Architecture Specification**](professional/02_nml_architecture_spec.md) | AST lowering rules, dynamic parameter synthesis, and constructor polymorphism. |
| **M03** | [**Pipeline Desugaring & Runtime Codegen**](professional/03_pipeline_and_runtime.md) | Pipe operator lowering, TorchDynamo kernel fusion, AMP, and activation checkpointing. |
| **M04** | [**Bare-Metal GPU Engineering with Triton**](professional/04_triton_accelerator_deep_dive.md) | SPMD execution, SRAM block tiling, pointer math, and dynamic grid dispatchers. |
| **M05** | [**Distributed Systems Architecture & DDP**](professional/05_distributed_systems_ddp_fsdp.md) | NCCL collective ring all-reduce, DistributedSampler epoch ordering, and `torchrun`. |
| **M06** | [**Static Analysis & Shape Inference Engine**](professional/06_static_analysis_and_shape_engine.md) | Abstract interpretation, spatial dimension propagation, and broadcast verification. |
| **M07** | [**Compiler Internals & Extending TensorLoom**](professional/07_compiler_internals_and_extending.md) | Parser architecture, visitor passes, building custom backends (C++/CUDA, JAX, ONNX). |

---

## 💻 Quick CLI Reference

```bash
# Static verification & type checking
python -m tensorloom check path/to/script.tl

# Structural AST inspection & parameter calculation
python -m tensorloom info path/to/model.nml

# Transpile to production PyTorch code
python -m tensorloom compile path/to/script.tl -o build/compiled.py

# Compile and run immediately
python -m tensorloom run path/to/script.tl
```
