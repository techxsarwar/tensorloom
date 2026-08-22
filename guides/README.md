# 📚 The TensorLoom Documentation & Learning Center

Welcome to the official documentation hub for **TensorLoom** — the high-performance, GPU-efficient domain-specific language for AI training.

Choose the learning style that fits you best:

---

## 🧭 Choose Your Learning Track

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TensorLoom Learning Hub                           │
├─────────────────────────┬─────────────────────────┬─────────────────────────┤
│  ⚡ Bite-Sized Track    │  🚀 Intuitive Track     │  🎓 Professional Track  │
│  (1–2 min micro-lessons │  (LEGO, superhero, and  │  (Rigorous math, GPU    │
│   zero-fluff cheat cards)│   intuitive analogies)  │   systems, compiler IR) │
├─────────────────────────┼─────────────────────────┼─────────────────────────┤
│  👉 [Start 2-Min Crash] │  👉 [Start Intuitive]   │  👉 [Start Pro Track]   │
│     (bite_sized/README) │     (#-intuitive-track) │     (professional/READ) │
└─────────────────────────┴─────────────────────────┴─────────────────────────┘
```

---

## ⚡ 1. Bite-Sized Micro-Lessons (1–2 Min Each)

*Ultra-concise, punchy cards. Learn everything in under 15 minutes.*

| # | Micro-Lesson | Time | Concept Covered |
| :-: | :--- | :-: | :--- |
| **01** | [**Hello TensorLoom**](bite_sized/01_hello_tensorloom.md) | 1 min | 10-line complete program & 3 essential CLI commands |
| **02** | [**Variables & Types**](bite_sized/02_variables_and_types.md) | 1 min | Declaring data with `let` and type annotations |
| **03** | [**The Pipe Operator (`\|>`)**](bite_sized/03_pipe_operator.md) | 1 min | Clean left-to-right functional dataflow |
| **04** | [**Imperative Models**](bite_sized/04_imperative_models.md) | 2 min | Writing `model` classes with `layer` and `fn forward` |
| **05** | [**Declarative NML**](bite_sized/05_declarative_nml.md) | 2 min | The 4 blocks of `.nml` blueprints and auto-rewrites |
| **06** | [**The `train` Block**](bite_sized/06_train_block.md) | 1 min | Automated training loops, AMP, & checkpoints |
| **07** | [**Cross-File Imports**](bite_sized/07_cross_file_imports.md) | 1 min | `import arch.nml as Alias` with overrides |
| **08** | [**Inline Triton Kernels**](bite_sized/08_triton_kernels.md) | 2 min | Custom GPU accelerator code with `@kernel` |
| **09** | [**Multi-GPU DDP**](bite_sized/09_distributed_ddp.md) | 1 min | 1-line cluster scaling with `distributed = true` |
| **10** | [**Shape Checking & Info**](bite_sized/10_shape_checking.md) | 1 min | Catching bugs in <3ms with `tlc check` & `tlc info` |

---

## 🚀 2. Intuitive & Kid-Friendly Adventure Track

*Deep-dive conceptual lessons using intuitive analogies: LEGO builders, robotic elves, and superhero teams.*

| Chapter | Adventure Guide | What You Will Discover |
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

## 🎓 3. Professional Systems Engineering Track

*For ML researchers, systems engineers, and compiler architects seeking a formal, mathematical deep-dive.*

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
# Static verification & type checking (<3ms)
python -m tensorloom check path/to/script.tl

# Structural AST inspection & parameter calculation
python -m tensorloom info path/to/model.nml

# Transpile to production PyTorch code
python -m tensorloom compile path/to/script.tl -o build/compiled.py

# Compile and run immediately
python -m tensorloom run path/to/script.tl
```
