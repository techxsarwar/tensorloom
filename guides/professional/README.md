# 🎓 TensorLoom Professional Engineering Curriculum

Welcome to the **TensorLoom Professional Track**. This curriculum is designed for machine learning engineers, systems architects, compiler researchers, and quantitative developers who need a rigorous, mathematical, and systems-level understanding of the TensorLoom language, compiler internals, and execution model.

---

## 🧭 Curriculum Overview & Learning Path

```
                    ┌─────────────────────────────────────────┐
                    │       01. Language Fundamentals         │
                    │  (Grammar, Types, AST, Scoping, Lexing) │
                    └────────────────────┬────────────────────┘
                                         │
                    ┌────────────────────▼────────────────────┐
                    │      02. Formal NML Specification       │
                    │  (Declarative AST, Polymorphism, Lower) │
                    └────────────────────┬────────────────────┘
                                         │
                    ┌────────────────────▼────────────────────┐
                    │   03. Execution Pipeline & Codegen      │
                    │ (Desugaring, Fusion, AMP, Checkpointing)│
                    └────────────────────┬────────────────────┘
                                         │
         ┌───────────────────────────────┴───────────────────────────────┐
         │                                                               │
┌────────▼─────────────────────────┐                   ┌─────────────────▼───────────────┐
│ 04. Triton Accelerator Internals │                   │ 05. Distributed Systems (DDP)   │
│ (SPMD, Tiling, Coalescing, Grid) │                   │ (NCCL Collectives, Rank Sync)   │
└────────┬─────────────────────────┘                   └─────────────────┬───────────────┘
         │                                                               │
         └───────────────────────────────┬───────────────────────────────┘
                                         │
                    ┌────────────────────▼────────────────────┐
                    │ 06. Static Analysis & Shape Inference   │
                    │ (Abstract Interpretation, Dimension IR) │
                    └────────────────────┬────────────────────┘
                                         │
                    ┌────────────────────▼────────────────────┐
                    │ 07. Compiler Internals & Extending      │
                    │ (Lexer, Recursive-Descent, Backends)    │
                    └─────────────────────────────────────────┘
```

---

## 📚 Course Modules

| Module | Title | Primary Focus |
| :--- | :--- | :--- |
| **[Module 01](01_language_fundamentals.md)** | **Language Fundamentals** | Lexical analysis, 30+ token types, EBNF grammar, scoping, `let` bindings, operator precedence, and AST node hierarchy. |
| **[Module 02](02_nml_architecture_spec.md)** | **Formal NML Specification** | Formal syntax of `.nml`, AST node representation (`NMLModel`), `@config` parameter synthesis, dynamic constructor polymorphism, and class inlining. |
| **[Module 03](03_pipeline_and_runtime.md)** | **Pipeline & Runtime Model** | Pipe desugaring algorithm (`\|>`), PyTorch code generator architecture, Kernel Fusion with `torch.compile(max-autotune)`, and non-reentrant activation checkpointing. |
| **[Module 04](04_triton_accelerator_deep_dive.md)** | **Triton Accelerator Deep Dive** | SPMD GPU memory models, coalesced global memory access, tiled pointer arithmetic, `@triton.jit` transpilation, and dynamic grid synthesis (`cdiv`). |
| **[Module 05](05_distributed_systems_ddp_fsdp.md)** | **Distributed Systems & DDP** | Multi-GPU collective communication (NCCL), `DistributedDataParallel` graph wrapping, rank-gated I/O, deterministic `DistributedSampler` epoch shuffling, and multi-node scaling. |
| **[Module 06](06_static_analysis_and_shape_engine.md)** | **Static Analysis & Shape Engine** | Abstract interpretation of tensor operations, broadcast widening matrices, spatial dimension computation for Conv/Linear/Pool/Attention, and compile-time error trapping (<3ms). |
| **[Module 07](07_compiler_internals_and_extending.md)** | **Compiler Internals & Extending** | Architecture of hand-written lexer and recursive-descent parser, AST traversal visitors, designing custom backend targets (C++/CUDA, JAX, ONNX), and contributing. |

---

## 🛠️ CLI Quick Reference for Engineers

```bash
# Static verification & type checking
python -m tensorloom check path/to/script.tl

# Structural AST inspection & parameter calculation
python -m tensorloom info path/to/model.nml

# Emit debug token stream
python -m tensorloom tokens path/to/script.tl

# Dump Abstract Syntax Tree (AST)
python -m tensorloom ast path/to/script.tl

# Production code generation
python -m tensorloom compile path/to/script.tl -o build/compiled.py
```
