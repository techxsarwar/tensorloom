# ⚡ TensorLoom Bite-Sized Micro-Lessons (2-Min Each)

Fast, punchy, zero-fluff lessons to master TensorLoom in under 20 minutes.

---

## 🧭 Micro-Lesson Map

| # | Lesson | Read Time | What You'll Learn |
| :-: | :--- | :-: | :--- |
| **01** | [**Hello TensorLoom**](01_hello_tensorloom.md) | 1 min | The 30-second compiler tour & CLI commands |
| **02** | [**Variables & Types**](02_variables_and_types.md) | 1 min | Declaring data with `let` and type hints |
| **03** | [**The Pipe Operator (`\|>`)**](03_pipe_operator.md) | 1 min | Clean left-to-right functional dataflow |
| **04** | [**Imperative Models**](04_imperative_models.md) | 2 min | Writing `model` classes with `layer` and `fn forward` |
| **05** | [**Declarative NML**](05_declarative_nml.md) | 2 min | The 4 blocks of `.nml` blueprints |
| **06** | [**The `train` Block**](06_train_block.md) | 1 min | Automated training loops, AMP, & checkpoints |
| **07** | [**Cross-File Imports**](07_cross_file_imports.md) | 1 min | `import arch.nml as Alias` with overrides |
| **08** | [**Inline Triton Kernels**](08_triton_kernels.md) | 2 min | Custom GPU accelerator code with `@kernel` |
| **09** | [**Multi-GPU DDP**](09_distributed_ddp.md) | 1 min | 1-line cluster scaling with `distributed = true` |
| **10** | [**Shape Checking & Info**](10_shape_checking.md) | 1 min | Catching bugs in <3ms with `tlc check` & `tlc info` |

---

### 💻 Quick Start Command
```bash
python -m tensorloom run examples/mnist.tl
```
