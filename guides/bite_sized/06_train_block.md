# ⚡ Lesson 06: The `train` Block (1 Min)

The `train` block eliminates 40+ lines of PyTorch training loop boilerplate.

---

### 1. Complete Training Block
```
let net = MyModel()
let data = load_dataset()

train net on data:
    epochs = 20
    optimizer = Adam(lr=0.001)
    loss = CrossEntropy
    precision = fp16
    checkpoint every 5 epochs
```

---

### 2. Available Directives

| Directive | Example | What It Does |
| :--- | :--- | :--- |
| `epochs` | `epochs = 10` | Number of training epochs |
| `optimizer` | `Adam(lr=1e-3)` | Optimizer & learning rate |
| `loss` | `CrossEntropy`, `MSE` | Loss function |
| `precision` | `fp16`, `bf16`, `fp32` | Automatic Mixed Precision |
| `checkpoint` | `checkpoint every 5 epochs` | Periodic model saving |
| `distributed` | `distributed = true` | Multi-GPU cluster scaling |

---

### 💡 Key Takeaway
Write 6 lines of configuration; TensorLoom generates the complete training loop with GPU memory transfers, loss scaling, metric tracking, and checkpointing.
