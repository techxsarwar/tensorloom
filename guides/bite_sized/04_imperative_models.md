# ⚡ Lesson 04: Imperative Models in `.tl` (2 Min)

Define custom neural network layers and forward passes directly inside `.tl` scripts with `model`.

---

### 1. Basic Model Structure
```
model MLP:
    layer fc1 = Linear(784, 256)
    layer fc2 = Linear(256, 10)

    fn forward(self, x: Tensor) -> Tensor:
        return x |> self.fc1 |> relu |> self.fc2
```

---

### 2. Multi-Layer Residual Block
```
model ResBlock:
    layer conv1 = Conv2d(64, 64, kernel_size=3, padding=1)
    layer bn1   = BatchNorm2d(64)
    layer conv2 = Conv2d(64, 64, kernel_size=3, padding=1)
    layer bn2   = BatchNorm2d(64)

    fn forward(self, x: Tensor) -> Tensor:
        let residual = x
        x = x |> self.conv1 |> self.bn1 |> relu
        x = x |> self.conv2 |> self.bn2
        return relu(x + residual)
```

---

### 3. What the Compiler Generates
- A complete Python class inheriting from `torch.nn.Module`.
- Automatic initialization of layer submodules in `__init__`.
- Forward method mapping.

---

### 💡 Key Takeaway
`model` defines layers cleanly. `fn forward` defines how data flows through them.
