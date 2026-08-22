# ⚡ Lesson 01: Hello TensorLoom (1 Min)

TensorLoom turns clean `.tl` scripts and `.nml` blueprints into high-performance, fused PyTorch code.

---

### 1. The 10-Line Complete Program (`hello.tl`)
```
model TinyNet:
    layer fc = Linear(784, 10)
    fn forward(self, x: Tensor) -> Tensor:
        return x |> self.fc

let net = TinyNet()
let data = load_dataset()

train net on data:
    epochs = 3
    optimizer = Adam(lr=0.001)
    loss = CrossEntropy
```

---

### 2. The 3 Essential CLI Commands
```bash
# 1. Inspect model size & memory
python -m tensorloom info hello.tl

# 2. Transpile to clean PyTorch Python
python -m tensorloom compile hello.tl -o hello.py

# 3. Compile and execute immediately
python -m tensorloom run hello.tl
```

---

### 💡 Key Takeaway
You write high-level architectures and training directives; TensorLoom handles CUDA placement, kernel fusion, and PyTorch boilerplate automatically.
