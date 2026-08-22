# ⚡ Guide 02: 5-Minute Quickstart Tutorial

In this guide, you will learn how to write, inspect, compile, and execute your very first TensorLoom program.

---

## 1. Installation

TensorLoom can be installed via `pip`:

```bash
pip install tensorloom
```

To verify the installation:

```bash
python -m tensorloom --help
```

---

## 2. Your First TensorLoom Program (`hello_mnist.tl`)

Create a new file named `hello_mnist.tl`:

```
// hello_mnist.tl — Simple Digit Classifier

model MNISTClassifier:
    layer fc1 = Linear(784, 128)
    layer fc2 = Linear(128, 10)

    fn forward(self, x: Tensor) -> Tensor:
        return x |> self.fc1 |> relu |> self.fc2

let net = MNISTClassifier()
let data = load_dataset()

train net on data:
    epochs = 5
    optimizer = Adam(lr=0.001)
    loss = CrossEntropy
    precision = fp16
```

---

## 3. Inspecting the Model (Static Analysis)

Before executing or compiling, you can inspect the model structure, parameter counts, and memory layout:

```bash
python -m tensorloom info hello_mnist.tl
```

**Output:**
```
  +================================================+
  |   TensorLoom Compiler  v0.1.0                  |
  |   GPU-Efficient Language for AI Training        |
  +================================================+

[info] Model Summary: MNISTClassifier
   Total Parameters: 101,770 (0.10M)
   Trainable:        101,770
   Estimated Memory: ~0.39 MB (FP32) / ~0.19 MB (FP16)
   Status:           Valid
```

---

## 4. Compiling to PyTorch Code

To transpile your `.tl` script into standard PyTorch Python:

```bash
python -m tensorloom compile hello_mnist.tl -o hello_mnist_compiled.py
```

Inspect `hello_mnist_compiled.py` to see the generated PyTorch code:
- Automatic CUDA device detection (`torch.device("cuda" if ...)`).
- Model class extending `nn.Module`.
- Automatic Kernel Fusion via `torch.compile(net, mode="max-autotune")`.
- Training loop instrumented with `autocast` and `GradScaler`.

---

## 5. Running the Program

You can execute the compiled script directly with Python:

```bash
python hello_mnist_compiled.py
```

Or compile and execute in a single command using TensorLoom's `run` action:

```bash
python -m tensorloom run hello_mnist.tl
```

Congratulations! You've just written, compiled, and executed your first TensorLoom model.
