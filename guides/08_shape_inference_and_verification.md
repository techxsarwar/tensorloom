# 🔍 Guide 08: Compile-Time Static Shape Inference & Profiling

One of the most frustrating aspects of dynamic deep learning frameworks is discovering a shape mismatch error (e.g. `RuntimeError: mat1 and mat2 shapes cannot be multiplied (64x256 and 128x10)`) only at runtime.

TensorLoom's **Static Shape Inference Engine** prevents this by calculating spatial tensor dimensions at compile time in milliseconds.

---

## 1. How Shape Inference Works

As the compiler walks the Abstract Syntax Tree (AST), the Analyzer computes the output shape of every layer and operation:

```
[Input Tensor: (B, 3, 224, 224)]
       │
       ▼ Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
[Shape: (B, 64, 112, 112)]
       │
       ▼ MaxPool2d(kernel_size=3, stride=2, padding=1)
[Shape: (B, 64, 56, 56)]
       │
       ▼ Flatten(start_dim=1)
[Shape: (B, 200704)]
       │
       ▼ Linear(200704, 1000)
[Output Shape: (B, 1000)]
```

If an intermediate layer defines an incompatible dimension (e.g. `Linear(1024, 10)` following a layer producing `200704`), the compiler halts immediately:

```
❌ ShapeError at line 14:
   Dimension mismatch in layer 'fc':
   Expected input features: 1024
   Actual input features:   200704
   Compile time: 2.3ms
```

---

## 2. Model Inspection (`tlc info`)

Use the `info` command to inspect total parameters, trainable weights, and estimated activation memory before allocating GPU resources:

```bash
python -m tensorloom info examples/vision_transformer.nml
```

**Output:**
```
  +================================================+
  |   TensorLoom Compiler  v0.1.0                  |
  |   GPU-Efficient Language for AI Training        |
  +================================================+

[info] Model Summary: ViT
   Total Parameters: 86,567,656 (86.57M)
   Trainable:        86,567,656
   Estimated Memory: ~330.23 MB (FP32) / ~165.12 MB (FP16)
   Status:           Valid
```

---

## 3. Static Type Checking (`tlc check`)

To validate an entire project without generating code:

```bash
python -m tensorloom check examples/train_resnet.tl
```

This ensures that all foreign `.nml` symbols exist, keyword arguments match the target `@config`, and all tensor operations are valid.
