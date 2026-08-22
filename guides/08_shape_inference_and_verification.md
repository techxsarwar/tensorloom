# 🔍 Guide 08: The Shape-Sorting Toy (Static Verification)

> *Remember the wooden toy with square, circle, and triangle holes when you were a toddler? You learned early on: a square block simply will NOT fit into a triangle hole. TensorLoom does the exact same thing for numbers inside your neural network!*

---

## 🕳️ Why Tensor Dimensions Matter

Inside a neural network, numbers travel in packages called **Tensors** (multi-dimensional grids of numbers).

Every layer has an **input doorway** and an **output doorway**:
- If Layer 1 produces a box with **64 numbers**...
- But Layer 2 is expecting a box with **128 numbers**...
- **CRASH!** The numbers don't fit through the doorway!

```
     Layer 1 Output                Layer 2 Input
   ┌─────────────────┐           ┌─────────────────┐
   │   64 Numbers    │    ❌     │   128 Numbers   │
   │  [■ ■ ■ ■ ■ ■]  │  =======> │ [■ ■ ■ ■ ■ ■ ■] │
   └─────────────────┘           └─────────────────┘
                   "DOORWAY MISMATCH!"
```

In normal PyTorch, Python doesn't check the doorways ahead of time. It waits until you run the code, loads 50 Gigabytes of images, warms up the GPU, and **then crashes 10 minutes later**!

---

## 🦅 How TensorLoom's Eagle Eye Works

TensorLoom has an internal **Static Shape Inference Engine**. As soon as you hit `compile` or `check`, the compiler traces the path of your tensors from start to finish in **under 3 milliseconds**!

```
[Input Picture: (Batch, 3, 224, 224)]
       │
       ▼ Conv2d(3, 64, kernel=7, stride=2, padding=3)
[Shape is now: (Batch, 64, 112, 112)]
       │
       ▼ MaxPool2d(kernel=3, stride=2, padding=1)
[Shape is now: (Batch, 64, 56, 56)]
       │
       ▼ Flatten(start_dim=1)
[Shape is now: (Batch, 200,704)]
       │
       ▼ Linear(200704, 1000)
[Output Shape: (Batch, 1000)]
```

If you accidentally write `Linear(1024, 1000)` instead of `200704`, TensorLoom catches it **INSTANTLY**:

```
❌ ShapeError at line 14:
   Dimension mismatch in layer 'classifier':
   Expected input features: 1024
   Actual input features:   200704
   Compile time: 2.3ms
```

---

## 🩺 The Doctor's Checkup (`tlc check`)

Before running your training pipeline, you can run a doctor's health check on your code:

```bash
python -m tensorloom check my_model.tl
```

If everything passes, it gives you the green light in **0.005 seconds**!

---

## 📊 The Memory Weighing Scale (`tlc info`)

Ever wonder how big your model is before renting an expensive cloud server?

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
   Total Parameters: 86,567,656 (86.57M tiny gears)
   Trainable:        86,567,656
   Estimated Memory: ~330.23 MB (FP32) / ~165.12 MB (FP16)
   Status:           Valid & Ready to Train!
```

It calculates the exact memory footprint in FP32 and FP16 modes so you never run out of GPU memory (`CUDA Out of Memory`) mid-training! 🛡️✨
