# ⚡ Lesson 10: Static Shape Checking & Info (1 Min)

Catch dimension bugs in **<3ms** before launching expensive GPU jobs.

---

### 1. Doctor's Checkup (`tlc check`)
Validate your entire model's layer dimensions and type flow:

```bash
python -m tensorloom check my_script.tl
```

If layer dimensions don't align, TensorLoom catches the mismatch instantly:
```
❌ ShapeError at line 14:
   Dimension mismatch in layer 'fc':
   Expected input features: 1024
   Actual input features:   200704
   Compile time: 2.3ms
```

---

### 2. Model Inspection (`tlc info`)
Calculate parameters and estimated GPU activation memory:

```bash
python -m tensorloom info examples/vision_transformer.nml
```

**Output:**
```
[info] Model Summary: ViT
   Total Parameters: 86,567,656 (86.57M)
   Trainable:        86,567,656
   Estimated Memory: ~330.23 MB (FP32) / ~165.12 MB (FP16)
   Status:           Valid & Ready!
```

---

### 💡 Key Takeaway
Never crash from a `Dimension mismatch` or `CUDA Out of Memory` error at runtime again.
