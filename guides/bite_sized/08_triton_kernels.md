# ⚡ Lesson 08: Inline Triton Kernels (2 Min)

Write bare-metal GPU accelerator code directly inside your `.tl` scripts with `@kernel`.

---

### 1. Vector Addition Kernel
```
@kernel def vector_add(x_ptr, y_ptr, z_ptr, n, BLOCK: tl.constexpr):
    let pid = tl.program_id(axis=0)
    let offsets = pid * BLOCK + tl.arange(0, BLOCK)
    let mask = offsets < n
    let x = tl.load(x_ptr + offsets, mask=mask)
    let y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(z_ptr + offsets, x + y, mask=mask)
```

---

### 2. What TensorLoom Generates
1. **`@triton.jit` Kernel**: The compiled Triton function.
2. **Auto Launcher Function**: Allocates the output tensor and calculates grid size using `triton.cdiv(n, BLOCK)`:

```python
# Call directly in Python:
z = vector_add_launcher(tensor_a, tensor_b)
```

---

### 3. In-Place Scaling Example
```
@kernel def scale(x_ptr, alpha, n, BLOCK: tl.constexpr):
    let pid = tl.program_id(axis=0)
    let offsets = pid * BLOCK + tl.arange(0, BLOCK)
    let mask = offsets < n
    let x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(x_ptr + offsets, x * alpha, mask=mask)
```

---

### 💡 Key Takeaway
No C++ bindings or CUDA needed. Write Triton kernels inline; TensorLoom synthesizes the grid launchers automatically.
