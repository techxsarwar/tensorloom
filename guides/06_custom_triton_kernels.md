# ⚡ Guide 06: Inline Triton GPU Accelerator Kernels

When standard PyTorch layers and operations are not fast enough, TensorLoom provides a direct **bare-metal escape hatch**: inline `@kernel` blocks powered by OpenAI Triton.

---

## 1. Defining a Kernel in TensorLoom

You can define custom GPU kernels directly inside any `.tl` file without writing C++ extensions or CUDA code:

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

## 2. What the Compiler Generates

When you compile the code above, the TensorLoom backend emits two components:

### 1. The `@triton.jit` Kernel
```python
@triton.jit
def _vector_add_kernel(x_ptr, y_ptr, z_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = ((pid * BLOCK) + tl.arange(0, BLOCK))
    mask = (offsets < n)
    x = tl.load((x_ptr + offsets), mask=mask)
    y = tl.load((y_ptr + offsets), mask=mask)
    tl.store((z_ptr + offsets), (x + y), mask=mask)
```

### 2. The Automated Grid Launcher
The compiler inspects kernel arguments and generates an automated launcher function that calculates grid dimensions and allocates output tensors:

```python
def vector_add_launcher(x, y, BLOCK=1024):
    n = x.numel()
    z = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    _vector_add_kernel[grid](x, y, z, n, BLOCK=BLOCK)
    return z
```

---

## 3. In-Place Kernel Modification

If your kernel modifies the first tensor in place (e.g. a scaling kernel):

```
@kernel def scale_tensor(x_ptr, alpha, n, BLOCK: tl.constexpr):
    let pid = tl.program_id(axis=0)
    let offsets = pid * BLOCK + tl.arange(0, BLOCK)
    let mask = offsets < n
    let x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(x_ptr + offsets, x * alpha, mask=mask)
```

The launcher will modify the input tensor directly without allocating extra GPU memory.
