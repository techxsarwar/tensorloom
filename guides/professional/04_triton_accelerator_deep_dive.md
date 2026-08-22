# ⚡ Module 04: Bare-Metal GPU Accelerator Engineering with Triton

This module explores the systems architecture of TensorLoom’s inline `@kernel` blocks, Single Program Multiple Data (SPMD) execution, GPU memory hierarchies, pointer arithmetic, coalesced global memory access, and automatic grid calculation.

---

## 1. Modern GPU Memory Hierarchy & Tiled Programming

Modern GPUs (NVIDIA Ampere, Hopper, Blackwell) achieve peak compute through deep hierarchical memory architectures:

```
┌─────────────────────────────────────────────────────────────┐
│                 GPU Global Memory (HBM3/GDDR6)              │
│                High Capacity (~80GB), High Latency          │
└──────────────────────────────┬──────────────────────────────┘
                               │ Coalesced Memory Access (128-byte transactions)
┌──────────────────────────────▼──────────────────────────────┐
│                  Streaming Multiprocessor (SM)              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │           SRAM Shared Memory / L1 Cache (~228KB)       │ │
│  │           Extremely Low Latency, High Bandwidth        │ │
│  └───────────────────────────┬────────────────────────────┘ │
│                              │                              │
│  ┌───────────────────────────▼────────────────────────────┐ │
│  │              Tensor Cores / CUDA Vector ALU            │ │
│  │                Registers (64K 32-bit registers)        │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

Standard deep learning frameworks suffer from **memory bandwidth bottlenecks** (Memory-Bound Operations). Pointwise activations (like LayerNorm, Softmax, GELU, and residual additions) spend 90% of their execution time reading from and writing back to Global HBM memory.

Triton addresses this by enabling **Block-Tiled Computations** where data is loaded once into fast on-chip SRAM/Registers, computed upon, and written back in coalesced memory transactions.

---

## 2. TensorLoom `@kernel` Syntax & Transpilation

TensorLoom provides first-class support for inline Triton kernels via `@kernel`:

```
@kernel def fused_add_scale(x_ptr, y_ptr, z_ptr, alpha, n, BLOCK: tl.constexpr):
    let pid = tl.program_id(axis=0)
    let offsets = pid * BLOCK + tl.arange(0, BLOCK)
    let mask = offsets < n
    let x = tl.load(x_ptr + offsets, mask=mask)
    let y = tl.load(y_ptr + offsets, mask=mask)
    let out = (x + y) * alpha
    tl.store(z_ptr + offsets, out, mask=mask)
```

### 2.1 Compiler Transpilation to `@triton.jit`
The code generator lowers this into an optimized Triton JIT function:

```python
import triton
import triton.language as tl

@triton.jit
def _fused_add_scale_kernel(x_ptr, y_ptr, z_ptr, alpha, n, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = ((pid * BLOCK) + tl.arange(0, BLOCK))
    mask = (offsets < n)
    x = tl.load((x_ptr + offsets), mask=mask)
    y = tl.load((y_ptr + offsets), mask=mask)
    out = ((x + y) * alpha)
    tl.store((z_ptr + offsets), out, mask=mask)
```

---

## 3. Automated Grid Dispatcher & Launcher Synthesis

In raw Triton, calling a kernel requires manual tensor allocation and grid dimension calculation using ceiling division:

$$\text{Grid Size} = \left\lceil \frac{N}{\text{BLOCK}} \right\rceil = \text{triton.cdiv}(N, \text{BLOCK})$$

TensorLoom synthesizes this boilerplate automatically:

```python
def fused_add_scale_launcher(x, y, alpha, BLOCK=1024):
    n = x.numel()
    z = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
    _fused_add_scale_kernel[grid](x, y, z, alpha, n, BLOCK=BLOCK)
    return z
```

### Key Optimizations:
- **Zero Memory Allocation for In-Place Kernels**: If the kernel does not write to a separate output pointer, the launcher reuses the input tensor in-place.
- **Dynamic Meta-Grid Lambda**: Grid sizing is evaluated dynamically at runtime, allowing the kernel to support dynamic batch sizes and sequence lengths.
