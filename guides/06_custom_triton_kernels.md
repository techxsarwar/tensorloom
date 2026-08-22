# ⚡ Guide 06: The Nitro Turbo Button (Inline Triton GPU Kernels)

> *Most of the time, driving a car in automatic mode is great. But when you enter an F1 championship race, you want direct manual control of the twin-turbo supercharger! That is what `@kernel` does for your GPU graphics card.*

---

## 🎮 What is a GPU Kernel?

Inside your computer's graphics card (GPU), there aren't just 4 or 8 CPU processors. There are **thousands of tiny computing cores** all working at the exact same time!

Think of a GPU like a **huge factory with 10,000 tiny worker elves**:
- If you ask 1 elf to paint 10,000 toy cars, it takes forever.
- But if you hand 1 car to each of the 10,000 elves, **all 10,000 cars get painted in 1 second!**

A **Kernel** is the instruction sheet you give to those 10,000 elves so they all work simultaneously without bumping into each other.

---

## ✍️ Writing a Custom GPU Kernel in TensorLoom

Instead of writing complicated C++ or low-level NVIDIA CUDA code, TensorLoom lets you write clean **OpenAI Triton GPU kernels** directly inside your `.tl` file with `@kernel`:

```
@kernel def vector_add(x_ptr, y_ptr, z_ptr, n, BLOCK: tl.constexpr):
    // 1. Which elf number am I? (Program ID)
    let pid = tl.program_id(axis=0)
    
    // 2. Which slice of numbers should my team of elves work on?
    let offsets = pid * BLOCK + tl.arange(0, BLOCK)
    
    // 3. Make sure we don't read past the end of the list!
    let mask = offsets < n
    
    // 4. Grab numbers from GPU memory
    let x = tl.load(x_ptr + offsets, mask=mask)
    let y = tl.load(y_ptr + offsets, mask=mask)
    
    // 5. Add them together and store the answer in the result list!
    tl.store(z_ptr + offsets, x + y, mask=mask)
```

---

## 🤖 The Automated Launcher: No Math Homework for You!

In normal GPU programming, running a kernel is annoying because you have to calculate the **Grid Dimensions**:
> *"If I have 1,000,000 numbers and each block holds 1024 numbers, how many worker blocks do I need to launch?"*

TensorLoom's compiler does that math for you automatically! It creates a Python launcher function:

```python
def vector_add_launcher(x, y, BLOCK=1024):
    n = x.numel()
    z = torch.empty_like(x)  # Allocates output tensor
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)  # Auto grid division!
    _vector_add_kernel[grid](x, y, z, n, BLOCK=BLOCK)
    return z
```

Now, anywhere in your Python code, you can just call:

```python
result = vector_add_launcher(tensor_a, tensor_b)
```

And your 10,000 GPU elves will compute the answer at bare-metal memory bandwidth speed! 🚀
