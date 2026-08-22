// vector_add.tl — Custom Triton Kernel in TensorLoom
//
// This demonstrates TensorLoom's @kernel syntax for inline GPU
// programming via OpenAI Triton. The compiler automatically:
//   1. Emits a @triton.jit decorated kernel function
//   2. Auto-generates a launcher wrapper with grid calculation
//   3. Handles pointer/tensor conversion seamlessly
//   4. Imports triton and triton.language as tl automatically
//
// The transpiled output can be executed directly on CUDA GPUs
// with massive parallelism across thousands of streaming cores.

@kernel def vector_add(x_ptr, y_ptr, z_ptr, num_elements, BLOCK_SIZE: tl.constexpr):
    let pid = tl.program_id(axis=0)
    let block_start = pid * BLOCK_SIZE
    let offsets = block_start + tl.arange(0, BLOCK_SIZE)
    let mask = offsets < num_elements

    let x = tl.load(x_ptr + offsets, mask=mask)
    let y = tl.load(y_ptr + offsets, mask=mask)
    let z = x + y

    tl.store(z_ptr + offsets, z, mask=mask)
