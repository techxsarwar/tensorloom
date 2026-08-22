// TensorLoom — Hello World
// Basic tensor operations demonstrating the language fundamentals

let x = tensor([1.0, 2.0, 3.0], dtype=float32, device=gpu)
let y = tensor([4.0, 5.0, 6.0], dtype=float32, device=gpu)

// Native matmul + bias — fused into a single GPU kernel
let z = x @ y + 0.5

// Element-wise operations
let scaled = z * 2.0
let activated = scaled |> relu

print(activated)
