# ⚡ Lesson 02: Variables & Types (1 Min)

Variables in TensorLoom are declared using `let`.

---

### 1. Basic Declarations
```
let lr = 0.001                      // Float
let batch_size = 64                 // Integer
let model_name = "Transformer"      // String
let use_cuda = true                 // Boolean
let layer_sizes = [784, 128, 10]    // List
```

---

### 2. Type Annotations (Optional)
You can add explicit type annotations for compiler verification:

```
let lr: Float = 0.001
let epochs: Int = 10
let device: String = "cuda:0"
let x: Tensor[Batch, 784] = load_batch()
```

---

### 3. Tensor Literals
Create constant tensors directly in code:

```
let bias = [0.1, 0.2, 0.3]
let weights = [[1.0, 2.0], [3.0, 4.0]]
```

---

### 💡 Key Takeaway
`let` is your universal binding keyword. Variables are immutable by default and infer their types automatically.
