# 🛠️ Guide 04: Imperative `.tl` Scripting & Pipelines

TensorLoom (`.tl`) scripts are imperative programs that define data loading, model instantiations, execution workflows, and GPU training blocks.

---

## 1. Variables & Types

Variables in `.tl` are declared with `let`:

```
let lr = 0.001
let batch_size = 64
let device_name = "cuda"
let is_training = true
let layer_dims = [784, 256, 128, 10]
```

---

## 2. Functional Pipe Operator (`|>`)

TensorLoom features a first-class pipe operator (`|>`) that chains transformations linearly without messy nested function calls:

```
// Traditional nested code:
// y = relu(bn(conv(x)))

// TensorLoom Pipe notation:
let y = x |> conv |> bn |> relu
```

During compilation, the pipe operator is desugared into optimized nested calls that serve as clean fusion targets for `torch.compile`.

---

## 3. Inline Model Definitions

You can declare models directly inside `.tl` scripts:

```
model ResBlock:
    layer conv1 = Conv2d(64, 64, kernel_size=3, padding=1)
    layer bn1   = BatchNorm2d(64)
    layer conv2 = Conv2d(64, 64, kernel_size=3, padding=1)
    layer bn2   = BatchNorm2d(64)

    fn forward(self, x: Tensor) -> Tensor:
        let residual = x
        x = x |> self.conv1 |> self.bn1 |> relu
        x = x |> self.conv2 |> self.bn2
        return relu(x + residual)
```

---

## 4. The High-Level `train` Block

The `train` block represents a complete training loop with automated GPU resource management:

```
train model_name on dataset_name:
    epochs = 20
    optimizer = Adam(lr=0.001, weight_decay=1e-4)
    loss = CrossEntropy
    precision = fp16
    checkpoint every 5 epochs
    distributed = true
```

### Supported Training Directives:

| Directive | Options | Description |
| :--- | :--- | :--- |
| `epochs` | Integer (e.g. `10`) | Total number of training epochs |
| `optimizer` | `Adam`, `SGD`, `AdamW`, `RMSprop` | Optimizer and hyperparameters |
| `loss` | `CrossEntropy`, `MSE`, `BCE`, `L1` | Criterion function |
| `precision` | `fp16`, `bf16`, `fp32` | Mixed precision execution mode |
| `checkpoint` | `checkpoint every N epochs` | Automated model snapshotting |
| `distributed` | `true`, `false` | Multi-GPU Distributed Data Parallel scaling |
