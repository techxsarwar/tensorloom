# ⚡ Lesson 03: The Pipe Operator (`|>`) (1 Min)

The pipe operator (`|>`) passes the result of the left expression directly into the function or layer on the right.

---

### 1. Comparison

#### ❌ The Ugly Nested Way (Hard to read backwards)
```python
output = relu(bn(conv(x)))
```

#### ✅ The TensorLoom Pipe Way (Clean left-to-right flow)
```
let output = x |> conv |> bn |> relu
```

---

### 2. Chaining Multiple Layers
```
let final_logits = x |> self.layer1 |> relu |> self.layer2 |> relu |> self.layer3
```

---

### 3. Passing Additional Arguments
If the target function takes arguments, pass them in parentheses:

```
let normed = x |> self.dropout(p=0.2) |> self.norm
```

---

### 💡 Key Takeaway
`|>` reads like a pipeline. The compiler automatically desugars it into optimized nested calls for `torch.compile` fusion.
