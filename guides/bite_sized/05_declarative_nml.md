# ⚡ Lesson 05: Declarative NML (`.nml`) (2 Min)

`.nml` files are reusable, declarative **architectural blueprints**.

---

### 1. The 4 Magic Blocks (`transformer.nml`)
```
@model TransformerBlock:
    @config:
        d_model = 512
        n_heads = 8

    @layers:
        attention = MultiHeadAttention(d_model, n_heads)
        norm      = LayerNorm(d_model)
        ff        = Linear(d_model, d_model)

    @forward(x):
        let residual = x
        x = norm(x)
        x = residual + attention(x)
        return ff(x)
```

---

### 2. The 3 Automatic Rewrites
1. **`@config` ➔ `__init__` kwargs**: `def __init__(self, d_model=512, n_heads=8):`
2. **`@layers` scope resolution**: `MultiHeadAttention(d_model, ...)` becomes `nn.MultiheadAttention(self.d_model, ...)`
3. **`@forward` auto-calling**: `norm(x)` becomes `self.norm(x)`

---

### 3. Polymorphic Instantiation
Override any `@config` default when creating the model:

```python
# Small model
small = TransformerBlock(d_model=256, n_heads=4)

# Giant model
large = TransformerBlock(d_model=2048, n_heads=16)
```

---

### 💡 Key Takeaway
NML separates the *structure* of a model from the *code that trains it*.
