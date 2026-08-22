# 📐 Guide 03: Declarative Neural Markup Language (`.nml`)

## What is NML?

**Neural Markup Language (`.nml`)** is TensorLoom's declarative architecture definition language. 

While imperative scripts (`.tl`) handle training loops, datasets, and execution hardware, `.nml` files serve as pure, reusable **architectural blueprints**.

---

## The 4-Block NML Architecture

An `.nml` model is declared using four structured sections:

```
@model ModelName:
    @config:
        // Default hyperparameters & constructor arguments
        param = value

    @layers:
        // Layer declarations referencing config values
        layer_name = LayerType(args...)

    @forward(inputs...):
        // Forward execution computation graph
        statements...
        return output
```

---

## Comprehensive Example: Transformer Block

Create `transformer.nml`:

```
@model TransformerBlock:
    @config:
        d_model = 512
        n_heads = 8
        dropout_rate = 0.1

    @layers:
        attention = MultiHeadAttention(d_model, n_heads)
        norm1     = LayerNorm(d_model)
        norm2     = LayerNorm(d_model)
        ff        = Linear(d_model, d_model)
        dropout   = Dropout(dropout_rate)

    @forward(x):
        let residual = x
        x = norm1(x)
        x = residual + dropout(attention(x))
        let residual2 = x
        x = norm2(x)
        x = residual2 + dropout(ff(x))
        return x
```

---

## The 3 Intelligent Compiler Transformations

When the TensorLoom compiler processes an `.nml` file, it performs three transformations behind the scenes:

### 1. Config → Overridable `__init__` Kwargs
Every key in `@config` becomes a keyword parameter with a default value. When instantiating the class, users can override any default:
```python
# Generated __init__
def __init__(self, d_model=512, n_heads=8, dropout_rate=0.1):
    super().__init__()
    self.d_model = d_model
    self.n_heads = n_heads
    self.dropout_rate = dropout_rate
```

### 2. Variable Scope Resolution to `self.`
Any reference to a config parameter inside `@layers` is automatically resolved to `self.<param>`:
```python
# MultiHeadAttention(d_model, n_heads) becomes:
self.attention = nn.MultiheadAttention(self.d_model, self.n_heads)
```

### 3. Forward Call Auto-Prefixing
Calls to declared layers in `@forward` are automatically prefixed with `self.`:
```python
# x = norm1(x) becomes:
x = self.norm1(x)
```

---

## Instantiating and Overriding NML Models

You can use NML models directly in Python or cross-file in `.tl` scripts with custom overrides:

```python
# Use all defaults (d_model=512, n_heads=8)
block1 = TransformerBlock()

# Override for a small model
small_block = TransformerBlock(d_model=256, n_heads=4)

# Override for a large model
large_block = TransformerBlock(d_model=1024, n_heads=16, dropout_rate=0.2)
```
