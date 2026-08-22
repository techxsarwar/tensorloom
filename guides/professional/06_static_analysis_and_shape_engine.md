# 🔍 Module 06: Static Analysis & Shape Inference Engine

This module provides the formal specification of TensorLoom’s static shape analysis engine (`src/tensorloom/analyzer/shape_inference.py`), abstract interpretation rules for tensor calculus, broadcast compatibility algorithms, and compile-time error diagnostics.

---

## 1. Abstract Interpretation & Shape Representation

TensorLoom models multi-dimensional tensor geometries through the `Shape` algebraic data structure:

```python
class Shape:
    def __init__(self, *dims: int | str):
        self.dims = tuple(dims)
        self.ndim = len(self.dims)

    def is_scalar(self) -> bool:
        return self.ndim == 0 or (self.ndim == 1 and self.dims[0] == 1)

    def is_known(self) -> bool:
        return all(isinstance(d, int) for d in self.dims)
```

Symbolic dimensions (such as dynamic batch sizes like `B` or `"batch"`) are preserved symbolically during abstract interpretation.

---

## 2. Spatial Dimension Propagation Rules

The analyzer maintains formal transition transfer functions $f_{\text{layer}}: \text{Shape} \to \text{Shape}$ across standard deep learning operators:

### 2.1 2D Convolution (`Conv2d`)
Given input tensor $(B, C_{\text{in}}, H_{\text{in}}, W_{\text{in}})$ with parameters $(C_{\text{out}}, K, S, P)$:

$$H_{\text{out}} = \left\lfloor \frac{H_{\text{in}} + 2P - K}{S} \right\rfloor + 1$$

$$W_{\text{out}} = \left\lfloor \frac{W_{\text{in}} + 2P - K}{S} \right\rfloor + 1$$

$$\text{Shape}_{\text{out}} = (B, C_{\text{out}}, H_{\text{out}}, W_{\text{out}})$$

### 2.2 Linear / Fully-Connected (`Linear`)
Given input $(B, \dots, D_{\text{in}})$ with weight matrix $W \in \mathbb{R}^{D_{\text{out}} \times D_{\text{in}}}$:

$$\text{Condition: } \text{Last dimension of } X = D_{\text{in}}$$

$$\text{Shape}_{\text{out}} = (B, \dots, D_{\text{out}})$$

### 2.3 Multi-Head Attention (`MultiheadAttention`)
Given query, key, value projections with embedding dimension $D_{\text{model}}$:

$$\text{Condition: } D_Q = D_K = D_V = D_{\text{model}}$$

$$\text{Shape}_{\text{out}} = (B, \text{SeqLen}, D_{\text{model}})$$

---

## 3. NumPy-Style Broadcast Compatibility Algorithm

When binary operations ($+$, $-$, $*$, $/$) are evaluated on non-identical shapes, the analyzer verifies broadcast compatibility according to the standard multidimensional broadcasting rules:

```python
def broadcast_shapes(s1: Shape, s2: Shape) -> Shape:
    """Computes output shape of elementwise binary operation under broadcasting."""
    dims1 = list(s1.dims)
    dims2 = list(s2.dims)
    
    # Right-align dimensions by prepending 1s
    max_len = max(len(dims1), len(dims2))
    dims1 = [1] * (max_len - len(dims1)) + dims1
    dims2 = [1] * (max_len - len(dims2)) + dims2
    
    out_dims = []
    for d1, d2 in zip(dims1, dims2):
        if d1 == d2:
            out_dims.append(d1)
        elif d1 == 1:
            out_dims.append(d2)
        elif d2 == 1:
            out_dims.append(d1)
        else:
            raise ShapeMismatchError(f"Cannot broadcast dimension {d1} with {d2}")
            
    return Shape(*out_dims)
```

---

## 4. Compile-Time Diagnostics

The Analyzer intercepts shape violations during compilation, providing high-precision source location traces:

```
[compile] Checking: models/vit.tl
❌ ShapeError at line 18, column 12:
   Dimension mismatch in matrix multiplication:
   Cannot multiply matrix A (shape: [64, 512]) by matrix B (shape: [768, 1000])
   Inner dimensions 512 and 768 do not match.
   Compile time: 2.1ms
```
