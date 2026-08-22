# 📐 Module 02: Formal NML Architecture Specification

This module covers the formal specification of the **Neural Markup Language (`.nml`)**, its Intermediate Representation (IR), AST lowering rules, and polymorphic constructor synthesis.

---

## 1. Formal Grammar of NML

An `.nml` file is a declarative domain-specific grammar parsed by `_parse_nml_model` in `src/tensorloom/parser/parser.py`:

```ebnf
NMLModel        ::= "@model" IDENTIFIER ":" NEWLINE INDENT NMLBlock* DEDENT

NMLBlock        ::= NMLConfigBlock
                  | NMLLayersBlock
                  | NMLForwardBlock

NMLConfigBlock  ::= "@config" ":" NEWLINE INDENT (IDENTIFIER "=" Literal NEWLINE)* DEDENT

NMLLayersBlock  ::= "@layers" ":" NEWLINE INDENT (IDENTIFIER "=" LayerInstantiation NEWLINE)* DEDENT

NMLForwardBlock ::= "@forward" "(" ParamList? ")" ":" NEWLINE INDENT Statement* DEDENT
```

---

## 2. AST Representation: `NMLModel` Node

The compiler parses an NML specification into the `NMLModel` AST node (`src/tensorloom/parser/ast_nodes.py`):

```python
@dataclass
class NMLModel(ASTNode):
    name: str
    config: dict[str, Any] = field(default_factory=dict)
    layers: list[NMLLayer] = field(default_factory=list)
    forward_params: list[str] = field(default_factory=list)
    forward_body: list[ASTNode] = field(default_factory=list)

@dataclass
class NMLLayer(ASTNode):
    name: str
    layer_type: str
    args: list[ASTNode] = field(default_factory=list)
    kwargs: dict[str, ASTNode] = field(default_factory=dict)
```

---

## 3. The Transpilation Engine & AST Rewrites

When lowering `NMLModel` to target PyTorch code (`PyTorchBackend._emit_nml_model`), the compiler executes three deterministic transformation passes:

```
┌─────────────────────────────────────────────────────────────┐
│                       NMLModel AST                          │
├──────────────────────────────┬──────────────────────────────┤
│ Pass 1: Config Synthesis     │ @config key=val              │
│                              │   ➔ def __init__(..., k=v):  │
│                              │   ➔ self.k = k               │
├──────────────────────────────┼──────────────────────────────┤
│ Pass 2: Scope Resolution     │ Layer args matching @config  │
│                              │   ➔ self.config_val          │
├──────────────────────────────┼──────────────────────────────┤
│ Pass 3: Forward Dispatch     │ Identifier calls in @forward │
│                              │   ➔ self.layer(args)         │
└──────────────────────────────┴──────────────────────────────┘
```

### 3.1 Pass 1: `__init__` Parameter Signature Synthesis
Every key-value pair in `node.config` is synthesized into overridable default keyword arguments in `__init__`:
```python
# AST: config = {"d_model": 512, "n_heads": 8, "dropout_rate": 0.1}
# Generated:
def __init__(self, d_model=512, n_heads=8, dropout_rate=0.1):
    super().__init__()
    self.d_model = d_model
    self.n_heads = n_heads
    self.dropout_rate = dropout_rate
```

### 3.2 Pass 2: Config Reference Rewriting
Layer arguments that match names in `node.config` are rewritten to `self.<name>`:
```python
# AST: LayerInst(type="MultiheadAttention", args=[Variable("d_model"), Variable("n_heads")])
# Generated:
self.attention = nn.MultiheadAttention(self.d_model, self.n_heads)
```

### 3.3 Pass 3: Forward Call Prefixing
Statements in `node.forward_body` are walked recursively. Any `FunctionCall` targeting a declared layer is rewritten to a bound method call:
```python
# AST: FunctionCall(target="norm1", args=[Variable("x")])
# Generated:
x = self.norm1(x)
```

---

## 4. Polymorphic Instantiation & Constructor Overrides

Because `@config` defaults are lowered to Python keyword arguments, downstream scripts can polymorphically instantiate the model with arbitrary architectural scaling:

```python
# Standard Base Model
base_model = TransformerBlock()

# Scaled Down for Edge Deployment (Mobile / IoT)
edge_model = TransformerBlock(d_model=128, n_heads=2, dropout_rate=0.0)

# Scaled Up for Large-Scale Training (LLM Tier)
huge_model = TransformerBlock(d_model=4096, n_heads=32, dropout_rate=0.15)
```
