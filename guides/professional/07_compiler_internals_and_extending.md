# 🏗️ Module 07: Compiler Internals & Extending TensorLoom

This module provides a comprehensive guide for compiler engineers wishing to understand TensorLoom’s internal architecture, add new syntax tokens, implement AST transformation passes, or construct custom compilation backends (such as C++/CUDA, JAX, or ONNX).

---

## 1. Compiler Subsystem Architecture

```
                    ┌────────────────────────────┐
                    │      Source File (.tl)     │
                    └─────────────┬──────────────┘
                                  │
                                  ▼
                    ┌────────────────────────────┐
                    │      Lexer (Scanner)       │  src/tensorloom/lexer/
                    │  Zero-copy token generator │
                    └─────────────┬──────────────┘
                                  │ Token Stream
                                  ▼
                    ┌────────────────────────────┐
                    │  Recursive-Descent Parser  │  src/tensorloom/parser/
                    │  LL(k) predictive grammar  │
                    └─────────────┬──────────────┘
                                  │ Abstract Syntax Tree (AST)
                                  ▼
                    ┌────────────────────────────┐
                    │  Static Analysis Engine    │  src/tensorloom/analyzer/
                    │  • Shape Inference         │
                    │  • Type Checker            │
                    │  • DDP Feature Scanner     │
                    └─────────────┬──────────────┘
                                  │ Validated AST
                                  ▼
                    ┌────────────────────────────┐
                    │ Code Generation Backend(s) │  src/tensorloom/codegen/
                    │  • PyTorchBackend          │
                    │  • TritonEmitter           │
                    │  • (Custom Backends)       │
                    └─────────────┬──────────────┘
                                  │
                                  ▼
                    ┌────────────────────────────┐
                    │     Target Output (.py)    │
                    └────────────────────────────┘
```

---

## 2. Adding a New AST Node & Token

To extend TensorLoom with a new language feature:

### Step 1: Register the Token Type (`src/tensorloom/lexer/tokens.py`)
```python
class TokenType(Enum):
    # Add new keyword or operator
    LORA = "LORA"
```

### Step 2: Update the Lexer Scanner (`src/tensorloom/lexer/lexer.py`)
```python
KEYWORDS = {
    # ...
    "lora": TokenType.LORA,
}
```

### Step 3: Define the AST Node (`src/tensorloom/parser/ast_nodes.py`)
```python
@dataclass
class LoRAConfig(ASTNode):
    rank: int = 8
    alpha: float = 16.0
    target_modules: list[str] = field(default_factory=list)
```

### Step 4: Add Parser Dispatch (`src/tensorloom/parser/parser.py`)
```python
def _parse_lora_config(self) -> LoRAConfig:
    self._expect(TokenType.LORA)
    # Parse parameters...
    return LoRAConfig(rank=rank, alpha=alpha)
```

---

## 3. Implementing a Custom Code Generation Backend

All backend code generators implement a clean visitor pattern over the AST.

To build a custom target (e.g., generating JAX/Flax or raw C++/CUDA):

```python
from tensorloom.parser.ast_nodes import Program, ASTNode, ModelDefinition, LetStatement

class JAXBackend:
    """Compiles TensorLoom AST into JAX/Flax Python code."""
    def __init__(self):
        self.code: list[str] = []

    def generate(self, program: Program) -> str:
        self.code.append("import jax")
        self.code.append("import jax.numpy as jnp")
        self.code.append("from flax import linen as nn")
        
        for stmt in program.statements:
            self._emit_statement(stmt)
            
        return "\n".join(self.code)

    def _emit_statement(self, stmt: ASTNode):
        if isinstance(stmt, ModelDefinition):
            self._emit_flax_module(stmt)
        elif isinstance(stmt, LetStatement):
            self._emit_let(stmt)
```

---

## 4. Contributing & Development Workflow

### 4.1 Running the Complete Test Suite
```bash
python -m pytest tests/ -v --tb=short
```

### 4.2 Formatting & Linting
```bash
black src/ tests/
isort src/ tests/
```

### 4.3 Building Distribution Artifacts
```bash
python -m build
python -m twine check dist/*
```
