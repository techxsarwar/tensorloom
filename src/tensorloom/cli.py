"""
TensorLoom CLI — The `tlc` command-line compiler driver.

Commands:
  tlc run <file.tl>                    Parse, compile, and execute
  tlc compile <file.tl> -o <out.py>    Transpile to Python only
  tlc check <file.tl>                  Type-check and validate
  tlc info <file.tl>                   Show estimated GPU memory usage
  tlc tokens <file.tl>                 Debug: dump token stream
  tlc ast <file.tl>                    Debug: dump AST
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

from tensorloom.lexer.lexer import Lexer, LexerError
from tensorloom.parser.parser import Parser, ParseError
from tensorloom.codegen.pytorch_backend import PyTorchBackend
from tensorloom.analyzer.type_checker import TypeChecker
from tensorloom.analyzer.shape_inference import ShapeInferenceEngine


# ── Banner ────────────────────────────────────────────────────
BANNER = r"""
  +================================================+
  |   TensorLoom Compiler  v0.1.0                  |
  |   GPU-Efficient Language for AI Training        |
  +================================================+
"""


def main() -> None:
    """Entry point for the `tlc` command."""
    parser = argparse.ArgumentParser(
        prog="tlc",
        description="TensorLoom Compiler — GPU-Efficient Language for AI Training",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # tlc run
    run_parser = subparsers.add_parser("run", help="Compile and execute a TensorLoom file")
    run_parser.add_argument("file", help="Path to .tl or .nml file")

    # tlc compile
    compile_parser = subparsers.add_parser("compile", help="Transpile to Python")
    compile_parser.add_argument("file", help="Path to .tl or .nml file")
    compile_parser.add_argument("-o", "--output", help="Output .py file path")

    # tlc check
    check_parser = subparsers.add_parser("check", help="Type-check and validate")
    check_parser.add_argument("file", help="Path to .tl or .nml file")

    # tlc info
    info_parser = subparsers.add_parser("info", help="Show estimated GPU memory usage")
    info_parser.add_argument("file", help="Path to .tl or .nml file")

    # tlc tokens  (debug)
    tokens_parser = subparsers.add_parser("tokens", help="[Debug] Dump token stream")
    tokens_parser.add_argument("file", help="Path to .tl or .nml file")

    # tlc ast  (debug)
    ast_parser = subparsers.add_parser("ast", help="[Debug] Dump AST")
    ast_parser.add_argument("file", help="Path to .tl or .nml file")

    args = parser.parse_args()

    if not args.command:
        print(BANNER)
        parser.print_help()
        sys.exit(0)

    # Validate file
    filepath = args.file
    if not os.path.isfile(filepath):
        _error(f"File not found: {filepath}")
    if not filepath.endswith((".tl", ".nml")):
        _error("Unsupported file type. Use .tl or .nml")

    # Read source
    with open(filepath, "r", encoding="utf-8") as f:
        source = f.read()

    # Dispatch
    if args.command == "tokens":
        _cmd_tokens(source, filepath)
    elif args.command == "ast":
        _cmd_ast(source, filepath)
    elif args.command == "check":
        _cmd_check(source, filepath)
    elif args.command == "compile":
        _cmd_compile(source, filepath, args.output)
    elif args.command == "run":
        _cmd_run(source, filepath)
    elif args.command == "info":
        _cmd_info(source, filepath)


# ══════════════════════════════════════════════════════════════
#  Commands
# ══════════════════════════════════════════════════════════════

def _cmd_tokens(source: str, filepath: str) -> None:
    """Dump the token stream for debugging."""
    print(f"[tokens] Tokenizing: {filepath}")
    tokens = _lex(source, filepath)
    print(f"   {len(tokens)} tokens generated\n")
    for tok in tokens:
        print(f"   {tok}")


def _cmd_ast(source: str, filepath: str) -> None:
    """Dump the AST for debugging."""
    print(f"[ast] Parsing AST: {filepath}")
    tokens = _lex(source, filepath)
    ast = _parse(tokens)
    _print_ast(ast, indent=0)


def _cmd_check(source: str, filepath: str) -> None:
    """Type-check, shape-check, and validate without executing."""
    print(f"[check] Checking: {filepath}")
    t0 = time.perf_counter()
    tokens = _lex(source, filepath)
    ast = _parse(tokens)
    parse_time = (time.perf_counter() - t0) * 1000
    stmt_count = len(ast.statements)
    print(f"   {len(tokens)} tokens, {stmt_count} statements")
    print(f"   Parse time: {parse_time:.1f}ms")

    # Semantic analysis
    checker = TypeChecker()
    errors, warnings = checker.analyze(ast)

    # Shape inference
    shape_engine = ShapeInferenceEngine()
    shape_report = shape_engine.infer(ast)

    elapsed = (time.perf_counter() - t0) * 1000

    # Report shape flow for models
    for model_name, layer_shapes in shape_report.model_shapes.items():
        print(f"\n   Shape flow for '{model_name}':")
        for spec in layer_shapes:
            print(f"   +-- {spec.layer_name}: {spec.input_shape} -> {spec.output_shape}")

    # Report errors
    total_errors = errors + shape_report.errors
    total_warnings = warnings + [w for w in shape_report.warnings]

    for w in total_warnings:
        print(f"   [WARN] L{w.line}: {w.message}")

    for e in total_errors:
        msg = e.message if hasattr(e, 'message') else str(e)
        line = e.line if hasattr(e, 'line') else '?'
        print(f"   [ERROR] L{line}: {msg}")

    if total_errors:
        print(f"\n   [FAIL] {len(total_errors)} error(s) found in {elapsed:.1f}ms")
        sys.exit(1)
    else:
        print(f"\n   [OK] No errors found ({elapsed:.1f}ms)")


def _cmd_compile(source: str, filepath: str, output: str | None) -> None:
    """Transpile to PyTorch Python code."""
    print(BANNER)
    print(f"[compile] Compiling: {filepath}")
    t0 = time.perf_counter()

    tokens = _lex(source, filepath)
    ast = _parse(tokens)
    source_dir = os.path.dirname(os.path.abspath(filepath)) or "."
    backend = PyTorchBackend(source_dir=source_dir)
    python_code = backend.generate(ast)

    elapsed = (time.perf_counter() - t0) * 1000

    # Determine output path
    if not output:
        output = filepath.rsplit(".", 1)[0] + "_compiled.py"

    with open(output, "w", encoding="utf-8") as f:
        f.write(python_code)

    print(f"   [OK] Emitted: {output}")
    print(f"   Compile time: {elapsed:.1f}ms")
    print(f"   Lines: {python_code.count(chr(10))}")


def _cmd_run(source: str, filepath: str) -> None:
    """Compile and immediately execute."""
    print(BANNER)
    print(f"[run] Compiling & Running: {filepath}")
    t0 = time.perf_counter()

    tokens = _lex(source, filepath)
    ast = _parse(tokens)
    backend = PyTorchBackend()
    python_code = backend.generate(ast)

    compile_time = (time.perf_counter() - t0) * 1000

    # Write to temp file
    output = filepath.rsplit(".", 1)[0] + "_compiled.py"
    with open(output, "w", encoding="utf-8") as f:
        f.write(python_code)

    print(f"   [OK] Compiled in {compile_time:.1f}ms -> {output}")
    print(f"   Executing...\n")
    print("-" * 50)

    # Execute
    result = subprocess.run(
        [sys.executable, output],
        cwd=os.path.dirname(os.path.abspath(filepath)) or ".",
    )

    print("-" * 50)
    if result.returncode == 0:
        print(f"\n   [OK] Execution completed successfully.")
    else:
        print(f"\n   [FAIL] Execution failed with code {result.returncode}")
        sys.exit(result.returncode)


def _cmd_info(source: str, filepath: str) -> None:
    """Estimate GPU memory usage and parameter count for the program."""
    print(f"[info] Analyzing: {filepath}")
    tokens = _lex(source, filepath)
    ast = _parse(tokens)

    from tensorloom.parser.ast_nodes import (
        ModelDefinition, NMLModel, ImportStatement, FunctionCall, Identifier, NumberLiteral
    )

    models_to_analyze: list[tuple[str, list, dict]] = []

    for stmt in ast.statements:
        if isinstance(stmt, ModelDefinition):
            models_to_analyze.append((stmt.name, stmt.layers, {}))
        elif isinstance(stmt, NMLModel):
            # Resolve config dict
            config_vals = {}
            for k, v in stmt.config.items():
                if isinstance(v, NumberLiteral):
                    config_vals[k] = v.value
                elif hasattr(v, "value"):
                    config_vals[k] = v.value
            models_to_analyze.append((stmt.name, stmt.layers, config_vals))
        elif isinstance(stmt, ImportStatement) and stmt.is_nml:
            # Cross-file NML import
            nml_filename = ".".join(stmt.module_path[:-1]) + ".nml"
            source_dir = os.path.dirname(os.path.abspath(filepath)) or "."
            nml_path = os.path.join(source_dir, nml_filename)
            if os.path.exists(nml_path):
                with open(nml_path, "r", encoding="utf-8") as f:
                    nml_src = f.read()
                nml_toks = _lex(nml_src, nml_filename)
                nml_ast = _parse(nml_toks)
                for s in nml_ast.statements:
                    if isinstance(s, NMLModel):
                        cfg = {k: getattr(v, "value", v) for k, v in s.config.items()}
                        name = stmt.alias or s.name
                        models_to_analyze.append((name, s.layers, cfg))

    def _resolve_val(node, config: dict):
        if isinstance(node, NumberLiteral):
            return node.value
        elif isinstance(node, Identifier) and node.name in config:
            return config[node.name]
        elif hasattr(node, "value"):
            return node.value
        return None

    for model_name, layers, config in models_to_analyze:
        total_params = 0
        layers_info: list[tuple[str, int, str]] = []

        print(f"\n   Model: {model_name}")
        for layer in layers:
            call = layer.layer_type if isinstance(layer.layer_type, FunctionCall) else None
            if not call or not isinstance(call.callee, Identifier):
                continue

            ltype = call.callee.name
            params = 0

            # Linear(in_features, out_features)
            if ltype == "Linear":
                args = [_resolve_val(a, config) for a in call.args if _resolve_val(a, config) is not None]
                if len(args) >= 2:
                    in_f, out_f = int(args[0]), int(args[1])
                    params = in_f * out_f + out_f  # weights + bias

            # Conv2d(in_channels, out_channels, kernel_size, ...)
            elif ltype == "Conv2d":
                args = [_resolve_val(a, config) for a in call.args if _resolve_val(a, config) is not None]
                ksize = 3
                if "kernel_size" in call.kwargs:
                    ksize = int(_resolve_val(call.kwargs["kernel_size"], config) or 3)
                elif len(args) >= 3:
                    ksize = int(args[2])
                if len(args) >= 2:
                    in_c, out_c = int(args[0]), int(args[1])
                    params = in_c * out_c * ksize * ksize + out_c

            # MultiheadAttention / MultiHeadAttention(embed_dim, num_heads)
            elif ltype in ("MultiheadAttention", "MultiHeadAttention"):
                args = [_resolve_val(a, config) for a in call.args if _resolve_val(a, config) is not None]
                if len(args) >= 1:
                    dim = int(args[0])
                    params = 4 * (dim * dim + dim)  # Q, K, V, Out projections

            # Embedding(num_embeddings, embedding_dim)
            elif ltype == "Embedding":
                args = [_resolve_val(a, config) for a in call.args if _resolve_val(a, config) is not None]
                if len(args) >= 2:
                    num_e, e_dim = int(args[0]), int(args[1])
                    params = num_e * e_dim

            # LayerNorm(normalized_shape) / BatchNorm2d(num_features)
            elif ltype in ("LayerNorm", "BatchNorm2d", "BatchNorm1d"):
                args = [_resolve_val(a, config) for a in call.args if _resolve_val(a, config) is not None]
                if len(args) >= 1:
                    dim = int(args[0])
                    params = 2 * dim  # gamma + beta

            total_params += params
            layers_info.append((layer.name, params, ltype))

        for name, params, ltype in layers_info:
            mem_mb = (params * 4) / (1024 * 1024)  # float32
            print(f"   +-- {name} ({ltype}): {params:,} params ({mem_mb:.3f} MB)")

        total_mem = (total_params * 4) / (1024 * 1024)
        train_mem = total_mem * 3  # params + gradients + optimizer states
        print(f"   |")
        print(f"   +-- Total Parameters: {total_params:,}")
        print(f"   +-- Model Memory (FP32): {total_mem:.2f} MB")
        print(f"   +-- Training Memory Est.: {train_mem:.2f} MB")
        print(f"   +-- Training Memory (FP16/BF16): {train_mem / 2:.2f} MB")

    # Shape flow analysis
    shape_engine = ShapeInferenceEngine()
    shape_report = shape_engine.infer(ast)

    for model_name, layer_shapes in shape_report.model_shapes.items():
        print(f"\n   Shape flow for '{model_name}':")
        for spec in layer_shapes:
            print(f"   +-- {spec.layer_name}: {spec.input_shape} -> {spec.output_shape}")

    for e in shape_report.errors:
        print(f"   [SHAPE ERROR] L{e.line}: {e.message}")


# ══════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════

def _lex(source: str, filepath: str) -> list:
    """Run the lexer, handling errors."""
    try:
        lexer = Lexer(source, filename=filepath)
        return lexer.tokenize()
    except LexerError as e:
        _error(str(e))
        return []  # unreachable


def _parse(tokens: list):
    """Run the parser, handling errors."""
    try:
        parser = Parser(tokens)
        return parser.parse()
    except ParseError as e:
        _error(str(e))


def _error(msg: str) -> None:
    """Print an error and exit."""
    print(f"\n   ✗ {msg}", file=sys.stderr)
    sys.exit(1)


def _print_ast(node, indent: int = 0) -> None:
    """Pretty-print an AST node recursively."""
    prefix = "   " + "  " * indent
    name = type(node).__name__

    # Collect non-default, non-location fields
    from dataclasses import fields as dc_fields
    try:
        field_list = dc_fields(node)
    except TypeError:
        print(f"{prefix}{node!r}")
        return

    simple_fields = {}
    complex_fields = {}

    for f in field_list:
        if f.name in ("line", "column"):
            continue
        val = getattr(node, f.name)
        if isinstance(val, list):
            complex_fields[f.name] = val
        elif hasattr(val, "__dataclass_fields__"):
            complex_fields[f.name] = val
        elif isinstance(val, dict):
            complex_fields[f.name] = val
        else:
            simple_fields[f.name] = val

    # Print node header
    simple_str = ", ".join(f"{k}={v!r}" for k, v in simple_fields.items())
    print(f"{prefix}{name}({simple_str})")

    # Print complex children
    for key, val in complex_fields.items():
        if isinstance(val, list):
            if val:
                print(f"{prefix}  {key}:")
                for item in val:
                    _print_ast(item, indent + 2)
        elif isinstance(val, dict):
            if val:
                print(f"{prefix}  {key}:")
                for k, v in val.items():
                    print(f"{prefix}    {k}:")
                    _print_ast(v, indent + 3)
        else:
            print(f"{prefix}  {key}:")
            _print_ast(val, indent + 2)


if __name__ == "__main__":
    main()
