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
    """Estimate GPU memory usage for the program."""
    print(f"[info] Analyzing: {filepath}")
    tokens = _lex(source, filepath)
    ast = _parse(tokens)

    # Simple heuristic memory estimation
    total_params = 0
    layers_info: list[tuple[str, int]] = []

    for stmt in ast.statements:
        from tensorloom.parser.ast_nodes import ModelDefinition, FunctionCall, Identifier, NumberLiteral
        if isinstance(stmt, ModelDefinition):
            print(f"\n   Model: {stmt.name}")
            for layer in stmt.layers:
                if isinstance(layer.layer_type, FunctionCall):
                    call = layer.layer_type
                    if isinstance(call.callee, Identifier) and call.callee.name == "Linear":
                        dims = [a for a in call.args if isinstance(a, NumberLiteral)]
                        if len(dims) >= 2:
                            params = int(dims[0].value) * int(dims[1].value) + int(dims[1].value)
                            total_params += params
                            layers_info.append((layer.name, params))

            for name, params in layers_info:
                mem_mb = (params * 4) / (1024 * 1024)  # float32
                print(f"   +-- {name}: {params:,} params ({mem_mb:.2f} MB)")

            total_mem = (total_params * 4) / (1024 * 1024)
            train_mem = total_mem * 3  # params + gradients + optimizer states
            print(f"   |")
            print(f"   +-- Total parameters: {total_params:,}")
            print(f"   +-- Model memory (fp32): {total_mem:.2f} MB")
            print(f"   +-- Training memory est.: {train_mem:.2f} MB")
            print(f"   +-- Training memory (fp16): {train_mem / 2:.2f} MB")

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
