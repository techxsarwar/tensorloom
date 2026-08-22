"""
TensorLoom Parser — Recursive-descent parser.

Consumes a token stream from the Lexer and produces an Abstract Syntax Tree.
Supports:
  - import, let, assignment, return, if/elif/else, for
  - model definitions with layer declarations and methods
  - train blocks with config, callbacks, and checkpoint directives
  - Expressions: arithmetic, matmul (@), pipe (|>), function calls,
    member access, tensor literals, f-strings, comparisons, booleans
"""
from __future__ import annotations

from tensorloom.lexer.tokens import Token, TokenType
from tensorloom.parser.ast_nodes import (
    ASTNode,
    AssignStatement,
    BinaryOp,
    BooleanLiteral,
    CheckpointConfig,
    ExpressionStatement,
    ForLoop,
    FStringLiteral,
    FunctionCall,
    FunctionDef,
    Identifier,
    IfStatement,
    ImportStatement,
    IndexAccess,
    KernelDef,
    KernelParam,
    LayerDeclaration,
    LetStatement,
    ListLiteral,
    MemberAccess,
    ModelDefinition,
    NMLModel,
    NoneLiteral,
    NumberLiteral,
    Parameter,
    PipeExpression,
    PipeStage,
    Program,
    ReturnStatement,
    StringLiteral,
    TensorLiteral,
    TrainBlock,
    TrainCallback,
    UnaryOp,
)


class ParseError(Exception):
    """Raised when the parser encounters an unexpected token."""

    def __init__(self, message: str, token: Token) -> None:
        self.token = token
        super().__init__(f"ParseError at L{token.line}:{token.column}: {message}")


class Parser:
    """Recursive-descent parser for TensorLoom (.tl) and NML (.nml) files."""

    def __init__(self, tokens: list[Token]) -> None:
        self.tokens = tokens
        self.pos = 0

    # ══════════════════════════════════════════════════════════
    #  Public API
    # ══════════════════════════════════════════════════════════

    def parse(self) -> Program:
        """Parse the entire token stream into a Program AST node."""
        program = Program(line=1, column=1, statements=[])

        while not self._is_at_end():
            self._skip_newlines()
            if self._is_at_end():
                break
            stmt = self._parse_statement()
            if stmt is not None:
                program.statements.append(stmt)

        return program

    # ══════════════════════════════════════════════════════════
    #  Statements
    # ══════════════════════════════════════════════════════════

    def _parse_statement(self) -> ASTNode | None:
        """Dispatch to the correct statement parser based on the current token."""
        tok = self._current()

        if tok.type == TokenType.IMPORT:
            return self._parse_import()

        if tok.type == TokenType.LET:
            return self._parse_let()

        if tok.type == TokenType.MODEL:
            return self._parse_model()

        if tok.type == TokenType.TRAIN:
            return self._parse_train()

        if tok.type == TokenType.FN:
            return self._parse_function_def()

        if tok.type == TokenType.RETURN:
            return self._parse_return()

        if tok.type == TokenType.IF:
            return self._parse_if()

        if tok.type == TokenType.FOR:
            return self._parse_for()

        # @ decorator blocks
        if tok.type == TokenType.AT:
            if self.pos + 1 < len(self.tokens):
                next_tok = self.tokens[self.pos + 1]
                if next_tok.type == TokenType.IDENTIFIER and next_tok.value == "kernel":
                    return self._parse_kernel_def()
                # @model for NML declarative blocks
                if next_tok.type == TokenType.MODEL:
                    return self._parse_nml_model()

        # Assignment  (identifier = expr)  or  expression statement
        return self._parse_assignment_or_expr()

    # ── import ────────────────────────────────────────────────

    def _parse_import(self) -> ImportStatement:
        """Parse: import std.io  or  import transformer.nml as MyBlock"""
        tok = self._expect(TokenType.IMPORT)
        parts: list[str] = []
        parts.append(self._expect(TokenType.IDENTIFIER).value)
        while self._match(TokenType.DOT):
            parts.append(self._expect(TokenType.IDENTIFIER).value)

        # Check if this is an NML import (path ends with .nml)
        is_nml = len(parts) >= 2 and parts[-1] == "nml"

        # Check for 'as Alias'
        alias = None
        if (not self._is_at_end()
                and self._current().type == TokenType.IDENTIFIER
                and self._current().value == "as"):
            self._advance()  # consume 'as'
            alias = self._expect(TokenType.IDENTIFIER).value

        self._consume_newline()
        return ImportStatement(
            line=tok.line, column=tok.column,
            module_path=parts, alias=alias, is_nml=is_nml,
        )

    # ── @kernel ──────────────────────────────────────────────

    def _parse_kernel_def(self) -> KernelDef:
        """Parse:  @kernel def name(x_ptr, BLOCK_SIZE: tl.constexpr): body"""
        tok = self._expect(TokenType.AT)       # @
        self._expect(TokenType.IDENTIFIER)      # "kernel"
        self._expect_keyword_or_name("def")     # "def" (treat as identifier)
        name = self._expect(TokenType.IDENTIFIER).value
        self._expect(TokenType.LPAREN)

        params: list[KernelParam] = []
        while not self._check(TokenType.RPAREN):
            if params:
                self._expect(TokenType.COMMA)
            param_name = self._expect(TokenType.IDENTIFIER).value
            is_constexpr = False

            # Check for : tl.constexpr type hint
            if self._check(TokenType.COLON):
                self._advance()  # consume ':'
                # Expect tl.constexpr  or just constexpr
                hint_name = self._expect(TokenType.IDENTIFIER).value
                if self._check(TokenType.DOT):
                    self._advance()  # consume '.'
                    sub = self._expect(TokenType.IDENTIFIER).value
                    if sub == "constexpr":
                        is_constexpr = True
                elif hint_name == "constexpr":
                    is_constexpr = True

            params.append(KernelParam(
                line=tok.line, column=tok.column,
                name=param_name, is_constexpr=is_constexpr,
            ))

        self._expect(TokenType.RPAREN)
        self._expect(TokenType.COLON)
        self._consume_newline()
        self._expect(TokenType.INDENT)

        body = self._parse_block()

        self._expect(TokenType.DEDENT)

        return KernelDef(
            line=tok.line, column=tok.column,
            name=name, params=params, body=body,
        )

    def _expect_keyword_or_name(self, expected: str) -> Token:
        """Expect a token that matches a specific value (keyword or identifier)."""
        tok = self._current()
        if tok.value == expected:
            self._advance()
            return tok
        raise ParseError(f"Expected '{expected}', got '{tok.value}'", tok)

    # ── @model (NML) ─────────────────────────────────────────

    def _parse_nml_model(self) -> NMLModel:
        """Parse NML declarative model:
        
        @model Name:
            @config:
                key = value
            @layers:
                name = Type(args)
            @forward(params):
                body
        """
        tok = self._expect(TokenType.AT)       # @
        self._expect(TokenType.MODEL)           # model
        name = self._expect(TokenType.IDENTIFIER).value
        self._expect(TokenType.COLON)
        self._consume_newline()
        self._expect(TokenType.INDENT)

        config: dict[str, ASTNode] = {}
        layers: list[LayerDeclaration] = []
        forward_params: list[str] = []
        forward_body: list[ASTNode] = []

        while not self._check(TokenType.DEDENT) and not self._is_at_end():
            self._skip_newlines()
            if self._check(TokenType.DEDENT) or self._is_at_end():
                break

            # Each sub-section starts with @
            if self._check(TokenType.AT):
                self._advance()  # consume @
                section_name = self._expect(TokenType.IDENTIFIER).value

                if section_name == "config":
                    self._expect(TokenType.COLON)
                    self._consume_newline()
                    self._expect(TokenType.INDENT)
                    config = self._parse_nml_config_block()
                    self._expect(TokenType.DEDENT)

                elif section_name == "layers":
                    self._expect(TokenType.COLON)
                    self._consume_newline()
                    self._expect(TokenType.INDENT)
                    layers = self._parse_nml_layers_block()
                    self._expect(TokenType.DEDENT)

                elif section_name == "forward":
                    # @forward(x):  or  @forward(x, mask):
                    self._expect(TokenType.LPAREN)
                    while not self._check(TokenType.RPAREN):
                        if forward_params:
                            self._expect(TokenType.COMMA)
                        forward_params.append(
                            self._expect(TokenType.IDENTIFIER).value
                        )
                    self._expect(TokenType.RPAREN)
                    self._expect(TokenType.COLON)
                    self._consume_newline()
                    self._expect(TokenType.INDENT)
                    forward_body = self._parse_block()
                    self._expect(TokenType.DEDENT)
                else:
                    raise ParseError(
                        f"Unknown NML section '@{section_name}'. "
                        "Expected @config, @layers, or @forward",
                        self._current(),
                    )
            else:
                raise ParseError(
                    "Expected @config, @layers, or @forward section",
                    self._current(),
                )

        self._expect(TokenType.DEDENT)

        return NMLModel(
            line=tok.line, column=tok.column,
            name=name, config=config, layers=layers,
            forward_params=forward_params, forward_body=forward_body,
        )

    def _parse_nml_config_block(self) -> dict[str, ASTNode]:
        """Parse @config key=value pairs."""
        config: dict[str, ASTNode] = {}
        while not self._check(TokenType.DEDENT) and not self._is_at_end():
            self._skip_newlines()
            if self._check(TokenType.DEDENT):
                break
            key = self._expect(TokenType.IDENTIFIER).value
            self._expect(TokenType.EQUALS)
            value = self._parse_expression()
            config[key] = value
            self._consume_newline()
        return config

    def _parse_nml_layers_block(self) -> list[LayerDeclaration]:
        """Parse @layers declarations:  name = Type(args)"""
        layers: list[LayerDeclaration] = []
        while not self._check(TokenType.DEDENT) and not self._is_at_end():
            self._skip_newlines()
            if self._check(TokenType.DEDENT):
                break
            tok = self._current()
            layer_name = self._expect(TokenType.IDENTIFIER).value
            self._expect(TokenType.EQUALS)
            layer_type = self._parse_expression()
            layers.append(LayerDeclaration(
                line=tok.line, column=tok.column,
                name=layer_name, layer_type=layer_type,
            ))
            self._consume_newline()
        return layers

    # ── let ───────────────────────────────────────────────────

    def _parse_let(self) -> LetStatement:
        tok = self._expect(TokenType.LET)
        name = self._expect(TokenType.IDENTIFIER).value
        self._expect(TokenType.EQUALS)
        value = self._parse_expression()
        self._consume_newline()
        return LetStatement(line=tok.line, column=tok.column, name=name, value=value)

    # ── model ─────────────────────────────────────────────────

    def _parse_model(self) -> ModelDefinition:
        tok = self._expect(TokenType.MODEL)
        name = self._expect(TokenType.IDENTIFIER).value
        self._expect(TokenType.COLON)
        self._consume_newline()
        self._expect(TokenType.INDENT)

        layers: list[LayerDeclaration] = []
        methods: list[FunctionDef] = []

        while not self._check(TokenType.DEDENT) and not self._is_at_end():
            self._skip_newlines()
            if self._check(TokenType.DEDENT):
                break
            if self._check(TokenType.LAYER):
                layers.append(self._parse_layer_decl())
            elif self._check(TokenType.FN):
                methods.append(self._parse_function_def())
            else:
                raise ParseError(
                    f"Expected 'layer' or 'fn' in model body, got {self._current().type.name}",
                    self._current(),
                )

        self._expect(TokenType.DEDENT)
        return ModelDefinition(
            line=tok.line, column=tok.column, name=name, layers=layers, methods=methods
        )

    def _parse_layer_decl(self) -> LayerDeclaration:
        tok = self._expect(TokenType.LAYER)
        name = self._expect(TokenType.IDENTIFIER).value
        self._expect(TokenType.EQUALS)
        layer_type = self._parse_call_expression()
        self._consume_newline()
        return LayerDeclaration(line=tok.line, column=tok.column, name=name, layer_type=layer_type)

    # ── train ─────────────────────────────────────────────────

    def _parse_train(self) -> TrainBlock:
        tok = self._expect(TokenType.TRAIN)
        model_name = self._expect_name().value
        self._expect(TokenType.ON)
        data_name = self._expect_name().value
        self._expect(TokenType.COLON)
        self._consume_newline()
        self._expect(TokenType.INDENT)

        params: dict[str, ASTNode] = {}
        callbacks: list[TrainCallback] = []
        checkpoint_config: CheckpointConfig | None = None

        while not self._check(TokenType.DEDENT) and not self._is_at_end():
            self._skip_newlines()
            if self._check(TokenType.DEDENT):
                break

            # on epoch_end(metrics): ...
            if self._check(TokenType.ON):
                callbacks.append(self._parse_train_callback())
                continue

            # checkpoint every N epochs
            if self._check(TokenType.IDENTIFIER) and self._current().value == "checkpoint":
                checkpoint_config = self._parse_checkpoint_config()
                continue

            # key = value
            key_tok = self._expect(TokenType.IDENTIFIER)
            key = key_tok.value

            # Handle "X every N unit" pattern  (e.g. "checkpoint every 2 epochs")
            if self._check(TokenType.EVERY):
                self._advance()  # consume 'every'
                freq = self._parse_expression()
                unit = self._expect(TokenType.IDENTIFIER).value
                checkpoint_config = CheckpointConfig(
                    line=key_tok.line, column=key_tok.column,
                    frequency=freq, unit=unit,
                )
                self._consume_newline()
                continue

            self._expect(TokenType.EQUALS)
            value = self._parse_expression()
            params[key] = value
            self._consume_newline()

        self._expect(TokenType.DEDENT)
        return TrainBlock(
            line=tok.line, column=tok.column,
            model_name=model_name, data_name=data_name,
            params=params, callbacks=callbacks,
            checkpoint_config=checkpoint_config,
        )

    def _parse_train_callback(self) -> TrainCallback:
        tok = self._expect(TokenType.ON)
        event = self._expect(TokenType.IDENTIFIER).value
        self._expect(TokenType.LPAREN)
        param_name = self._expect(TokenType.IDENTIFIER).value
        self._expect(TokenType.RPAREN)
        self._expect(TokenType.COLON)
        self._consume_newline()
        self._expect(TokenType.INDENT)
        body = self._parse_block()
        self._expect(TokenType.DEDENT)
        return TrainCallback(
            line=tok.line, column=tok.column,
            event=event, param_name=param_name, body=body,
        )

    def _parse_checkpoint_config(self) -> CheckpointConfig:
        tok = self._advance()  # consume 'checkpoint'
        self._expect(TokenType.EVERY)
        freq = self._parse_expression()
        unit = "epochs"
        if self._check(TokenType.IDENTIFIER):
            unit = self._advance().value
        self._consume_newline()
        return CheckpointConfig(line=tok.line, column=tok.column, frequency=freq, unit=unit)

    # ── fn (function definition) ──────────────────────────────

    def _parse_function_def(self) -> FunctionDef:
        tok = self._expect(TokenType.FN)
        name = self._expect(TokenType.IDENTIFIER).value
        self._expect(TokenType.LPAREN)
        params = self._parse_param_list()
        self._expect(TokenType.RPAREN)

        return_type: str | None = None
        if self._match(TokenType.ARROW):
            return_type = self._expect_one_of(TokenType.IDENTIFIER, TokenType.TENSOR_TYPE).value

        self._expect(TokenType.COLON)
        self._consume_newline()
        self._expect(TokenType.INDENT)
        body = self._parse_block()
        self._expect(TokenType.DEDENT)

        return FunctionDef(
            line=tok.line, column=tok.column,
            name=name, params=params, return_type=return_type, body=body,
        )

    def _parse_param_list(self) -> list[Parameter]:
        """Parse a comma-separated parameter list."""
        params: list[Parameter] = []
        if self._check(TokenType.RPAREN):
            return params

        params.append(self._parse_param())
        while self._match(TokenType.COMMA):
            if self._check(TokenType.RPAREN):
                break
            params.append(self._parse_param())
        return params

    def _parse_param(self) -> Parameter:
        tok = self._expect_one_of(TokenType.IDENTIFIER, TokenType.SELF)
        name = tok.value
        type_hint: str | None = None
        if self._match(TokenType.COLON):
            type_hint = self._expect_one_of(TokenType.IDENTIFIER, TokenType.TENSOR_TYPE).value
        return Parameter(line=tok.line, column=tok.column, name=name, type_hint=type_hint)

    # ── return ────────────────────────────────────────────────

    def _parse_return(self) -> ReturnStatement:
        tok = self._expect(TokenType.RETURN)
        value: ASTNode | None = None
        if not self._check(TokenType.NEWLINE) and not self._check(TokenType.DEDENT) and not self._is_at_end():
            value = self._parse_expression()
        self._consume_newline()
        return ReturnStatement(line=tok.line, column=tok.column, value=value)

    # ── if / elif / else ──────────────────────────────────────

    def _parse_if(self) -> IfStatement:
        tok = self._expect(TokenType.IF)
        condition = self._parse_expression()
        self._expect(TokenType.COLON)
        self._consume_newline()
        self._expect(TokenType.INDENT)
        body = self._parse_block()
        self._expect(TokenType.DEDENT)

        elif_clauses: list[tuple[ASTNode, list[ASTNode]]] = []
        else_body: list[ASTNode] = []

        while self._check(TokenType.ELIF):
            self._advance()
            elif_cond = self._parse_expression()
            self._expect(TokenType.COLON)
            self._consume_newline()
            self._expect(TokenType.INDENT)
            elif_body = self._parse_block()
            self._expect(TokenType.DEDENT)
            elif_clauses.append((elif_cond, elif_body))

        if self._match(TokenType.ELSE):
            self._expect(TokenType.COLON)
            self._consume_newline()
            self._expect(TokenType.INDENT)
            else_body = self._parse_block()
            self._expect(TokenType.DEDENT)

        return IfStatement(
            line=tok.line, column=tok.column,
            condition=condition, body=body,
            elif_clauses=elif_clauses, else_body=else_body,
        )

    # ── for ───────────────────────────────────────────────────

    def _parse_for(self) -> ForLoop:
        tok = self._expect(TokenType.FOR)
        var = self._expect(TokenType.IDENTIFIER).value
        self._expect(TokenType.IN)
        iterable = self._parse_expression()
        self._expect(TokenType.COLON)
        self._consume_newline()
        self._expect(TokenType.INDENT)
        body = self._parse_block()
        self._expect(TokenType.DEDENT)
        return ForLoop(
            line=tok.line, column=tok.column,
            variable=var, iterable=iterable, body=body,
        )

    # ── assignment or expression statement ────────────────────

    def _parse_assignment_or_expr(self) -> ASTNode:
        """Parse either `target = value` or a bare expression statement."""
        expr = self._parse_expression()

        # Check for assignment:  x = ...,  self.x = ...
        if self._check(TokenType.EQUALS) and not self._check_at(1, TokenType.EQUALS):
            self._advance()  # consume '='
            value = self._parse_expression()
            self._consume_newline()
            return AssignStatement(
                line=expr.line, column=expr.column, target=expr, value=value
            )

        self._consume_newline()
        return ExpressionStatement(line=expr.line, column=expr.column, expression=expr)

    # ── block (list of statements until DEDENT) ───────────────

    def _parse_block(self) -> list[ASTNode]:
        """Parse statements until we hit DEDENT or EOF."""
        stmts: list[ASTNode] = []
        while not self._check(TokenType.DEDENT) and not self._is_at_end():
            self._skip_newlines()
            if self._check(TokenType.DEDENT) or self._is_at_end():
                break
            stmt = self._parse_statement()
            if stmt is not None:
                stmts.append(stmt)
        return stmts

    # ══════════════════════════════════════════════════════════
    #  Expressions  (precedence climbing)
    # ══════════════════════════════════════════════════════════

    def _parse_expression(self) -> ASTNode:
        """Entry point for expression parsing — lowest precedence."""
        return self._parse_pipe()

    # ── pipe (|>) ─────────────────────────────────────────────

    def _parse_pipe(self) -> ASTNode:
        """Parse pipe expressions:  expr |> func |> func(args)"""
        left = self._parse_or()

        if not self._check(TokenType.PIPE_ARROW):
            return left

        stages: list[PipeStage] = []
        while self._match(TokenType.PIPE_ARROW):
            func_tok = self._expect(TokenType.IDENTIFIER)
            args: list[ASTNode] = []
            kwargs: dict[str, ASTNode] = {}

            if self._match(TokenType.LPAREN):
                args, kwargs = self._parse_arg_list()
                self._expect(TokenType.RPAREN)

            stages.append(PipeStage(
                line=func_tok.line, column=func_tok.column,
                func_name=func_tok.value, args=args, kwargs=kwargs,
            ))

        return PipeExpression(
            line=left.line, column=left.column, value=left, stages=stages
        )

    # ── boolean or / and ──────────────────────────────────────

    def _parse_or(self) -> ASTNode:
        left = self._parse_and()
        while self._check(TokenType.OR):
            op_tok = self._advance()
            right = self._parse_and()
            left = BinaryOp(line=op_tok.line, column=op_tok.column, op="or", left=left, right=right)
        return left

    def _parse_and(self) -> ASTNode:
        left = self._parse_not()
        while self._check(TokenType.AND):
            op_tok = self._advance()
            right = self._parse_not()
            left = BinaryOp(line=op_tok.line, column=op_tok.column, op="and", left=left, right=right)
        return left

    def _parse_not(self) -> ASTNode:
        if self._check(TokenType.NOT):
            op_tok = self._advance()
            operand = self._parse_not()
            return UnaryOp(line=op_tok.line, column=op_tok.column, op="not", operand=operand)
        return self._parse_comparison()

    # ── comparison ────────────────────────────────────────────

    def _parse_comparison(self) -> ASTNode:
        left = self._parse_addition()
        comp_types = {
            TokenType.DOUBLE_EQUALS: "==",
            TokenType.NOT_EQUALS: "!=",
            TokenType.LESS: "<",
            TokenType.GREATER: ">",
            TokenType.LESS_EQUAL: "<=",
            TokenType.GREATER_EQUAL: ">=",
        }
        while self._current().type in comp_types:
            op_tok = self._advance()
            right = self._parse_addition()
            left = BinaryOp(
                line=op_tok.line, column=op_tok.column,
                op=comp_types[op_tok.type], left=left, right=right,
            )
        return left

    # ── addition / subtraction ────────────────────────────────

    def _parse_addition(self) -> ASTNode:
        left = self._parse_multiplication()
        while self._current().type in (TokenType.PLUS, TokenType.MINUS):
            op_tok = self._advance()
            right = self._parse_multiplication()
            left = BinaryOp(
                line=op_tok.line, column=op_tok.column,
                op=op_tok.value, left=left, right=right,
            )
        return left

    # ── multiplication / division / matmul ────────────────────

    def _parse_multiplication(self) -> ASTNode:
        left = self._parse_power()
        while self._current().type in (TokenType.STAR, TokenType.SLASH, TokenType.PERCENT, TokenType.AT):
            op_tok = self._advance()
            right = self._parse_power()
            left = BinaryOp(
                line=op_tok.line, column=op_tok.column,
                op=op_tok.value, left=left, right=right,
            )
        return left

    # ── power (**) ────────────────────────────────────────────

    def _parse_power(self) -> ASTNode:
        base = self._parse_unary()
        if self._match(TokenType.DOUBLE_STAR):
            exp = self._parse_unary()
            return BinaryOp(
                line=base.line, column=base.column, op="**", left=base, right=exp
            )
        return base

    # ── unary (-x) ────────────────────────────────────────────

    def _parse_unary(self) -> ASTNode:
        if self._check(TokenType.MINUS):
            op_tok = self._advance()
            operand = self._parse_unary()
            return UnaryOp(line=op_tok.line, column=op_tok.column, op="-", operand=operand)
        return self._parse_postfix()

    # ── postfix (call, member access, indexing) ───────────────

    def _parse_postfix(self) -> ASTNode:
        node = self._parse_primary()

        while True:
            if self._check(TokenType.LPAREN):
                # Function call
                self._advance()
                args, kwargs = self._parse_arg_list()
                self._expect(TokenType.RPAREN)
                node = FunctionCall(
                    line=node.line, column=node.column,
                    callee=node, args=args, kwargs=kwargs,
                )
            elif self._check(TokenType.DOT):
                self._advance()
                member_tok = self._expect(TokenType.IDENTIFIER)
                node = MemberAccess(
                    line=node.line, column=node.column,
                    object=node, member=member_tok.value,
                )
            elif self._check(TokenType.LBRACKET):
                self._advance()
                index = self._parse_expression()
                self._expect(TokenType.RBRACKET)
                node = IndexAccess(
                    line=node.line, column=node.column,
                    object=node, index=index,
                )
            else:
                break

        return node

    # ── primary (atoms) ───────────────────────────────────────

    def _parse_primary(self) -> ASTNode:
        tok = self._current()

        # Numbers
        if tok.type == TokenType.INTEGER:
            self._advance()
            return NumberLiteral(line=tok.line, column=tok.column, value=int(tok.value), is_integer=True)

        if tok.type == TokenType.FLOAT:
            self._advance()
            return NumberLiteral(line=tok.line, column=tok.column, value=float(tok.value), is_integer=False)

        # Strings (including f-strings stored as STRING by lexer)
        if tok.type == TokenType.STRING:
            self._advance()
            if tok.value.startswith("f\""):
                return FStringLiteral(
                    line=tok.line, column=tok.column,
                    parts=[StringLiteral(line=tok.line, column=tok.column, value=tok.value)],
                )
            return StringLiteral(line=tok.line, column=tok.column, value=tok.value)

        # Booleans
        if tok.type == TokenType.TRUE:
            self._advance()
            return BooleanLiteral(line=tok.line, column=tok.column, value=True)
        if tok.type == TokenType.FALSE:
            self._advance()
            return BooleanLiteral(line=tok.line, column=tok.column, value=False)

        # None
        if tok.type == TokenType.NONE:
            self._advance()
            return NoneLiteral(line=tok.line, column=tok.column)

        # self
        if tok.type == TokenType.SELF:
            self._advance()
            return Identifier(line=tok.line, column=tok.column, name="self")

        # tensor(...) literal
        if tok.type == TokenType.IDENTIFIER and tok.value == "tensor" and self._check_at(1, TokenType.LPAREN):
            return self._parse_tensor_literal()

        # Identifier (variable or function name)
        if tok.type == TokenType.IDENTIFIER:
            self._advance()
            return Identifier(line=tok.line, column=tok.column, name=tok.value)

        # Parenthesized expression
        if tok.type == TokenType.LPAREN:
            self._advance()
            expr = self._parse_expression()
            self._expect(TokenType.RPAREN)
            return expr

        # List literal
        if tok.type == TokenType.LBRACKET:
            return self._parse_list_literal()

        raise ParseError(f"Unexpected token: {tok.type.name} ({tok.value!r})", tok)

    # ── tensor literal ────────────────────────────────────────

    def _parse_tensor_literal(self) -> TensorLiteral:
        tok = self._expect(TokenType.IDENTIFIER)  # 'tensor'
        self._expect(TokenType.LPAREN)

        elements: list[ASTNode] = []
        kwargs: dict[str, ASTNode] = {}

        # Parse the first argument — expect a list [...]
        if self._check(TokenType.LBRACKET):
            self._advance()
            if not self._check(TokenType.RBRACKET):
                elements.append(self._parse_expression())
                while self._match(TokenType.COMMA):
                    if self._check(TokenType.RBRACKET):
                        break
                    elements.append(self._parse_expression())
            self._expect(TokenType.RBRACKET)

        # Parse remaining kwargs
        while self._match(TokenType.COMMA):
            if self._check(TokenType.RPAREN):
                break
            key = self._expect(TokenType.IDENTIFIER).value
            self._expect(TokenType.EQUALS)
            val = self._parse_expression()
            kwargs[key] = val

        self._expect(TokenType.RPAREN)
        return TensorLiteral(line=tok.line, column=tok.column, elements=elements, kwargs=kwargs)

    # ── list literal ──────────────────────────────────────────

    def _parse_list_literal(self) -> ListLiteral:
        tok = self._expect(TokenType.LBRACKET)
        elements: list[ASTNode] = []
        if not self._check(TokenType.RBRACKET):
            elements.append(self._parse_expression())
            while self._match(TokenType.COMMA):
                if self._check(TokenType.RBRACKET):
                    break
                elements.append(self._parse_expression())
        self._expect(TokenType.RBRACKET)
        return ListLiteral(line=tok.line, column=tok.column, elements=elements)

    # ── call expression (used in layer declarations) ──────────

    def _parse_call_expression(self) -> FunctionCall:
        """Parse a function call like Linear(784, 256)."""
        name_tok = self._expect(TokenType.IDENTIFIER)
        callee = Identifier(line=name_tok.line, column=name_tok.column, name=name_tok.value)
        self._expect(TokenType.LPAREN)
        args, kwargs = self._parse_arg_list()
        self._expect(TokenType.RPAREN)
        return FunctionCall(
            line=name_tok.line, column=name_tok.column,
            callee=callee, args=args, kwargs=kwargs,
        )

    # ── argument list ─────────────────────────────────────────

    def _parse_arg_list(self) -> tuple[list[ASTNode], dict[str, ASTNode]]:
        """Parse a comma-separated list of positional and keyword arguments."""
        args: list[ASTNode] = []
        kwargs: dict[str, ASTNode] = {}

        if self._check(TokenType.RPAREN):
            return args, kwargs

        self._parse_single_arg(args, kwargs)
        while self._match(TokenType.COMMA):
            if self._check(TokenType.RPAREN):
                break
            self._parse_single_arg(args, kwargs)

        return args, kwargs

    def _parse_single_arg(self, args: list[ASTNode], kwargs: dict[str, ASTNode]) -> None:
        """Parse one argument — positional or keyword."""
        # Check for keyword arg:  name = value
        if (self._check(TokenType.IDENTIFIER)
                and self._check_at(1, TokenType.EQUALS)
                and not self._check_at(2, TokenType.EQUALS)):
            key = self._advance().value
            self._advance()  # consume '='
            val = self._parse_expression()
            kwargs[key] = val
        else:
            args.append(self._parse_expression())

    # ══════════════════════════════════════════════════════════
    #  Token Helpers
    # ══════════════════════════════════════════════════════════

    def _current(self) -> Token:
        if self.pos >= len(self.tokens):
            return Token(TokenType.EOF, "", 0, 0)
        return self.tokens[self.pos]

    def _peek_at(self, offset: int) -> Token:
        idx = self.pos + offset
        if idx >= len(self.tokens):
            return Token(TokenType.EOF, "", 0, 0)
        return self.tokens[idx]

    def _advance(self) -> Token:
        tok = self._current()
        self.pos += 1
        return tok

    def _check(self, token_type: TokenType) -> bool:
        return self._current().type == token_type

    def _check_at(self, offset: int, token_type: TokenType) -> bool:
        return self._peek_at(offset).type == token_type

    def _match(self, token_type: TokenType) -> bool:
        if self._check(token_type):
            self._advance()
            return True
        return False

    def _expect(self, token_type: TokenType) -> Token:
        if not self._check(token_type):
            raise ParseError(
                f"Expected {token_type.name}, got {self._current().type.name} ({self._current().value!r})",
                self._current(),
            )
        return self._advance()

    def _expect_name(self) -> Token:
        """Expect an identifier or a keyword used as a name.
        
        In positions like `train <name> on <name>`, users may use names
        that collide with keywords (e.g., `model`, `data`).  This method
        accepts any keyword token as a valid name.
        """
        # Accept IDENTIFIER directly
        if self._check(TokenType.IDENTIFIER):
            return self._advance()
        # Accept any keyword token as a name
        keyword_types = {
            TokenType.MODEL, TokenType.LAYER, TokenType.LET, TokenType.FN,
            TokenType.TRAIN, TokenType.ON, TokenType.IMPORT, TokenType.RETURN,
            TokenType.IF, TokenType.ELSE, TokenType.ELIF, TokenType.FOR,
            TokenType.IN, TokenType.WHILE, TokenType.SELF, TokenType.EVERY,
            TokenType.TRUE, TokenType.FALSE, TokenType.NONE,
            TokenType.TENSOR_TYPE,
        }
        if self._current().type in keyword_types:
            return self._advance()
        raise ParseError(
            f"Expected a name, got {self._current().type.name} ({self._current().value!r})",
            self._current(),
        )

    def _expect_one_of(self, *types: TokenType) -> Token:
        for t in types:
            if self._check(t):
                return self._advance()
        names = ", ".join(t.name for t in types)
        raise ParseError(
            f"Expected one of [{names}], got {self._current().type.name}",
            self._current(),
        )

    def _is_at_end(self) -> bool:
        return self._current().type == TokenType.EOF

    def _skip_newlines(self) -> None:
        while self._check(TokenType.NEWLINE):
            self._advance()

    def _consume_newline(self) -> None:
        """Consume a NEWLINE if present; tolerate DEDENT/EOF boundary."""
        if self._check(TokenType.NEWLINE):
            self._advance()
