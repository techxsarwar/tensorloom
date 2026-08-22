"""
Tests for the TensorLoom Parser.
"""
import pytest
from tensorloom.lexer.lexer import Lexer
from tensorloom.parser.parser import Parser, ParseError
from tensorloom.parser.ast_nodes import (
    AssignStatement,
    BinaryOp,
    BooleanLiteral,
    ExpressionStatement,
    ForLoop,
    FunctionCall,
    FunctionDef,
    Identifier,
    IfStatement,
    ImportStatement,
    LetStatement,
    ListLiteral,
    ModelDefinition,
    NumberLiteral,
    PipeExpression,
    Program,
    ReturnStatement,
    StringLiteral,
    TensorLiteral,
    TrainBlock,
)


def parse(source: str) -> Program:
    """Helper: lex + parse a source string."""
    tokens = Lexer(source).tokenize()
    return Parser(tokens).parse()


class TestImports:
    def test_simple_import(self):
        prog = parse("import nn\n")
        assert len(prog.statements) == 1
        assert isinstance(prog.statements[0], ImportStatement)
        assert prog.statements[0].module_path == ["nn"]

    def test_dotted_import(self):
        prog = parse("import data.mnist\n")
        stmt = prog.statements[0]
        assert isinstance(stmt, ImportStatement)
        assert stmt.module_path == ["data", "mnist"]


class TestLetStatements:
    def test_let_integer(self):
        prog = parse("let x = 42\n")
        stmt = prog.statements[0]
        assert isinstance(stmt, LetStatement)
        assert stmt.name == "x"
        assert isinstance(stmt.value, NumberLiteral)
        assert stmt.value.value == 42

    def test_let_float(self):
        prog = parse("let y = 3.14\n")
        stmt = prog.statements[0]
        assert isinstance(stmt, LetStatement)
        assert isinstance(stmt.value, NumberLiteral)
        assert stmt.value.value == 3.14

    def test_let_string(self):
        prog = parse('let s = "hello"\n')
        stmt = prog.statements[0]
        assert isinstance(stmt, LetStatement)
        assert isinstance(stmt.value, StringLiteral)
        assert stmt.value.value == "hello"

    def test_let_boolean(self):
        prog = parse("let flag = true\n")
        stmt = prog.statements[0]
        assert isinstance(stmt, LetStatement)
        assert isinstance(stmt.value, BooleanLiteral)
        assert stmt.value.value is True

    def test_let_expression(self):
        prog = parse("let z = x + 1\n")
        stmt = prog.statements[0]
        assert isinstance(stmt, LetStatement)
        assert isinstance(stmt.value, BinaryOp)
        assert stmt.value.op == "+"


class TestExpressions:
    def test_binary_arithmetic(self):
        prog = parse("let r = a + b * c\n")
        stmt = prog.statements[0]
        assert isinstance(stmt, LetStatement)
        # Should be: a + (b * c) due to precedence
        assert isinstance(stmt.value, BinaryOp)
        assert stmt.value.op == "+"

    def test_matmul(self):
        prog = parse("let z = x @ y\n")
        stmt = prog.statements[0]
        assert isinstance(stmt.value, BinaryOp)
        assert stmt.value.op == "@"

    def test_matmul_plus_bias(self):
        prog = parse("let z = x @ y + 0.5\n")
        stmt = prog.statements[0]
        val = stmt.value
        assert isinstance(val, BinaryOp)
        assert val.op == "+"
        assert isinstance(val.left, BinaryOp)
        assert val.left.op == "@"

    def test_pipe_single(self):
        prog = parse("let z = x |> relu\n")
        stmt = prog.statements[0]
        assert isinstance(stmt.value, PipeExpression)
        assert len(stmt.value.stages) == 1
        assert stmt.value.stages[0].func_name == "relu"

    def test_pipe_chain(self):
        prog = parse("let z = x |> relu |> dropout(0.2)\n")
        stmt = prog.statements[0]
        pipe = stmt.value
        assert isinstance(pipe, PipeExpression)
        assert len(pipe.stages) == 2
        assert pipe.stages[0].func_name == "relu"
        assert pipe.stages[1].func_name == "dropout"
        assert len(pipe.stages[1].args) == 1

    def test_function_call(self):
        prog = parse("print(x)\n")
        stmt = prog.statements[0]
        assert isinstance(stmt, ExpressionStatement)
        assert isinstance(stmt.expression, FunctionCall)

    def test_function_call_kwargs(self):
        prog = parse("let d = mnist.load(batch_size=64, shuffle=true)\n")
        stmt = prog.statements[0]
        assert isinstance(stmt, LetStatement)
        call = stmt.value
        assert isinstance(call, FunctionCall)
        assert "batch_size" in call.kwargs

    def test_list_literal(self):
        prog = parse("let a = [1, 2, 3]\n")
        stmt = prog.statements[0]
        assert isinstance(stmt.value, ListLiteral)
        assert len(stmt.value.elements) == 3


class TestTensorLiteral:
    def test_basic_tensor(self):
        prog = parse("let x = tensor([1.0, 2.0, 3.0], dtype=float32, device=gpu)\n")
        stmt = prog.statements[0]
        assert isinstance(stmt, LetStatement)
        assert isinstance(stmt.value, TensorLiteral)
        assert len(stmt.value.elements) == 3
        assert "dtype" in stmt.value.kwargs
        assert "device" in stmt.value.kwargs


class TestModelDefinition:
    def test_simple_model(self):
        source = """model Net:
    layer fc1 = Linear(784, 256)
    layer fc2 = Linear(256, 10)
    fn forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x) |> relu
        x = self.fc2(x) |> softmax
        return x
"""
        prog = parse(source)
        assert len(prog.statements) == 1
        model = prog.statements[0]
        assert isinstance(model, ModelDefinition)
        assert model.name == "Net"
        assert len(model.layers) == 2
        assert model.layers[0].name == "fc1"
        assert model.layers[1].name == "fc2"
        assert len(model.methods) == 1
        assert model.methods[0].name == "forward"

    def test_model_with_multiple_layers(self):
        source = """model Deep:
    layer l1 = Linear(100, 50)
    layer l2 = Linear(50, 25)
    layer l3 = Linear(25, 10)
    fn forward(self, x: Tensor) -> Tensor:
        return x
"""
        prog = parse(source)
        model = prog.statements[0]
        assert isinstance(model, ModelDefinition)
        assert len(model.layers) == 3


class TestTrainBlock:
    def test_basic_train(self):
        source = """train net on data:
    epochs = 10
    optimizer = Adam(lr=0.001)
    loss = CrossEntropy
"""
        prog = parse(source)
        assert len(prog.statements) == 1
        train = prog.statements[0]
        assert isinstance(train, TrainBlock)
        assert train.model_name == "net"
        assert train.data_name == "data"
        assert "epochs" in train.params
        assert "optimizer" in train.params
        assert "loss" in train.params

    def test_train_with_precision(self):
        source = """train model on dataset:
    epochs = 5
    precision = fp16
"""
        prog = parse(source)
        train = prog.statements[0]
        assert "precision" in train.params
        p = train.params["precision"]
        assert isinstance(p, Identifier)
        assert p.name == "fp16"

    def test_train_with_checkpoint(self):
        source = """train net on data:
    epochs = 10
    checkpoint every 2 epochs
"""
        prog = parse(source)
        train = prog.statements[0]
        assert train.checkpoint_config is not None
        assert isinstance(train.checkpoint_config.frequency, NumberLiteral)
        assert train.checkpoint_config.frequency.value == 2


class TestControlFlow:
    def test_if_statement(self):
        source = """if x > 0:
    let y = 1
"""
        prog = parse(source)
        stmt = prog.statements[0]
        assert isinstance(stmt, IfStatement)
        assert isinstance(stmt.condition, BinaryOp)
        assert len(stmt.body) == 1

    def test_for_loop(self):
        source = """for i in range(10):
    print(i)
"""
        prog = parse(source)
        stmt = prog.statements[0]
        assert isinstance(stmt, ForLoop)
        assert stmt.variable == "i"
        assert len(stmt.body) == 1


class TestCompleteProgram:
    def test_hello_program(self):
        source = """let x = tensor([1.0, 2.0, 3.0], dtype=float32, device=gpu)
let y = tensor([4.0, 5.0, 6.0], dtype=float32, device=gpu)
let z = x @ y + 0.5
print(z)
"""
        prog = parse(source)
        assert len(prog.statements) == 4

    def test_full_training_pipeline(self):
        source = """import nn

model Classifier:
    layer fc1 = Linear(784, 256)
    layer fc2 = Linear(256, 10)
    fn forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x) |> relu
        x = self.fc2(x) |> softmax
        return x

let net = Classifier()
let data = load(batch_size=64)

train net on data:
    epochs = 10
    optimizer = Adam(lr=0.001)
    loss = CrossEntropy
    precision = fp16
"""
        prog = parse(source)
        # Should have: import, model, let, let, train
        assert len(prog.statements) == 5
        assert isinstance(prog.statements[0], ImportStatement)
        assert isinstance(prog.statements[1], ModelDefinition)
        assert isinstance(prog.statements[2], LetStatement)
        assert isinstance(prog.statements[3], LetStatement)
        assert isinstance(prog.statements[4], TrainBlock)
