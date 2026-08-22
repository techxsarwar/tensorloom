// broken_model.tl — Intentional dimension mismatch for testing tlc check
//
// This model has a bug: fc1 outputs 256 features but fc2
// expects 128 inputs. TensorLoom's shape inference should
// catch this at compile time — no GPU time wasted!

model BrokenClassifier:
    layer fc1 = Linear(784, 256)
    layer fc2 = Linear(128, 10)
    fn forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x) |> relu
        x = self.fc2(x) |> softmax
        return x
