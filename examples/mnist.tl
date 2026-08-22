// TensorLoom — MNIST Classifier
// Full training pipeline demonstrating model definition, data loading, and training

import data.mnist
import nn

// Model definition with the 'model' keyword
model MNISTClassifier:
    layer input  = Linear(784, 256)
    layer hidden = Linear(256, 128)
    layer output = Linear(128, 10)

    fn forward(self, x: Tensor) -> Tensor:
        x = self.input(x) |> relu |> dropout(0.2)
        x = self.hidden(x) |> relu |> dropout(0.2)
        x = self.output(x) |> softmax
        return x

// Instantiate model and load data
let net = MNISTClassifier()
let data = mnist.load(batch_size=64, shuffle=true)

// Training with built-in constructs
train net on data:
    epochs = 10
    optimizer = Adam(lr=0.001)
    loss = CrossEntropy
    precision = fp16
    checkpoint every 2 epochs

    on epoch_end(metrics):
        print(f"Epoch {metrics.epoch}: loss={metrics.loss:.4f}, acc={metrics.accuracy:.2%}")
