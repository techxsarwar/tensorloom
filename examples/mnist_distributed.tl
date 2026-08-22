// mnist_distributed.tl — TensorLoom Multi-GPU Training
//
// This is the exact same MNIST classifier as mnist.tl,
// but with `distributed = true` to automatically scale
// across all available GPUs via PyTorch DDP.
//
// The TensorLoom compiler automatically injects:
//   - dist.init_process_group / destroy_process_group
//   - DistributedDataParallel wrapping
//   - DistributedSampler with sampler.set_epoch()
//   - Rank-aware checkpointing and logging
//
// Launch: torchrun --nproc_per_node=4 mnist_distributed_compiled.py

import data.mnist
import nn

model MNISTClassifier:
    layer input  = Linear(784, 256)
    layer hidden = Linear(256, 128)
    layer output = Linear(128, 10)
    fn forward(self, x: Tensor) -> Tensor:
        x = self.input(x) |> relu |> dropout(0.2)
        x = self.hidden(x) |> relu |> dropout(0.2)
        x = self.output(x) |> softmax
        return x

let net = MNISTClassifier()

train net on data:
    epochs = 10
    batch_size = 64
    optimizer = Adam(lr=0.001)
    loss = CrossEntropy
    precision = fp16
    distributed = true
    checkpoint every 2 epochs

    on epoch_end(metrics):
        print(f"Epoch {metrics.epoch}: loss={metrics.loss:.4f}, acc={metrics.accuracy:.2%}")
