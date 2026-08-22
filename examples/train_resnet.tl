// train_resnet.tl — ResNet Training Pipeline
//
// Imports a residual block from resnet.nml and trains it
// with mixed precision and gradient checkpointing.

import resnet.nml as ResBlock

let net = ResBlock(channels=128)
let data = load_dataset()

train net on data:
    epochs = 20
    optimizer = Adam(lr=0.001)
    loss = CrossEntropy
    precision = fp16
    checkpoint every 5 epochs
