// train_transformer.tl — Cross-file NML Import Example
//
// Imports a Transformer model from transformer.nml and
// trains it with a simple training loop.

import transformer.nml as MyTransformer

let net = MyTransformer(d_model=256, n_heads=4)
let data = load_dataset()

train net on data:
    epochs = 5
    optimizer = Adam(lr=0.0001)
    loss = CrossEntropy
    precision = fp16
