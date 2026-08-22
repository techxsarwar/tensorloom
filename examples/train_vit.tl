// train_vit.tl — Vision Transformer Training Pipeline
//
// Imports ViT architecture from vision_transformer.nml
// and trains with all TensorLoom optimizations enabled.

import vision_transformer.nml as ViT

let net = ViT(embed_dim=384, num_heads=6, num_classes=100)
let data = load_dataset()

train net on data:
    epochs = 30
    optimizer = Adam(lr=0.0001)
    loss = CrossEntropy
    precision = fp16
    checkpoint every 10 epochs
