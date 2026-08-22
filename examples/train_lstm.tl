// train_lstm.tl — LSTM Sequence-to-Sequence Training
//
// Imports encoder from lstm_seq2seq.nml and trains
// with mixed precision for fast convergence.

import lstm_seq2seq.nml as SeqModel

let net = SeqModel(vocab_size=10000, hidden_dim=1024)
let data = load_dataset()

train net on data:
    epochs = 15
    optimizer = Adam(lr=0.0003)
    loss = CrossEntropy
    precision = fp16
