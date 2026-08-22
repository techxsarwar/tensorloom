# 🛠️ Guide 04: The Training Gym & Pipeline Scripts (`.tl`)

> *If `.nml` is the blueprint of a race car, `.tl` is the test track where you put the driver in the cockpit, fill the tank with high-octane fuel, and step on the gas pedal!*

---

## 🎯 What Happens in a `.tl` Script?

A TensorLoom (`.tl`) script is where action happens:
1. Creating numbers, strings, and lists with `let`.
2. Flowing tensors through mathematical waterslides using the pipe operator (`|>`).
3. Running high-speed training workouts with the `train` block.

---

## 1. Declaring Variables with `let`

In TensorLoom, variables are created cleanly using `let`:

```
let learning_rate = 0.001           // How big of a step the robot takes when learning
let batch_size = 64                 // How many flashcards it looks at simultaneously
let robot_name = "SuperBrain-3000"  // Text string
let use_gpu = true                  // Boolean flag
let dimensions = [784, 128, 10]     // List of layer sizes
```

---

## 2. The Waterslide Operator (`|>`)

In normal math and Python programming, functions wrap inside each other like Russian nesting dolls:

```python
# 😫 The Ugly Russian Nesting Doll Way (hard to read backwards):
output = relu(batch_norm(linear_layer(image)))
```

Notice how you have to read from the inside out? 

TensorLoom gives you the **Waterslide Pipe Operator (`|>`)**. Information flows smoothly from left to right, just like reading a comic book:

```
// 🏄 The TensorLoom Waterslide Way:
let output = image |> linear_layer |> batch_norm |> relu
```

### How the Waterslide Works:
1. `image` jumps into `linear_layer`.
2. The result slides directly into `batch_norm`.
3. That result slides into `relu`.
4. And lands softly into `output`!

---

## 3. The `train` Block: The AI's Personal Trainer

Training a neural network usually requires 40 lines of nested loops, zeroing gradients, computing losses, scaling gradients, and updating weights.

In TensorLoom, you simply tell the compiler what training settings you want:

```
train my_robot on training_data:
    epochs = 20
    optimizer = Adam(lr=0.001)
    loss = CrossEntropy
    precision = fp16
    checkpoint every 5 epochs
```

### What Each Setting Does:

| Setting | Kid Explanation | Real Technical Meaning |
| :--- | :--- | :--- |
| `epochs = 20` | "Read through the entire textbook 20 times." | Number of full dataset passes. |
| `optimizer = Adam(...)` | "The smart coach that nudges the math dials in the right direction." | Adaptive moment estimation optimizer. |
| `loss = CrossEntropy` | "The scoring referee that measures how wrong the robot's guess was." | Multi-class loss criterion. |
| `precision = fp16` | "Use lightweight 16-bit numbers to run 2x faster on the GPU!" | Automatic Mixed Precision (AMP). |
| `checkpoint every 5 epochs` | "Save a game checkpoint every 5 rounds so we don't lose progress." | Model weight snapshot serialization. |

The compiler takes these 6 lines and turns them into a production-grade PyTorch training loop complete with GPU memory transfers, loss scaling, metric tracking, and checkpoint saving!
