# ⚡ Guide 02: 5-Minute Quickstart (Building Your First AI Brain)

> *Welcome to your first day in the AI Robot Lab! In the next 5 minutes, we are going to build a smart robot brain that can read handwritten numbers, check its memory, and compile it into blazing-fast machine code.*

---

## 🛠️ Step 1: Install Your Robot Toolkit

Open your terminal (the black command window where you talk to your computer) and run:

```bash
pip install tensorloom
```

To make sure your compiler is ready and listening:

```bash
python -m tensorloom --help
```

If you see a colorful ASCII banner with `TensorLoom Compiler`, you're all set! 🚀

---

## 📝 Step 2: Write Your First Robot Brain (`hello_mnist.tl`)

Let's write a program that reads numbers written by human hands (the famous MNIST dataset). 

Create a file named `hello_mnist.tl` and write this code:

```
// hello_mnist.tl — Our Number-Reading Robot Brain

model DigitReader:
    // Layer 1: Takes 784 pixels (a 28x28 picture) and compresses them into 128 clues
    layer eye = Linear(784, 128)
    
    // Layer 2: Takes the 128 clues and decides which number (0 to 9) it is
    layer brain = Linear(128, 10)

    // The thought process: Picture -> Eye -> Energy Boost (ReLU) -> Brain Decision
    fn forward(self, picture: Tensor) -> Tensor:
        return picture |> self.eye |> relu |> self.brain

// 1. Create a fresh robot instance
let net = DigitReader()

// 2. Open our big book of practice flashcards
let data = load_dataset()

// 3. Send the robot to the training gym!
train net on data:
    epochs = 5
    optimizer = Adam(lr=0.001)
    loss = CrossEntropy
    precision = fp16
```

---

## 🔍 Step 3: The X-Ray Scan (`tlc info`)

Before running a real sports car, you check under the hood. Let's ask TensorLoom to X-ray our robot:

```bash
python -m tensorloom info hello_mnist.tl
```

### What TensorLoom Sees:
```
  +================================================+
  |   TensorLoom Compiler  v0.1.0                  |
  |   GPU-Efficient Language for AI Training        |
  +================================================+

[info] Model Summary: DigitReader
   Total Parameters: 101,770 (0.10M tiny math gears)
   Trainable:        101,770
   Estimated Memory: ~0.39 MB (FP32) / ~0.19 MB (FP16)
   Status:           Valid & Ready for GPU!
```

TensorLoom counted all **101,770 tiny math knobs** in our robot and told us it will only take **0.19 Megabytes** of memory in FP16 mode!

---

## ⚙️ Step 4: The Transformation (Transpile to PyTorch)

Now comes the magic! Let's tell TensorLoom to translate our clean code into super-optimized PyTorch Python:

```bash
python -m tensorloom compile hello_mnist.tl -o run_mnist.py
```

Open `run_mnist.py`. Look at what the compiler created for you automatically:
1. **Device Detection**: Automatically looks for a gaming GPU (`cuda`) and uses it; otherwise falls back to CPU.
2. **PyTorch `nn.Module`**: Generates the full Python class structure.
3. **`torch.compile(mode="max-autotune")`**: Tells the GPU to fuse operations for maximum speed.
4. **Mixed Precision (`autocast` + `GradScaler`)**: Makes the GPU do math twice as fast using 16-bit floating point numbers without losing accuracy.

---

## 🏃 Step 5: Run Your Robot!

Now, run your compiled script:

```bash
python run_mnist.py
```

Or do the compile-and-run in one quick step:

```bash
python -m tensorloom run hello_mnist.tl
```

You'll see your robot practicing through 5 epochs, getting smarter and smarter with every batch of numbers! 🧠✨
