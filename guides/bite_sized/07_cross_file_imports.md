# ⚡ Lesson 07: Cross-File Imports (1 Min)

Import `.nml` blueprints directly into your `.tl` scripts with `import ... as ...`.

---

### 1. The Workflow
```
models/
  └── resnet.nml         ➔ Declarative blueprint
train.tl                 ➔ Imports resnet.nml and trains it
```

---

### 2. Import & Override Syntax (`train.tl`)
```
// 1. Import the blueprint with an alias
import models.resnet.nml as ResNet

// 2. Instantiate with custom config overrides
let net = ResNet(channels=128)
let data = load_dataset()

// 3. Train it!
train net on data:
    epochs = 10
    optimizer = Adam(lr=0.001)
    loss = CrossEntropy
```

---

### 3. What the Compiler Does
1. Sub-compiles `resnet.nml` in isolation.
2. Renames the generated class to your alias (`ResNet`).
3. Injects the class directly into the top of the output Python file.

---

### 💡 Key Takeaway
The generated Python file is 100% self-contained—zero missing imports when deployed on cloud clusters.
