# Neural Network Technical Specification

## Overview
- Task: MNIST-like digit classification (0–9)
- Framework: NumPy (pure Python implementation)
- Data layout: column-major (each column is one example)

## Architecture
- Input: 784 features (28×28 grayscale)
- Hidden layer 1: 256 neurons, ReLU
- Hidden layer 2: 128 neurons, ReLU
- Output layer: 10 neurons, Softmax

### Parameter counts
- W1: 256×784 = 200,704, b1: 256
- W2: 128×256 = 32,768, b2: 128
- W3: 10×128 = 1,280, b3: 10
- Total trainable params: 235,146

## Initialization
- Hidden layers (W1, W2): He initialization (N(0, 2/fan_in))
- Output layer (W3): Glorot/Xavier-like (scale √(2/(h2+C)))
- Biases: zeros
- Dtype: float32
- Seed: 42 (MLP constructor)

## Forward pass (for batch size B)
- Shapes use (rows = classes/neurons, cols = batch examples)
- X: (784, B)
- z1 = W1 @ X + b1 → a1 = ReLU(z1), shape (256, B)
- z2 = W2 @ a1 + b2 → a2 = ReLU(z2), shape (128, B)
- logits = W3 @ a2 + b3, shape (10, B)
- probs = softmax(logits) column-wise, shape (10, B)

## Loss and gradients
- Loss: mean cross-entropy, L = −mean(log(probs[y_i, i])) with clipping to [1e−12, 1]
- Output gradient: dZ3 = probs − one_hot(Y)
- Layer grads (averaged over batch):
  - dW3 = (1/B) dZ3 @ a2.T, db3 = mean(dZ3, axis=1, keepdims=True)
  - Propagate: dA2 = W3.T @ dZ3, dZ2 = dA2 * ReLU’(z2)
  - dW2 = (1/B) dZ2 @ a1.T, db2 = mean(dZ2, axis=1, keepdims=True)
  - Propagate: dA1 = W2.T @ dZ2, dZ1 = dA1 * ReLU’(z1)
  - dW1 = (1/B) dZ1 @ X.T,  db1 = mean(dZ1, axis=1, keepdims=True)
- Update rule: SGD, θ ← θ − α · grad (no momentum)

## Training configuration
- Default CLI epochs: 10 (`--epochs`, main script)
- Default batch size: 128 (`--batch-size`)
- Default learning rate: 0.1 (`--alpha`)
- Shuffling: per-epoch permutation with `np.random.default_rng(seed)`
- Epoch loss: average of batch losses weighted by batch fraction (|batch|/m)

## Data pipeline (`data_prep.py`)
- Input CSV: first column = label, remaining 784 columns = pixels
- Shuffle rows with RNG seed (default 42)
- Split: first `dev_size=1000` rows → dev, rest → train
- Normalize pixels to [0, 1] by dividing by 255.0
- Column-major return (if `column_major=True`):
  - X_train: (784, m_train) Fortran-contiguous
  - y_train: (m_train,)
  - X_dev: (784, m_dev)
  - y_dev: (m_dev,)

## Evaluation & inference
- Accuracy: `argmax(probs, axis=0)` compared to labels
- Inference requires only forward pass; no loss/backprop needed (unless reporting metrics)
- Visualization helper: samples random columns and shows predicted class and confidence

## Checkpointing
- CSV per tensor in `#-epochs/csv`: W1, b1, W2, b2, W3, b3
- Loader reshapes 1D biases back to column vectors (n, 1)

## Reproducibility
- Seeds: `data_prep(seed=42)`, `MLP(seed=42)`

## Shapes cheat-sheet
- X: (784, B), Y: (B,)
- a1/z1: (256, B)
- a2/z2: (128, B)
- logits/probs: (10, B)
- dW1: (256, 784), db1: (256, 1)
- dW2: (128, 256), db2: (128, 1)
- dW3: (10, 128),  db3: (10, 1)

## Why this implementation is efficient
- Column-major batches (Fortran-contiguous): Inputs `X` are stored as (features, batch) with columns contiguous in memory. This aligns with how we consume columns as examples and lets BLAS operate on contiguous column blocks, reducing implicit copies/transposes and improving cache locality during GEMM (W @ X).
- Full vectorization over batches: All layers use matrix multiplies and broadcasts (no Python loops per example). NumPy delegates `@` to optimized BLAS (OpenBLAS/MKL), exploiting SIMD and multi-threading.
### FIRST TWO POINTS HELP INCREASE EFFICIENCY ALOT
- Consistent shapes minimize transposes: Keeping the (C, B) and (h, B) convention throughout forward/backward avoids extra `transpose` ops and temporary arrays.
- Broadcast-friendly bias shapes: Biases are (units, 1) so `+ b` broadcasts over columns without per-sample Python work or reshaping.
- Float32 everywhere: Halves memory bandwidth vs float64 and is plenty precise for MNIST; speeds GEMM and softmax.
- Stable softmax + clipped CE: Prevents NaNs/inf that would stall the compute path or force slow fallbacks.
- Batch-averaged grads: Computes means once per batch (axis reductions) instead of per-sample accumulations.
- Single RNG permutation per epoch: Generates indices once, then slices, avoiding per-iteration shuffles.
- Simple SGD without Python overhead: Manual backprop with closed-form gradients (softmax+CE ⇒ `probs − one_hot`) avoids autograd/tracing overhead.
