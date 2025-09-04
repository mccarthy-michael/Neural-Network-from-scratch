# Project READ_ME: Building a NumPy Neural Network (from scratch no Tensorflow or Keras) (MNIST)

## Project context
- Dataset: MNIST-like digits loaded from `train.csv` via `data_prep`.
- Model: 784 → h1 → h2 → 10 with ReLU in hidden layers and softmax output.
- Orientation: Column-major tensors (features, batch) to keep math close to the linear algebra formulas.
- Files:
  - Model/training: `my-neural-network.py`
  - Checkpoints: CSV weights/biases in `#-epochs/csv/`

## Timeline and key steps
1. Data preparation and shapes
   - Moved from row-major to column-major layout: `X` is `(features, batch)`, `Y` is `(batch,)`.
   - Ensured `float32` dtype and stable batching.

2. Architecture and initialization
   - Used He initialization for ReLU layers (`W1`, `W2`) to stabilize forward activations.
   - Used Xavier/Glorot for the softmax output layer (`W3`).

3. Class-based design
   - Chose a class (`MLP`) for stateful parameters and methods: `forward`, `back_prop`, `update_params`, `fit`, `evaluate`.
   - Kept helper utilities stateless where possible (`softmax`, `relu`, `one_hot`).

4. Forward pass and stability
   - Implemented numerically stable softmax (subtract column max).
   - Added cross-entropy loss with clipping.

5. Backpropagation
   - Implemented gradient flow with correct masking for ReLU and bias gradients summed over batch (axis=1, keepdims=True).

6. Training loop
   - Mini-batch gradient descent with averaged loss and gradient updates.
   - Added `evaluate`, epoch logging, and tester `test_model(...)`.

7. Persistence and reuse
   - Implemented `save_params_csv(...)` and `load_params_csv(...)` to avoid retraining every run.
   - Added CLI flags to choose train vs load in `__main__`.

## Where I got stuck (and why)

1. Orientation and axes (column-major vs row-major)
   - Why it was hard: Many tutorials use row-major `(batch, features)`; switching to `(features, batch)` changes where you put transposes and which axis to sum over for biases.
   - Fix: Standardized all shapes to column-major and documented axis conventions. Bias gradients now use `axis=1, keepdims=True`.

2. Initialization choices (He vs Xavier)
   - Why it was hard: It’s not obvious which init to use and where.
   - Fix: Use He for ReLU hidden layers; Xavier (Glorot) for the softmax output to keep logits/gradients balanced.

3. ReLU derivative and masking
   - Why it was hard: Easy to confuse pre-activations `Z` with activations `A` and forget the indicator mask.
   - Fix: Store `Z` in cache and use `(Z > 0)` mask for `dZ = dA * relu_grad(Z)`.

4. Bias gradient axis confusion
   - Why it was hard: Summing over the wrong axis silently produces wrong shapes/values.
   - Fix: In column-major, sum/mean over axis=1 for biases; average over batch to match loss scaling.

5. Softmax, cross-entropy, and stability
   - Why it was hard: Unstable softmax (without max subtraction) or log(0) in loss causes NaNs.
   - Fix: Stabilize softmax with max subtraction and clip probabilities before log.

6. “Nothing changes” after running
   - Why it was hard: Evaluating an untrained network twice looks identical.
   - Fix: Actually call `fit(...)` before evaluation, or load saved weights with `load_params_csv(...)`.

7. Domain gap in the drawing app
   - Symptom: High accuracy on dataset, low on hand-drawn input.
   - Why it was hard: MNIST digits are centered, scaled, and anti-aliased; raw drawings are not.
   - Fixes:
     - Preprocessing: crop bounding box, blur slightly, scale largest side to ~20 px, center on 28×28, normalize.
     - UI: pixel-grid editor to visualize the exact 28×28 inputs; thicker brush; live predictions.

## Conclusion

My compact MLP (784→256→128→10), with He initialization, stable softmax+cross‑entropy, and fully vectorized column‑major math, trains smoothly and quickly. On MNIST‑like data it reached ≥96% validation accuracy within ~5–10 epochs and ~100% train accuracy by ~20–30 epochs, largely due to sufficient model capacity and clean optimization. I am very happy with the results and efficency of my model, I enjoyed the process of learning about Neural Networks and it was a great introduction into the sector of Machine learning, my next project is to create an image classifaction program using CNNs.

## Remaining improvements
- Add simple data augmentation during training (shifts, small rotations, thickness jitter).
- Learning rate schedule and early stopping.
- Optional L2 weight decay or dropout.
- Package checkpoints into a single `.npz` (already present for a best model) or a small saver/loader utility.

## Final note

  Most diffcult part of the process was understanding the calculus for the backpropagation, a video that really helped - https://www.youtube.com/watch?v=SmZmBKc7Lrs

## References
  
  Holy Grail for understanding the underlying concepts of Neural Networks (Great visual representations) - https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi

  A bit outdated but good for where gaps appeared in knowledge - http://neuralnetworksanddeeplearning.com

  - Use of AI to generate MODEL_SPEC, and clean up README
