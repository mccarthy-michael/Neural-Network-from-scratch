"""
Notation and conventions
- Column-major tensors: columns are examples.
- X: input matrix of images, shape (d, m) where d=784 (features) and m=#examples.
- Y: label vector, shape (m,), integer class ids in [0, C-1].
- d: input_dim; h1, h2: hidden layer sizes; C: num_classes; B: batch size.
- Weights/biases shapes:
  * W1: (h1, d),  b1: (h1, 1)
  * W2: (h2, h1), b2: (h2, 1)
  * W3: (C, h2),  b3: (C, 1)
- Forward pass per batch X:
  z1 = W1 @ X + b1; a1 = ReLU(z1)
  z2 = W2 @ a1 + b2; a2 = ReLU(z2)
  z3 = W3 @ a2 + b3; probs = softmax(z3)  # probs has shape (C, m)
- one_hot(Y, C): (C, m) with ones at [Y[i], i].
- Loss (cross-entropy): L = -mean(log(probs[Y[i], i])) over columns i.
- Axis conventions: argmax(probs, axis=0) → class per column; bias grads are means over axis=1.
- rng: NumPy Generator (np.random.default_rng(seed)) for reproducible init/shuffle.
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from data_prep import data_prep as dp


"""

Orginal data prep

"""
# data = pd.read_csv("./train.csv")

# data = np.array(data)

# m, n = data.shape

# print(m, n)

# np.random.shuffle(data)

# # Turns each column instead of row into number 
# data_dev = data[0:1000].T

# Y_dev = data_dev[0]

# X_dev = data_dev[1:n]

# data_train = data[0:1000].T

# Y_train = data_train[0]
# X_train = data_train[1:n]

# # print(X_train[:, 0].shape)

# Goes through each element in Z and if its greater than 0 it returns Z, else returns 0 and neuron wont be "fired"

"""

Before I decided to use a classes

"""
# def relu(Z):
#     return np.maximum(0,Z)

# def softmax(Z):
#     return np.exp() / np.sum(np.exp(Z))

# def init_params(num_inputs=784, hidden=128, num_classes=10, dtype=np.float32, seed=42):
#     rng = np.random.default_rng(seed)
#     W1 = (rng.standard_normal((hidden,num_inputs)) * np.sqrt(2.0 / num_inputs)).astype(dtype)
#     b1 = np.zeros((hidden, 1), dtype=dtype)
#     # Xavier/Glorot init for output layer
#     W2 = (rng.standard_normal((num_classes, hidden)) * np.sqrt(1.0 / hidden)).astype(dtype)
#     b2 = np.zeros((num_classes, 1), dtype=dtype)

#     return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

# def forward(X,params):
#     Z1 = params["W1"] @ X + params["b1"]
#     A1 = relu(Z1)
#     Z2 = params["W2"] @ A1 + params["b2"]
#     A2 = softmax(Z2)
#     return Z1, A1

class MLP:
    def __init__(self, input_dim=784, h1=256, h2=128, num_classes=10, seed=42):
        rng = np.random.default_rng(seed)
        self.num_classes=num_classes
        # Column-major params: X is (input_dim, batch)
        self.W1 = (rng.standard_normal((h1, input_dim)) * np.sqrt(2.0 / input_dim)).astype(np.float32)  # He
        self.b1 = np.zeros((h1, 1), dtype=np.float32)

        self.W2 = (rng.standard_normal((h2, h1)) * np.sqrt(2.0 / h1)).astype(np.float32)                 # He
        self.b2 = np.zeros((h2, 1), dtype=np.float32)

        # Glorot for output layer feeding softmax
        self.W3 = (rng.standard_normal((num_classes, h2)) * np.sqrt(2.0 / (h2 + num_classes))).astype(np.float32)
        self.b3 = np.zeros((num_classes, 1), dtype=np.float32)

    @staticmethod
    def relu(x):
        return np.maximum(x, 0).astype(np.float32)

    @staticmethod
    def relu_grad(x):
        return (x > 0).astype(np.float32)

    @staticmethod
    def softmax(logits):
        # logits: (num_classes, batch)
        z = logits - np.max(logits, axis=0, keepdims=True)
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)
    
    @staticmethod
    def one_hot(Y, num_classes):
        y = np.asarray(Y, dtype=np.int64).ravel()
        m = y.size
        out = np.zeros((num_classes, m), dtype=np.float32)
        out[y, np.arange(m)] = 1.0
        return out

    def forward(self, X):
        # X: (input_dim, batch)
        z1 = self.W1 @ X + self.b1          # (h1, batch)
        a1 = self.relu(z1)
        z2 = self.W2 @ a1 + self.b2         # (h2, batch)
        a2 = self.relu(z2)
        logits = self.W3 @ a2 + self.b3    #logits # (num_classes, batch)
        probs = self.softmax(logits) #probs 
        cache = (X, z1, a1, z2, a2, logits, probs)
        return probs, cache
    
    
    # Gosh this is tricky, me no like calculus took the majority of my time
    def back_prop(self,cache,Y):
        m = Y.size
        one_hot_Y = MLP.one_hot(Y,self.num_classes)
        
        # Get loss with respect to z
        
        
        dZ3 = cache[6] - one_hot_Y

        # Calculate output layer gradients
        dW3 = 1/m * dZ3 @ cache[4].T
        db3 = dZ3.mean(axis=1,keepdims=True)
     
        dA2 = self.W3.T @ dZ3
        
        # Derive relu (reverse relu) * dA2 to get dZ2
        dZ2 = dA2 * self.relu_grad(cache[3])
        
        # Hidden layer 2
        dW2 = 1/m * dZ2 @ cache[2].T
        db2 = dZ2.mean(axis=1,keepdims=True)
        
        dA1 = self.W2.T @ dZ2
        
        # Same as for dZ2
        dZ1 = dA1 * self.relu_grad(cache[1])
        
        dW1 = 1/m * dZ1 @ cache[0].T
        
        db1 = dZ1.mean(axis=1, keepdims=True)
        
        return dW1, db1, dW2, db2, dW3, db3
        
    def update_params(self, dW1, db1, dW2, db2, dW3, db3, alpha):
        
        self.W1 -= alpha * dW1
        self.b1 -= alpha * db1
        self.W2 -= alpha * dW2
        self.b2 -= alpha * db2
        self.W3 -= alpha * dW3
        self.b3 -= alpha * db3
        
    #change to untrained/csv if accidentally overwrites
    def save_params_csv(self, dir_path="archived/csv"):
        """Save weights and biases as CSV files into dir_path."""
        import os
        os.makedirs(dir_path, exist_ok=True)
        np.savetxt(os.path.join(dir_path, "W1.csv"), self.W1, delimiter=",")
        np.savetxt(os.path.join(dir_path, "b1.csv"), self.b1, delimiter=",")
        np.savetxt(os.path.join(dir_path, "W2.csv"), self.W2, delimiter=",")
        np.savetxt(os.path.join(dir_path, "b2.csv"), self.b2, delimiter=",")
        np.savetxt(os.path.join(dir_path, "W3.csv"), self.W3, delimiter=",")
        np.savetxt(os.path.join(dir_path, "b3.csv"), self.b3, delimiter=",")

    def load_params_csv(self, dir_path="checkpoints/csv"):
        """Load weights and biases from CSV files into this model instance."""
        import os
        self.W1 = np.loadtxt(os.path.join(dir_path, "W1.csv"), delimiter=",").astype(np.float32)
        self.b1 = np.loadtxt(os.path.join(dir_path, "b1.csv"), delimiter=",").astype(np.float32)
        self.W2 = np.loadtxt(os.path.join(dir_path, "W2.csv"), delimiter=",").astype(np.float32)
        self.b2 = np.loadtxt(os.path.join(dir_path, "b2.csv"), delimiter=",").astype(np.float32)
        self.W3 = np.loadtxt(os.path.join(dir_path, "W3.csv"), delimiter=",").astype(np.float32)
        self.b3 = np.loadtxt(os.path.join(dir_path, "b3.csv"), delimiter=",").astype(np.float32)
        # Ensure biases are column vectors
        # Was annoying to fix bug
        if self.b1.ndim == 1:
            self.b1 = self.b1.reshape(-1, 1)
        if self.b2.ndim == 1:
            self.b2 = self.b2.reshape(-1, 1)
        if self.b3.ndim == 1:
            self.b3 = self.b3.reshape(-1, 1)

    # -------------------- Training helpers --------------------
    def compute_loss(self, probs, Y):
        """Cross-entropy loss for column-major softmax outputs.
        probs: (C, B), Y: (B,) 
        
        Used to represent weight updates
        
        """
        y = np.asarray(Y, dtype=np.int64).ravel()
        m = y.size
        # Pick the probability assigned to the true class per column
        p_true = probs[y, np.arange(m)]
        # Numerical stability
        p_true = np.clip(p_true, 1e-12, 1.0)
        return float(-np.log(p_true).mean())

    def evaluate(self, X, Y):
        """Return accuracy and loss on a dataset.
        X: (d, m), Y: (m,)
        """
        probs, _ = self.forward(X)
        preds = np.argmax(probs, axis=0)
        acc = float(np.mean(preds == Y))
        loss = self.compute_loss(probs, Y)
        return acc, loss

    def fit(self, X, Y, X_val=None, Y_val=None, epochs=10, batch_size=128, alpha=0.1, print_every=1, seed=42):
        """Mini-batch gradient descent training loop.
        Defaults tuned for MNIST-like data to reach ~90%+ in a few epochs.
        """
            
        rng = np.random.default_rng(seed)
        m = X.shape[1]
        for epoch in range(1, epochs + 1):
            perm = rng.permutation(m)
            epoch_loss = 0.0
            for start in range(0, m, batch_size):
                idx = perm[start:start + batch_size]
                Xb = X[:, idx]
                # print(Xb)
                Yb = Y[idx]
                # print(Yb)
                # performs forward prop to prepare for back
                probs, cache = self.forward(Xb)
                # Weight batch loss contribution by batch fraction to form an epoch average
                batch_loss = self.compute_loss(probs, Yb)
                epoch_loss += batch_loss * (Yb.size / m)
                # peforms back_prop
                dW1, db1, dW2, db2, dW3, db3 = self.back_prop(cache, Yb)
                #updates params, by tiny winy nudges each iteration
                self.update_params(dW1, db1, dW2, db2, dW3, db3, alpha)
            if print_every and (epoch % print_every == 0):
                train_acc, train_loss = self.evaluate(X, Y)
                if X_val is not None and Y_val is not None:
                    val_acc, val_loss = self.evaluate(X_val, Y_val)
                    print(f"Epoch {epoch}/{epochs} - loss {epoch_loss:.4f} - train_acc {train_acc*100:.2f}% - val_acc {val_acc*100:.2f}%")
                else:
                    print(f"Epoch {epoch}/{epochs} - loss {epoch_loss:.4f} - train_acc {train_acc*100:.2f}%")
    
    
# Helper function to visualize working NN
def test_model(model, X, Y, count=9, seed=5):
    """Visualize a few predictions from X with matplotlib.
    Shows count images with Pred/True and confidence.
    X: (784, m), Y: (m,)
    """
    rng = np.random.default_rng(seed)
    m = X.shape[1]
    count = int(min(count, m))
    idx = rng.choice(m, size=count, replace=False)
    Xb = X[:, idx]
    Yb = Y[idx]

    probs, _ = model.forward(Xb)
    preds = np.argmax(probs, axis=0)
    confs = probs[preds, np.arange(count)]

    # Grid size
    cols = int(np.ceil(np.sqrt(count)))
    rows = int(np.ceil(count / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    axes = np.array(axes).reshape(rows, cols)

    for i in range(rows * cols):
        ax = axes.flat[i]
        if i < count:
            img = Xb[:, i].reshape(28, 28)
            ax.imshow(img, cmap="gray")
            pred = int(preds[i])
            true = int(Yb[i])
            conf = float(confs[i])
            color = "g" if pred == true else "r"
            ax.set_title(f"P:{pred} T:{true} ({conf:.2f})", color=color, fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    plt.show()
    return preds, confs, idx

if __name__ == "__main__":
    import argparse
    
    """
        To init randomly and not train:
            "py my-nerural-network.py"
        To train and customise own neural network (will overite and store in archive):
            "py my-neural-network.py --train --epochs #"
        
        To test different epochs run in CLI:
            "py my-neural-network.py --ckpt 10-epochs/csv"  
            "py my-neural-network.py --ckpt 30-epochs/csv"  
            "py my-neural-network.py --ckpt 50-epochs/csv"  
            "py my-neural-network.py --ckpt 150-epochs/csv"  
    """

    # used argparse to add customization
    
    parser = argparse.ArgumentParser(description="Train or evaluate the MLP on MNIST-like CSV data.")
    parser.add_argument("--train", action="store_true", help="Train the model; otherwise load from checkpoints.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=128, help="Mini-batch size.")
    parser.add_argument("--alpha", type=float, default=0.1, help="Learning rate.")
    parser.add_argument("--ckpt", type=str, default="archived/csv", help="Directory for CSV checkpoints.")
    args = parser.parse_args()
    
    # Adjust seeds in calls to randomise dataset and dev_size to increase size of dataset, leave column major always as True or will break

    X_train, Y_train, X_dev, Y_dev = dp("./train.csv", dev_size=1000, seed=10, column_major=True)
    mlp = MLP(input_dim=784, h1=256, h2=128, num_classes=10, seed=42)
    mlp.save_params_csv()

    if args.train:
        mlp.fit(X_train, Y_train, X_val=X_dev, Y_val=Y_dev, epochs=args.epochs, batch_size=args.batch_size, alpha=args.alpha)
        mlp.save_params_csv(args.ckpt)
    else:
        mlp.load_params_csv(args.ckpt)
    
    test_model(mlp, X_dev, Y_dev, count=42)
