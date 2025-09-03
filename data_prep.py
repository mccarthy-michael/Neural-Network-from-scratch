import numpy as np
import pandas as pd


def data_prep(path, dev_size=1000, seed=42, column_major=True):
    rng = np.random.default_rng(seed)

    data = pd.read_csv(path).to_numpy()
    rng.shuffle(data)  # shuffle rows

    m, n = data.shape
    assert n >= 2, "Expected label + features in CSV"

    # Split: first column = label, rest = pixels
    dev = data[:dev_size]
    train = data[dev_size:]

    y_dev = dev[:, 0].astype(np.int64, copy=False)
    X_dev = dev[:, 1:].astype(np.float32, copy=False)

    y_train = train[:, 0].astype(np.int64, copy=False)
    X_train = train[:, 1:].astype(np.float32, copy=False)

    # Normalize pixels to [0,1]
    X_train /= 255.0
    X_dev /= 255.0

    if column_major:
        # Return (features, batch); transpose to make matrix multiplcation more effiecient 
        X_train = np.asfortranarray(X_train.T)  # (784, m_train)
        X_dev = np.asfortranarray(X_dev.T)      # (784, m_dev)

    return X_train, y_train, X_dev, y_dev