"""Preprocessing: split, standardization, bias column, unit conversion."""
import numpy as np


def train_test_split(X, y, test_size: float = 0.2, seed: int = 42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    cut = int(len(X) * (1 - test_size))
    tr, te = idx[:cut], idx[cut:]
    return X[tr], X[te], y[tr], y[te]


class Standardizer:
    """z = (x - mean) / std, fitted on TRAIN data only (avoids data leakage)."""

    def fit(self, A):
        self.mean_ = A.mean(axis=0)
        std = A.std(axis=0)
        self.std_ = np.where(std == 0, 1.0, std)
        return self

    def transform(self, A):
        return (A - self.mean_) / self.std_

    def fit_transform(self, A):
        return self.fit(A).transform(A)

    def inverse_transform(self, Z):
        return Z * self.std_ + self.mean_


def add_bias(X):
    """Prepend a column of 1s so the bias is just another weight."""
    return np.column_stack([np.ones(len(X)), X])


def to_original_units(w, x_scaler: Standardizer, y_scaler: Standardizer):
    """Convert weights learned on standardized data back to raw units."""
    w_feat = w[1:] * y_scaler.std_ / x_scaler.std_
    bias = y_scaler.mean_ + w[0] * y_scaler.std_ - np.sum(w_feat * x_scaler.mean_)
    return np.concatenate([[bias], w_feat])
