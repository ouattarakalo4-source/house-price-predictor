"""Linear regression from scratch: closed form, gradient descent, SGD, Ridge.

Loss (MSE):     J(w) = 1/m * ||Xw - y||^2  + l2 * ||w||^2   (bias NOT penalized)
Gradient:       dJ/dw = (2/m) * X^T (Xw - y) + 2 * l2 * w
Normal eq.:     dJ/dw = 0  =>  (X^T X + m*l2*P) w = X^T y   (P = identity, 0 for bias)
Descent step:   w <- w - lr * dJ/dw
Mini-batch SGD: same step, but the gradient uses a random subset of rows.
l2 = 0 gives plain linear regression; l2 > 0 gives Ridge regression.
"""
import numpy as np

from src.preprocessing import add_bias


# ---- Loss, gradient, metrics (Xb here already includes the bias column) ----
def mse(Xb, y, w, l2: float = 0.0):
    """MSE + Ridge penalty. w[0] is the bias and is never penalized."""
    r = Xb @ w - y
    return float(np.mean(r ** 2) + l2 * np.sum(w[1:] ** 2))


def gradient(Xb, y, w, l2: float = 0.0):
    g = (2 / len(y)) * Xb.T @ (Xb @ w - y)
    g[1:] += 2 * l2 * w[1:]
    return g


def numerical_gradient(Xb, y, w, eps: float = 1e-6, l2: float = 0.0):
    """Derivative from its definition: (J(w+eps) - J(w-eps)) / 2eps."""
    g = np.zeros_like(w, dtype=float)
    for j in range(len(w)):
        step = np.zeros_like(w, dtype=float)
        step[j] = eps
        g[j] = (mse(Xb, y, w + step, l2) - mse(Xb, y, w - step, l2)) / (2 * eps)
    return g


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1 - ss_res / ss_tot)


# ---- Model -----------------------------------------------------------------
class LinearRegression:
    def __init__(self):
        self.w = None
        self.loss_history = []

    def fit_normal(self, X, y, l2: float = 0.0):
        Xb = add_bias(X)
        m, d = Xb.shape
        P = np.eye(d)
        P[0, 0] = 0  # do not penalize the bias
        self.w = np.linalg.solve(Xb.T @ Xb + m * l2 * P, Xb.T @ y)
        return self

    def fit_gd(self, X, y, lr: float = 0.1, epochs: int = 500, l2: float = 0.0):
        """Full-batch gradient descent: one update per epoch."""
        Xb = add_bias(X)
        w = np.zeros(Xb.shape[1])
        self.loss_history = []
        for _ in range(epochs):
            w = w - lr * gradient(Xb, y, w, l2)
            self.loss_history.append(mse(Xb, y, w, l2))
        self.w = w
        return self

    def fit_sgd(self, X, y, lr: float = 0.05, epochs: int = 50,
                batch_size: int = 32, l2: float = 0.0, seed: int = 0):
        """Mini-batch SGD. batch_size=1 -> pure stochastic; batch_size=len(y) -> full GD.

        Each epoch: shuffle rows, split into batches, one update per batch.
        loss_history stores the FULL-data loss after each epoch.
        """
        Xb = add_bias(X)
        m = len(y)
        rng = np.random.default_rng(seed)
        w = np.zeros(Xb.shape[1])
        self.loss_history = []
        for _ in range(epochs):
            idx = rng.permutation(m)
            for start in range(0, m, batch_size):
                b = idx[start:start + batch_size]
                w = w - lr * gradient(Xb[b], y[b], w, l2)
            self.loss_history.append(mse(Xb, y, w, l2))
        self.w = w
        return self

    def predict(self, X):
        return add_bias(X) @ self.w
