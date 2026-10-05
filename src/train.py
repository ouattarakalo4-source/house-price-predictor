"""End-to-end training.

    python -m src.train                                  # synthetic data
    python -m src.train --l2 0.1                         # Ridge regression
    python -m src.train --dataset california --l2 0.01   # real data (needs internet once)
"""
import argparse

import numpy as np

from src.data import (CALIFORNIA_TARGET, NOISE_STD, TARGET, TRUE_WEIGHTS,
                      load_california, load_houses, to_arrays)
from src.model import LinearRegression, r2_score, rmse
from src.preprocessing import Standardizer, to_original_units, train_test_split


def main(dataset: str = "synthetic", l2: float = 0.0, lr: float = 0.1, epochs: int = 500):
    if dataset == "california":
        df, target, unit = load_california(), CALIFORNIA_TARGET, "x $100k"
    else:
        df, target, unit = load_houses(), TARGET, "$"
    names = [c for c in df.columns if c != target]
    X, y = to_arrays(df, target)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y)

    xs, ys = Standardizer().fit(X_tr), Standardizer().fit(y_tr)
    Z_tr, Z_te = xs.transform(X_tr), xs.transform(X_te)
    yz_tr = ys.transform(y_tr)

    models = {
        "Normal equation": LinearRegression().fit_normal(Z_tr, yz_tr, l2=l2),
        "Gradient descent": LinearRegression().fit_gd(Z_tr, yz_tr, lr=lr, epochs=epochs, l2=l2),
        "Mini-batch SGD": LinearRegression().fit_sgd(Z_tr, yz_tr, lr=0.02, epochs=50,
                                                     batch_size=32, l2=l2),
    }

    kind = f"Ridge (l2={l2})" if l2 > 0 else "plain"
    print(f"Dataset: {dataset} | {kind} | train={len(X_tr)} test={len(X_te)} | RMSE unit: {unit}\n")
    for name, m in models.items():
        pred = ys.inverse_transform(m.predict(Z_te))
        print(f"{name:17s} RMSE={rmse(y_te, pred):>10,.3f}  R2={r2_score(y_te, pred):.4f}")

    w = to_original_units(models["Gradient descent"].w, xs, ys)
    print("\nWeights (bias, " + ", ".join(names) + "):")
    print("  learned:", np.round(w, 3))
    if dataset == "synthetic":
        print("  true   :", TRUE_WEIGHTS, f"(noise std {NOISE_STD:,})")
        house = np.array([[150, 3, 10]], dtype=float)
        price = ys.inverse_transform(models["Gradient descent"].predict(xs.transform(house)))[0]
        print(f"\n150 m2, 3 bedrooms, 10 years old -> {price:,.0f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["synthetic", "california"], default="synthetic")
    ap.add_argument("--l2", type=float, default=0.0, help="Ridge strength (0 = plain regression)")
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=500)
    a = ap.parse_args()
    main(a.dataset, a.l2, a.lr, a.epochs)
