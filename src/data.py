"""Data generation and loading.

Synthetic data: price = bias + w1*size + w2*bedrooms + w3*age + Gaussian noise.
Run `python -m src.data` to (re)generate data/houses.csv.
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "houses.csv"
FEATURES = ["size_m2", "bedrooms", "age_years"]
TARGET = "price"

# Ground truth used to generate the data: bias, size, bedrooms, age
TRUE_WEIGHTS = np.array([50_000, 900, 4_000, -600])
NOISE_STD = 15_000


def generate_houses(n: int = 500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    size = rng.normal(120, 40, n).clip(30, 300)
    bedrooms = rng.integers(1, 6, n)
    age = rng.uniform(0, 50, n)
    noise = rng.normal(0, NOISE_STD, n)
    w = TRUE_WEIGHTS
    price = w[0] + w[1] * size + w[2] * bedrooms + w[3] * age + noise
    return pd.DataFrame(
        {
            "size_m2": size.round(1),
            "bedrooms": bedrooms,
            "age_years": age.round(1),
            "price": price.round(0),
        }
    )


def save_houses(df: pd.DataFrame, path: Path = DATA_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def load_houses(path: Path = DATA_PATH) -> pd.DataFrame:
    if not Path(path).exists():
        save_houses(generate_houses(), path)
    return pd.read_csv(path)


def to_arrays(df: pd.DataFrame, target: str = TARGET):
    """DataFrame -> (X matrix of shape (n, d), y vector of shape (n,)).

    Every column except `target` is a feature.
    """
    features = [c for c in df.columns if c != target]
    return df[features].to_numpy(dtype=float), df[target].to_numpy(dtype=float)


# ---- Real dataset: California Housing (downloaded once by scikit-learn) ----
CALIFORNIA_TARGET = "MedHouseVal"  # median house value, in units of $100,000


def load_california() -> pd.DataFrame:
    """8 features (MedInc, HouseAge, AveRooms, ...) + MedHouseVal. Needs internet once."""
    try:
        from sklearn.datasets import fetch_california_housing
    except ImportError as e:
        raise ImportError("California Housing needs scikit-learn: pip install scikit-learn") from e
    return fetch_california_housing(as_frame=True).frame


if __name__ == "__main__":
    out = save_houses(generate_houses())
    print(f"Saved {len(load_houses(out))} rows to {out}")
