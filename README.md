# House Price Predictor

🇫🇷 Version française : [README.fr.md](README.fr.md)

Predict house prices with **linear regression built from scratch** (NumPy only), to learn the maths behind machine learning.

## Learning goals

| Topic | Where |
|---|---|
| Vectors and matrices | `notebooks/01_exploration.ipynb`, `src/model.py` (`X @ w`) |
| Probability and statistics | `notebooks/02_statistics.ipynb`, `src/preprocessing.py` |
| Derivatives and calculus | `notebooks/04_gradient_descent.ipynb`, `src/model.py` (`gradient`, `numerical_gradient`) |
| Linear regression | `notebooks/03_linear_regression.ipynb`, `LinearRegression.fit_normal` |
| Gradient descent | `notebooks/04_gradient_descent.ipynb`, `LinearRegression.fit_gd` |

## Structure

```
house-price-predictor/
├── data/houses.csv            # synthetic dataset (500 houses)
├── notebooks/                 # follow in order 01 -> 04
├── src/
│   ├── data.py                # generate / load the dataset
│   ├── preprocessing.py       # split, standardization, bias column
│   ├── model.py               # loss, gradient, LinearRegression, metrics
│   └── train.py               # end-to-end training script
├── requirements.txt
└── .gitignore
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Run all commands from the project root.

```bash
python -m src.data     # (re)generate data/houses.csv
python -m src.train    # normal equation, gradient descent and mini-batch SGD
python -m src.train --l2 0.1                          # Ridge regression
python -m src.train --dataset california --l2 0.01    # real data (internet needed once)
jupyter notebook       # open notebooks/ and run them in order
```

Expected output of `python -m src.train`: both methods reach the same weights, with R² ≈ 0.80 and RMSE ≈ 15,000 (the noise std used to generate the data, so this is near the best possible score).

## The maths in one place

- Model: `y_hat = X w` (bias handled by a column of 1s)
- Loss: `J(w) = (1/m) * ||Xw - y||²`
- Gradient: `∇J = (2/m) * Xᵀ (Xw - y)`
- Normal equation: `XᵀX w = Xᵀy`
- Gradient descent: `w ← w - lr * ∇J(w)`

## Extensions included

- **Mini-batch / stochastic GD**: `LinearRegression.fit_sgd(batch_size=...)` (1 = stochastic, `len(y)` = full GD)
- **Ridge regularization**: `l2=` argument on `fit_normal`, `fit_gd`, `fit_sgd` (loss `+ λ‖w‖²`, gradient `+ 2λw`, bias not penalized)
- **Real dataset**: `load_california()` in `src/data.py` (needs `scikit-learn` and internet once)

Demos are in sections 7 to 9 of `notebooks/04_gradient_descent.ipynb`.

## Next steps

1. Learning-rate decay for SGD.
2. Pick `λ` with a validation set or cross-validation.
3. Add polynomial features to capture non-linear effects.
