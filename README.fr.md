# House Price Predictor (résumé en français)

Prédire le prix de maisons avec une **régression linéaire codée à la main** (NumPy uniquement), pour apprendre : vecteurs et matrices, probabilités et statistiques, dérivées, régression linéaire et descente de gradient.

## Structure du projet

```
house-price-predictor/
├── data/houses.csv            # 500 maisons synthétiques
├── notebooks/                 # 01 → 04, à suivre dans l'ordre
├── src/
│   ├── data.py                # génération et chargement des données
│   ├── preprocessing.py       # split train/test, standardisation, colonne de biais
│   ├── model.py               # perte, gradient, LinearRegression, métriques
│   └── train.py               # script d'entraînement complet
├── requirements.txt
├── README.md                  # version anglaise
├── README.fr.md               # ce fichier
└── .gitignore
```

## Installation et utilisation

```bash
python -m venv .venv
.venv\Scripts\activate           # Linux/Mac : source .venv/bin/activate
pip install -r requirements.txt

python -m src.data               # (re)générer data/houses.csv
python -m src.train              # équation normale, descente de gradient, SGD
python -m src.train --l2 0.1                          # régression Ridge
python -m src.train --dataset california --l2 0.01    # données réelles (internet requis une fois)
jupyter notebook                 # ouvrir notebooks/ dans l'ordre
```

Lancer les commandes depuis la racine du projet.

## Comment ça marche

Le modèle est `prix = w0 + w1·surface + w2·chambres + w3·âge`. Entraîner le modèle, c'est trouver les poids `w` qui minimisent l'erreur.

- **Données** : les prix sont générés avec des poids connus (50000, 900, 4000, -600) plus un bruit gaussien de 15 000. On peut donc vérifier que le modèle retrouve les bons poids.
- **Prétraitement** : 80 % des données pour l'entraînement, 20 % pour le test. La standardisation `z = (x − moyenne) / écart-type` est calculée sur l'entraînement seulement, pour éviter les fuites de données.
- **Perte (MSE)** : `J(w) = moyenne((Xw − y)²)`.
- **Gradient** : `∇J = (2/m) · Xᵀ(Xw − y)`, obtenu avec la règle de dérivation en chaîne et vérifié par une dérivée numérique.
- **Équation normale** : `XᵀX w = Xᵀy`, la solution exacte.
- **Descente de gradient** : `w ← w − lr · ∇J`, la solution par étapes successives.

## Les 4 notebooks

| Notebook | Contenu |
|---|---|
| 01 exploration | Forme de `X` et `y`, produit scalaire, `X @ w`, graphiques |
| 02 statistiques | Moyenne, variance, corrélation, loi normale, standardisation |
| 03 régression linéaire | Pente = cov/var, équation normale, résidus |
| 04 descente de gradient | Dérivées, gradient du MSE, taux d'apprentissage, SGD, Ridge, données réelles |

## Résultats

- L'équation normale et la descente de gradient donnent les mêmes poids, à environ 1e-16 près.
- R² ≈ 0,80 et RMSE ≈ 15 200, soit à peu près le bruit injecté (15 000) : le modèle est presque aussi bon que possible.
- Un taux d'apprentissage trop grand fait diverger la perte, et sans standardisation la descente de gradient échoue.

## Extensions incluses

1. **SGD / mini-batch** : `fit_sgd(batch_size=...)`. Avec `batch_size=len(y)`, on retrouve exactement la descente classique.
2. **Ridge** : l'argument `l2=` ajoute `+ λ‖w‖²` à la perte et `+ 2λw` au gradient, sans pénaliser le biais. Il aide surtout quand les données sont rares : sur 12 maisons, le RMSE test passe de 17 555 à 17 304 avec λ = 0,1.
3. **Données réelles** : `load_california()` charge California Housing (scikit-learn, internet requis une fois).

## Correspondance avec les objectifs d'apprentissage

| Sujet | Où |
|---|---|
| Vecteurs et matrices | `notebooks/01_exploration.ipynb`, `X @ w` dans `src/model.py` |
| Probabilités et statistiques | `notebooks/02_statistics.ipynb`, `src/preprocessing.py` |
| Dérivées et calcul | `notebooks/04_gradient_descent.ipynb`, `gradient` et `numerical_gradient` |
| Régression linéaire | `notebooks/03_linear_regression.ipynb`, `fit_normal` |
| Descente de gradient | `notebooks/04_gradient_descent.ipynb`, `fit_gd` et `fit_sgd` |
