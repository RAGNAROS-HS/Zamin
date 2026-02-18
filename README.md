# Zamin

> *"Back to basics"* — Implementing regression algorithms from scratch with a pure, low-level approach, then comparing against neural network baselines.

## Overview

This project explores **housing price prediction** on the Boston Housing dataset using multiple approaches — from hand-coded gradient descent to deep learning — to understand the fundamentals of regression at every level of abstraction.

## Dataset

**Source:** [Boston Housing (Kaggle)](https://www.kaggle.com/datasets/altavish/boston-housing-dataset)

- **506 samples**, **13 features**, **1 target** (`MEDV` — Median home value in $1000s)
- **Train/Test Split:** 80/20 (random state 42)
- **Preprocessing:** Z-score standardisation (mean/std computed on train set only, applied to both)

### Features

| Feature   | Description                                                    |
| --------- | -------------------------------------------------------------- |
| `CRIM`    | Per capita crime rate by town                                  |
| `ZN`      | Proportion of residential land zoned for lots > 25k sqft       |
| `INDUS`   | Proportion of non-retail business acres per town               |
| `CHAS`    | Charles River dummy variable (1 if tract bounds river)         |
| `NOX`     | Nitric oxide concentration (parts per 10 million)              |
| `RM`      | Average number of rooms per dwelling                           |
| `AGE`     | Proportion of owner-occupied units built pre-1940              |
| `DIS`     | Weighted distances to five Boston employment centres           |
| `RAD`     | Index of accessibility to radial highways                      |
| `TAX`     | Full-value property tax rate per $10,000                       |
| `PTRATIO` | Pupil-teacher ratio by town                                    |
| `B`       | 1000(Bk − 0.63)² where Bk is the proportion of Black residents |
| `LSTAT`   | Percentage lower status of the population                      |
| **MEDV**  | **Median value of owner-occupied homes ($1000s) — TARGET**     |

---

## Dataset Analysis

### Feature Distributions

Histograms for all 14 variables in the dataset:

<p align="center">
  <img src="histograms/CRIM.png" width="24%"> <img src="histograms/ZN.png" width="24%"> <img src="histograms/INDUS.png" width="24%"> <img src="histograms/CHAS.png" width="24%">
</p>
<p align="center">
  <img src="histograms/NOX.png" width="24%"> <img src="histograms/RM.png" width="24%"> <img src="histograms/AGE.png" width="24%"> <img src="histograms/DIS.png" width="24%">
</p>
<p align="center">
  <img src="histograms/RAD.png" width="24%"> <img src="histograms/TAX.png" width="24%"> <img src="histograms/PTRATIO.png" width="24%"> <img src="histograms/B.png" width="24%">
</p>
<p align="center">
  <img src="histograms/LSTAT.png" width="24%"> <img src="histograms/MEDV.png" width="24%">
</p>

### Correlation Analysis

|                                                   |                                                         |
| :-----------------------------------------------: | :-----------------------------------------------------: |
| ![Correlation Heatmap](heatMaps/corr_heatmap.png) | ![Correlation Clustermap](heatMaps/corr_clustermap.png) |
|          *Standard correlation heatmap*           |        *Hierarchical clustered correlation map*         |

**Key correlations with MEDV (target):**
- **Strong positive:** `RM` (rooms) — more rooms → higher price
- **Strong negative:** `LSTAT` (lower status %) — higher LSTAT → lower price
- **Notable negatives:** `PTRATIO`, `INDUS`, `TAX`, `CRIM`

---

## Methods Implemented

### 1. Gradient Descent — Linear Regression from Scratch (`rawRun.py`)

A **pure NumPy** implementation with no ML library abstractions:

- Manual weight initialisation (`w ~ N(0, 0.01)`, `b = 0`)
- MSE loss with analytical gradient computation
- Convergence tolerance check (`tol = 1e-4`)
- **Hyperparameters:** `α = 0.001`, `epochs = 1200`

```python
# Core update rule (no libraries)
dw = (2 / m) * X_train.T @ (y_pred - y_train)
db = (2 / m) * sum(y_pred - y_train)
w -= α * dw
b -= α * db
```

### 2. Neural Network — Keras/TensorFlow (`EvolutionaryAlgorithm.py`)

A **3-hidden-layer** dense network trained with Adam optimiser:

| Layer    | Units | Activation |
| -------- | ----- | ---------- |
| Input    | 13    | —          |
| Hidden 1 | 22    | ReLU       |
| Hidden 2 | 22    | ReLU       |
| Hidden 3 | 22    | ReLU       |
| Output   | 1     | Linear     |

- **Total parameters:** 1,343
- **Optimiser:** Adam
- **Loss:** MSE
- **Batch size:** 32, **Epochs:** 50 (with early stopping, patience=20)

### 3. Evolutionary Algorithm (Work in Progress)

Scaffolded in `EvolutionaryAlgorithm.py` — evolves neural network weights using:
- **Tournament selection** (k=3)
- **Gaussian mutation** (per-gene with configurable rate and sigma)
- **Uniform crossover** (per-gene swap with configurable rate)

> ⚠️ **Status:** Skeleton implemented — `fitness()`, `ea_training_loop()`, and `mutation()` functions are stubbed and awaiting completion.

---

## Results Comparison

### Metrics

| Metric   | Gradient Descent (Linear) | Neural Network (Keras) |
| -------- | :-----------------------: | :--------------------: |
| **MSE**  |          32.406           |         12.957         |
| **RMSE** |           5.692           |         3.599          |
| **MAE**  |           3.378           |         2.399          |
| **R²**   |          0.558            |         0.823          |



### Training Loss Curves

|                                                      |                                                                  |
| :--------------------------------------------------: | :--------------------------------------------------------------: |
|               ![GD Loss](gd_loss.png)                |                     ![NN Loss](nn_loss.png)                      |
| *Gradient descent MSE convergence over ~1200 epochs* | *NN train/validation loss over ~50 epochs (with early stopping)* |

### Test Set Predictions — Actual vs Predicted

|                                 |                                    |
| :-----------------------------: | :--------------------------------: |
| ![GD Scatter](test_scatter.png) | ![NN Scatter](nn_test_scatter.png) |
| *Gradient descent predictions*  |    *Neural network predictions*    |

Points closer to the red dashed line (y = x) indicate better predictions. The neural network shows tighter clustering around the ideal line, especially in the mid-range values.

---

## Project Structure

```
Zamin/
├── rawRun.py                 # Gradient descent linear regression (from scratch)
├── EvolutionaryAlgorithm.py  # Neural network + evolutionary algorithm (WIP)
├── dataPreProcess.py         # Dataset visualisation & correlation analysis
├── housing.csv               # Boston Housing dataset (506 × 14)
├── stickyNote.txt            # Project notes & core concepts
├── histograms/               # Feature distribution plots (all 14 variables)
├── heatMaps/                 # Correlation heatmap & clustermap
├── boxPlots/                 # (Reserved for future box plot analysis)
├── gd_loss.png               # GD training loss curve
├── gd_loss_convergence.png   # GD loss convergence detail
├── nn_loss.png               # NN train/val loss curve
├── test_scatter.png          # GD actual vs predicted scatter
├── nn_test_scatter.png       # NN actual vs predicted scatter
├── nnmod1.keras              # Saved Keras model
└── nnmod1.h5                 # Saved model (HDF5 format)
```

---

## Key Takeaways

1. **Linear regression via GD** achieves R² ≈ 0.67 — a solid baseline given the simplicity
2. **The neural network** nearly doubles the explained variance (R² ≈ 0.83), capturing non-linear feature interactions
3. **Z-score normalisation** is essential for gradient descent convergence with features on vastly different scales
4. The largest predictors of house price are **RM** (rooms, +) and **LSTAT** (lower status %, −)

## Future Work

- [ ] Complete the evolutionary algorithm for neuroevolution
- [ ] Add OLS and Ridge regression comparison (as noted in project plan)
- [ ] Feature engineering (polynomial features, interaction terms)
- [ ] Cross-validation for more robust metric estimates
- [ ] Hyperparameter tuning (learning rate schedules, network architecture search)
