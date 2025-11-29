# `model.py`: Poisson/Hawkes Temporal Features + L2 Logistic Regression

This module implements two things:

1. Poisson/Hawkes-inspired temporal features for cell–day crash counts  
2. An L2-regularized logistic regression model for predicting: “Will this cell have at least one crash tomorrow?”

---

## 1. How it works

We need,

1. a daily counts matrix  
   \( Y \in \mathbb{N}^{T \times C} \) where  
   - \(T\) = number of days  
   - \(C\) = number of grid cells  
   - \(Y[t, c]\) = # of crashes in cell `c` on day `t`.

then,

2. the model:
   - Computes Poisson/Hawkes-style temporal features from `Y`.
   - Builds a training set `(X, y)` where each row corresponds to a `(day t, cell c)` pair.
   - Trains a L2-regularized logistic regression that outputs  
     $$\(\hat p_{t,c} = \Pr(\text{crash in cell } c \text{ on day } t+1)\).$$

---

## 2. Contents of `model.py`

`model.py` provides:

- `compute_hawkes_features(Y, taus=(3, 7, 30))`  
  Computes exponential-decay “recency” features (Hawkes-like) at multiple time scales.

- `build_training_data_from_counts(Y, S_dict)`  
  Builds the feature matrix `X` and label vector `y` from daily counts and the Hawkes features.

- `LogisticConfig`  
  A small configuration dataclass for the logistic regression hyperparameters.

- `LogisticHotspot`  
  L2-regularized logistic regression with class weighting to handle imbalance.

---

## 3. Why use Poisson/Hawkes temporal features?

### 3.1 Intuition

Crashes are often self-exciting:
- A crash in a cell increases the chance of another crash in the near future (e.g., due to persistent risk factors).
- This extra risk decays over time.

This is the intuition behind a Hawkes process. We approximate that behaviour using exponential-decay features.

### 3.2 Discrete-time decayed counts

Given daily counts `Y[t, c]`, for each time scale $\(\tau\)$ (in days) we define:

$$
s^{(\tau)}_{t,c} = \rho \, s^{(\tau)}_{t-1,c} + Y[t-1, c],
\quad \rho = e^{-1/\tau}, \quad s^{(\tau)}_{0,c} = 0
$$

- $s^{(τ)}_{t, c}$ summarizes all past crashes in that cell, with an exponential decay:
  - recent days have more weight,
  - older events contribute less.
- Different $\(\tau\)$ values capture short, medium, and long memory.

### 3.3 Implementation

```python
from model import compute_hawkes_features

S_dict = compute_hawkes_features(Y, taus=(3, 7, 30))

