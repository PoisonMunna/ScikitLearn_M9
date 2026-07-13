# 📙 Module 9: Model Selection & Hyperparameter Tuning — Learning Notes

> **Goal**: Build better-performing models by robustly estimating generalization error and systematically finding the optimal model configuration (hyperparameters) without overfitting to the validation set.

---

## 📌 Table of Contents

1. [Cross Validation](#1-cross-validation)
2. [K-Fold](#2-k-fold)
3. [Stratified K-Fold](#3-stratified-k-fold)
4. [Leave-One-Out](#4-leave-one-out)
5. [GridSearchCV](#5-gridsearchcv)
6. [RandomizedSearchCV](#6-randomizedsearchcv)
7. [HalvingGridSearchCV](#7-halvinggridsearchcv)
8. [Choosing Hyperparameters](#8-choosing-hyperparameters)
9. [Avoiding Overfitting](#9-avoiding-overfitting)

---

## 1. Cross Validation

### What is Cross Validation?
A resampling procedure used to evaluate machine learning models on a limited data sample. The primary goal is to estimate how the model is expected to perform on an unseen dataset (generalization error).

### Why Use It?
A single train/test split can be highly dependent on how the data is partitioned (luck of the draw). Cross-validation provides a more robust and less biased estimate of model performance by averaging scores across multiple splits.

### Code Example: Basic Cross-Validation

```python
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer

X, y = load_breast_cancer(return_X_y=True)

# Evaluate model using 5-fold cross-validation
model = RandomForestClassifier(random_state=42)
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

print(f"Individual CV Scores: {scores}")
print(f"Mean Accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
```

---

## 2. K-Fold

### What is K-Fold?
The most common cross-validation technique. The dataset is randomly partitioned into $K$ consecutive folds (usually $K=5$ or $10$). Each fold is used once as the validation set while the remaining $K-1$ folds form the training set.

### Key Points
- **Pros**: Every data point gets to be in a test set exactly once, and in a training set $K-1$ times.
- **Cons**: If the dataset is highly imbalanced, some folds might not contain any instances of the minority class.

### Code Example: Visualizing K-Fold Splits

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

X = np.arange(20)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

plt.figure(figsize=(10, 3))
for i, (train_index, test_index) in enumerate(kf.split(X)):
    # Visual representation of splits
    plt.scatter(train_index, [i]*len(train_index), c='blue', marker='s', label='Train' if i==0 else "")
    plt.scatter(test_index, [i]*len(test_index), c='red', marker='s', label='Test' if i==0 else "")

plt.yticks(np.arange(5), ['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5'])
plt.xlabel("Data Point Index")
plt.title("K-Fold Cross Validation Splits (K=5)")
plt.legend()
plt.grid(axis='x', alpha=0.3)
plt.show()
```

---

## 3. Stratified K-Fold

### What is Stratified K-Fold?
An extension of K-Fold that returns stratified folds: each set contains approximately the same percentage of samples of each target class as the complete set.

### When to Use It?
**Mandatory for classification tasks**, especially when dealing with **imbalanced datasets**. Standard K-Fold might result in folds with zero minority class samples, causing errors or wildly inaccurate fold scores.

### Code Example

```python
from sklearn.model_selection import StratifiedKFold

# Imbalanced dataset scenario
X_imb = np.zeros((100, 2))
y_imb = np.array([0]*90 + [1]*10) # 90% class 0, 10% class 1

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, test_idx) in enumerate(skf.split(X_imb, y_imb), 1):
    train_class_ratio = np.mean(y_imb[train_idx] == 1)
    test_class_ratio = np.mean(y_imb[test_idx] == 1)
    print(f"Fold {fold} -> Train Class 1 ratio: {train_class_ratio:.2f} | Test Class 1 ratio: {test_class_ratio:.2f}")
```

---

## 4. Leave-One-Out (LOOCV)

### What is Leave-One-Out?
An extreme case of cross-validation where $K = N$ (the total number of data points). In each iteration, exactly **one** data point is used as the validation set, and the remaining $N-1$ points are used for training.

### Pros & Cons
- **Pros**: Completely unbiased; uses almost all data for training in every iteration.
- **Cons**: Computationally expensive (requires training the model $N$ times). High variance in the validation error estimate because the training sets are nearly identical.

### Code Example

```python
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.linear_model import LogisticRegression

X_small = np.random.rand(50, 4)
y_small = (X_small[:, 0] > 0.5).astype(int)

loo = LeaveOneOut()
model = LogisticRegression()

# Note: LOOCV can be very slow for large datasets or complex models!
scores = cross_val_score(model, X_small, y_small, cv=loo, scoring='accuracy')
print(f"LOOCV Mean Accuracy: {scores.mean():.4f}")
```

---

## 5. GridSearchCV

### What is GridSearchCV?
An exhaustive searching technique over a manually specified parameter grid. It trains and evaluates a model for **every single combination** of hyperparameters provided.

### Key Points
- **Guarantees** finding the best combination *within the specified grid*.
- **Drawback**: Computationally explosive. If you have 3 parameters with 5 values each, that's $5 \times 5 \times 5 = 125$ models to train per fold.

### Code Example

```python
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Create a pipeline to prevent data leakage during scaling
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC())
])

# Define the parameter grid
param_grid = {
    'svm__C': [0.1, 1, 10],
    'svm__kernel': ['linear', 'rbf'],
    'svm__gamma': ['scale', 0.1, 0.01]
}

grid = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid.fit(X, y)

print(f"Best Parameters: {grid.best_params_}")
print(f"Best CV Score: {grid.best_score_:.4f}")
```

---

## 6. RandomizedSearchCV

### What is RandomizedSearchCV?
Instead of trying every combination, it samples a fixed number of parameter settings (`n_iter`) from specified statistical distributions. 

### Why Use It?
- **Efficiency**: Computation time is independent of the number of parameter values; it depends only on `n_iter`.
- **Effectiveness**: Often finds a near-optimal model much faster than GridSearch, especially when some hyperparameters don't heavily influence the outcome.

### Code Example

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform, randint

# Define distributions instead of fixed lists
param_distributions = {
    'svm__C': loguniform(1e-2, 1e2),      # Log-uniform distribution
    'svm__gamma': loguniform(1e-4, 1e0),  # Log-uniform distribution
    'svm__kernel': ['linear', 'rbf', 'poly']
}

rand_search = RandomizedSearchCV(
    pipe, param_distributions, n_iter=20, cv=5, 
    scoring='accuracy', random_state=42, n_jobs=-1
)
rand_search.fit(X, y)

print(f"Best Parameters (Random): {rand_search.best_params_}")
print(f"Best CV Score (Random): {rand_search.best_score_:.4f}")
```

---

## 7. HalvingGridSearchCV

### What is HalvingGridSearchCV?
An advanced technique using **Successive Halving**. It starts by evaluating all candidate parameter combinations with a small amount of data. It then keeps the top-performing candidates, increases the data size (usually doubling it), and re-evaluates them. This repeats until a final subset is evaluated on the full dataset.

### Why Use It?
Drastically reduces computation time for large grids by "killing off" bad hyperparameter configurations early before wasting time training them on the full dataset.

### Code Example

```python
from sklearn.experimental import enable_halving_search_cv # Required to import
from sklearn.model_selection import HalvingGridSearchCV

# Smaller grid for demonstration
param_grid_halving = {
    'svm__C': [0.1, 1, 10, 100],
    'svm__kernel': ['linear', 'rbf']
}

halving_grid = HalvingGridSearchCV(
    pipe, param_grid_halving, cv=5, factor=2, 
    scoring='accuracy', random_state=42, n_jobs=-1
)
halving_grid.fit(X, y)

print(f"Best Parameters (Halving): {halving_grid.best_params_}")
```

---

## 8. Choosing Hyperparameters

### Guidelines for Tuning
Not all hyperparameters are created equal. Focus your search space on parameters that control model complexity and learning dynamics.

### Common Hyperparameters by Model Family
- **Tree-based (Random Forest, XGBoost)**: `max_depth`, `min_samples_split`, `min_samples_leaf`, `n_estimators`, `learning_rate`.
- **Support Vector Machines (SVM)**: `C` (regularization), `kernel` (linear, rbf, poly), `gamma` (kernel coefficient).
- **Linear Models (Ridge, Lasso, Logistic)**: `alpha` or `C` (regularization strength), `penalty` (l1, l2).
- **Neural Networks**: `learning_rate`, `batch_size`, `hidden_layer_sizes`, `dropout_rate`.

### Pro-Tip: Coarse-to-Fine Search
1. Run a `RandomizedSearchCV` with wide, logarithmic ranges (e.g., $10^{-3}$ to $10^3$) to find the general "neighborhood" of the best parameters.
2. Run a `GridSearchCV` with a narrow, fine-grained grid centered around the best parameters found in step 1.

---

## 9. Avoiding Overfitting

### The Tuning Trap
Hyperparameter tuning itself can cause overfitting if you repeatedly evaluate models on the same validation set and optimize specifically for it (information leakage into the tuning process).

### Strategies to Prevent Overfitting During Tuning
1. **Nested Cross-Validation**: Use an outer CV loop for performance estimation and an inner CV loop (like `GridSearchCV`) for hyperparameter tuning.
2. **Hold-out Test Set**: Keep a completely untouched test set. Tune on the validation set, and only evaluate the final chosen model on the test set **once**.
3. **Regularization**: Always include regularization parameters (like `C`, `alpha`, or `max_depth`) in your search grid to penalize complexity.
4. **Early Stopping**: For iterative models (Gradient Boosting, Neural Nets), stop training when validation error stops improving, rather than tuning the maximum number of iterations blindly.

### Code Example: Nested Cross-Validation

```python
from sklearn.model_selection import cross_val_score, KFold

# Inner loop for tuning
inner_cv = KFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(SVC(), {'C': [0.1, 1, 10]}, cv=inner_cv)

# Outer loop for unbiased performance estimation
outer_cv = KFold(n_splits=5, shuffle=True, random_state=1)
nested_scores = cross_val_score(grid, X, y, cv=outer_cv, scoring='accuracy')

print(f"Nested CV Scores: {nested_scores}")
print(f"True Generalization Estimate: {nested_scores.mean():.4f}")
```

---

### Validation & Tuning Strategy Guide

| Scenario / Constraint | Recommended Strategy | Why? |
|-----------------------|----------------------|------|
| Standard classification with balanced classes | **Stratified K-Fold** | Ensures class representation in every fold. |
| Very small dataset (< 100 samples) | **LOOCV** | Maximizes training data size per iteration. |
| Small parameter grid, need exact best | **GridSearchCV** | Exhaustive; guarantees finding the best in the grid. |
| Large parameter grid, limited compute | **RandomizedSearchCV** | Samples efficiently; finds near-optimal quickly. |
| Massive dataset, massive parameter grid | **HalvingGridSearchCV** | Successive halving kills bad params early, saving compute. |
| Need an unbiased estimate of final model | **Nested Cross-Validation** | Prevents the tuning process from leaking into the final evaluation. |

---

### ✅ Module 9 Checklist
- [x] Understand the purpose of Cross-Validation in estimating generalization error
- [x] Implement K-Fold and visualize the data partitioning
- [x] Apply Stratified K-Fold to maintain class distributions in imbalanced datasets
- [x] Understand the trade-offs (bias/variance/compute) of Leave-One-Out CV
- [x] Execute GridSearchCV for exhaustive hyperparameter tuning
- [x] Execute RandomizedSearchCV using statistical distributions for efficient tuning
- [x] Implement HalvingGridSearchCV for successive halving on large search spaces
- [x] Identify which hyperparameters to tune for Trees, SVMs, and Linear models
- [x] Apply Nested Cross-Validation to prevent overfitting during the tuning process
- [x] Utilize a strict hold-out test set for final model evaluation
