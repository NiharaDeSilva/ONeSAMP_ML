
# Model Training and Evaluation Modules

This directory contains modular Python scripts for training, evaluating, and loading machine learning models used in the project. It supports multiple regression models and is organized for reusability and clarity.

---

## File Overview

### `train.py`
Handles **end-to-end training of machine learning models**, including:
- Scaling input data with `StandardScaler`
- Fitting the model to training data
- Saving trained models (if applicable)

**Supported models:**
- Random Forest
- XGBoost (standard and quantile-based)
- Lasso Regression
- Ridge Regression

Each function in this module performs internal scaling and model training and returns the trained model for evaluation or reuse.

---

### `predict.py`
Responsible for:
- Making predictions on new input points (`Z`)
- Obtain Confidence interval for the prediction of new input point
- Evaluating models on test data using:
  - **R² (coefficient of determination)**
  - **RMSE (root mean squared error)**
  - **MAE (mean absolute error)**
- Computing and reporting **feature importance** (for interpretable models)

This module assumes models are already trained and optionally loaded using `model_utils`.

---

### `model_utils.py`
Utility functions for **model loading**, used to:
- Load pre-trained models from disk using `joblib`
- Optionally manage model directories and paths

This module does **not** perform training or prediction — it only manages model retrieval for reuse in evaluation or inference workflows.

---

## Workflow Summary

1. Use `train.py` to train and scale models from raw input data.
2. Use `predict.py` to:
   - Evaluate models on test data
   - Predict on new input points
   - Analyze model performance and interpretability
3. Use `model_utils.py` to load previously saved models when retraining is not necessary.



