
# Model Training and Inference Modules

This folder contains the core machine learning logic for training, saving, loading, and predicting with multiple models used in the project. 

---

## File Overview

### `train.py`
Responsible for training different machine learning models. Each model has its own dedicated training function with predefined hyperparameters for reproducibility.

**Supported models:**
- Random Forest (`train_random_forest`)
- XGBoost (`train_xgboost`)
- XGBoost Quantile Regressors (`train_xgboost_quantile`)
- Lasso Regression (`train_lasso`)
- Ridge Regression (`train_ridge`)

Each function returns a fitted model and optionally saves it using the `model_utils` functions.

---

### `predict.py`
Handles model evaluation and inference, including confidence intervals.

**Key features:**
- Evaluate models using common metrics: MSE, RMSE, MAE, R².
- Predict on new input (`Z`) using trained models.
- Estimate prediction confidence intervals:
  - For **Random Forest**: Aggregates predictions from all trees.
  - For **XGBoost**: Supports quantile regression (if lower/upper models provided).
  - For **Lasso Regression**:
  - For **Ridge Regression**:

---

### `model_utils.py`
Utility module for:
- Saving trained models using `joblib`
- Loading existing models for reuse
- Managing model file paths

---

## Usage

All modules are imported and called from `main.py`. You can choose a specific model or run all supported models using `run_model_training()` based on a selection input.

```python
from models.train import train_random_forest
from models.predict import predict_and_evaluate_rf
from models.model_utils import save_model, load_model
