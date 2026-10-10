# Prediction using trained models
# Released under the GNU GPLv3; see LICENSE for details.
# Developed by Boucher Lab,
import numpy as np
import pandas as pd
import os
from time import perf_counter
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.utils import resample
from models.calibration import calibration_curves
import copy
from config import PLOT_DIR

feature_names = ['Gametic_equilibrium', 'Mlocus_homozegosity_mean', 'Mlocus_homozegosity_variance', 'Fix_index', 'Emean_exhyt']

def set_size(cfg):
    loci = cfg.numLoci
    sampleSize = cfg.sampleSize
    return loci, sampleSize

def get_plot_dir(cfg):
    loci,sampleSize = set_size(cfg)
    if PLOT_DIR is None:
        raise ValueError("cfg.config.PLOT_DIR not set yet")
    plot_dir = os.path.join(PLOT_DIR, f"{sampleSize}x{loci}")
    os.makedirs(plot_dir, exist_ok=True)
    return plot_dir

"""
    Convert X into a numeric 2D representation.Keeps DataFrame column order if feature_names is provided.
"""
def _ensure_numeric_2d(X, feature_names=None, dtype=np.float32):
    if isinstance(X, pd.DataFrame):
        if feature_names is not None:
            missing = [c for c in feature_names if c not in X.columns]
            if missing:
                raise ValueError(f"Missing required feature columns: {missing}")
            X = X.loc[:, feature_names]

        # Force numeric conversion
        X = X.apply(pd.to_numeric, errors="raise")
        return X.astype(dtype)

    if isinstance(X, pd.Series):
        X = pd.to_numeric(X, errors="raise").to_numpy(dtype=dtype)
    else:
        X = np.asarray(X, dtype=dtype)

    if X.ndim == 1:
        X = X.reshape(1, -1)

    return X


def _ensure_numeric_1d(y, dtype=np.float32):
    if isinstance(y, (pd.Series, pd.DataFrame)):
        y = np.asarray(pd.to_numeric(np.ravel(y), errors="raise"), dtype=dtype)
    else:
        y = np.asarray(y, dtype=dtype).ravel()
    return y


def bootstrap_uncertainty(
    model,
    X_train,
    y_train,
    X_point,
    n_bootstrap=500,
    alpha=0.05,
    model_name="",
    random_state=42,
):
    """
    General non-parametric bootstrap uncertainty estimator.
    for RF, XGBoost, Lasso, Ridge.
    """
    start = perf_counter()

    # Sanitize inputs once up front
    X_train_clean = _ensure_numeric_2d(X_train, feature_names=feature_names)
    y_train_clean = _ensure_numeric_1d(y_train)
    X_point_clean = _ensure_numeric_2d(X_point, feature_names=feature_names)

    if X_train_clean.shape[1] != X_point_clean.shape[1]:
        raise ValueError(
            f"Feature mismatch: X_train has {X_train_clean.shape[1]} columns "
            f"but X_point has {X_point_clean.shape[1]} columns."
        )

    preds = []
    # Each call has its own reproducible sequence, including when run in threads.
    rng = np.random.RandomState(random_state)

    for _ in range(n_bootstrap):
        X_boot, y_boot = resample(X_train_clean, y_train_clean, random_state=rng)

        boot_model = copy.deepcopy(model)
        boot_model.fit(X_boot, y_boot)

        pred = boot_model.predict(X_point_clean)[0]
        preds.append(float(pred))

    preds = np.asarray(preds, dtype=np.float64)

    lower = np.percentile(preds, 100 * (alpha / 2))
    upper = np.percentile(preds, 100 * (1 - alpha / 2))

    return {
        "mean": float(np.mean(preds)),
        "median": float(np.median(preds)),
        "std": float(np.std(preds)),
        "min": float(np.min(preds)),
        "max": float(np.max(preds)),
        "lower_95ci": float(lower),
        "upper_95ci": float(upper),
        "model": model_name,
        "bootstrap_seconds": perf_counter() - start,
    }


def get_feature_importance(model, feature_names):
    importances = model.feature_importances_
    feature_importances = [(feature, round(score, 2)) for feature, score in zip(feature_names, importances)]
    feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)
    print("\nFeature importance")
    [print('Variable: {:30} : {}'.format(*pair)) for pair in feature_importances]
    return feature_importances


def get_coeficients_reg_models(model_coef, feature_names):
    for name, coef in sorted(zip(feature_names, model_coef), key=lambda x: abs(x[1]), reverse=True):
        print(f"{name}: {coef:.4f}")


#---------------------------------------Predict & Evaluate -------------------------------------#
def evaluate_regression_model(cfg, model, X_test, y_test, model_dir, model_name, timings=None):
    if timings is None:
        timings = {}
    start = perf_counter()
    y_pred_test = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred_test)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred_test)
    r2 = r2_score(y_test, y_pred_test)

    metrics = {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2
    }
    timings["evaluation_seconds"] = perf_counter() - start
    start = perf_counter()
    calibration_curves(y_test, y_pred_test, model_name, save_dir=get_plot_dir(cfg))
    timings["calibration_seconds"] = perf_counter() - start

    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, "metrics.txt"), "w") as f:
        f.write(
            f"RMSE: {metrics['rmse']:.4f}\n"
            f"MAE: {metrics['mae']:.4f}\n"
            f"R2: {metrics['r2']:.4f}\n"
        )
    return metrics


def print_stats_inline(name, stats):
    print(f"{name} => Mean: {stats['mean']:.4f}, Median: {stats['median']:.4f}, "
          f"95% CI: ({stats['lower_95ci']:.4f}, {stats['upper_95ci']:.4f})")


def predict_and_evaluate_model(cfg, model, X_train, y_train, X_test, y_test, Z, model_name, feature_names, model_dir):
    """
    Common evaluation + bootstrap CI + interpretation for all regression models.

    Parameters
    ----------
    model : fitted sklearn/xgboost model
    X_train, y_train : training data used for nonparametric bootstrap
    X_test, y_test : test data for metrics
    Z : single input row (already raw or scaled as needed for the model)
    model_name : str
        One of: RandomForest, XGBoost, Lasso, Ridge
    feature_names : list[str]
    """

    # Metrics on held-out test set
    timings = {}
    metrics = evaluate_regression_model(cfg, model, X_test, y_test, model_dir, model_name, timings)

    # Nonparametric bootstrap prediction statistics
    boot_stats = bootstrap_uncertainty(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_point=Z,
        n_bootstrap=500,
        model_name=model_name
    )

    timings["bootstrap_seconds"] = boot_stats["bootstrap_seconds"]
    print_stats_inline(f"{model_name} Prediction Stats", boot_stats)

    # Interpretation
    if model_name in ["RandomForest", "XGBoost"]:
        get_feature_importance(model, feature_names)

    elif model_name in ["Lasso", "Ridge"]:
        get_coeficients_reg_models(model.coef_, feature_names)

    return {
        "metrics": metrics,
        "bootstrap_stats": boot_stats,
        "timings": timings,
    }
