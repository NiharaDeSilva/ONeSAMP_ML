# Prediction using trained models
import numpy as np
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.utils import resample
from models.calibration import calibration_curves
import copy
import config as cfg

loci = cfg.config.numLoci
sampleSize = cfg.config.sampleSize
output_path = cfg.config.OUTPUT_PATH
feature_names = ['Gametic_equilibrium', 'Mlocus_homozegosity_mean', 'Mlocus_homozegosity_variance', 'Fix_index', 'Emean_exhyt']


def get_plot_dir():
    if cfg.config.PLOT_DIR is None:
        raise ValueError("cfg.config.PLOT_DIR not set yet")
    plot_dir = os.path.join(cfg.config.PLOT_DIR, f"{sampleSize}x{loci}")
    os.makedirs(plot_dir, exist_ok=True)
    return plot_dir


def bootstrap_uncertainty(model, X_train, y_train, X_point, n_bootstrap=500, alpha=0.05, model_name=""):
    """
    General non-parametric bootstrap uncertainty estimator.
    Returns CI, mean, median, min, max, std.
    """

    preds = []

    for b in range(n_bootstrap):
        # Step 1: Resample training data WITH replacement
        X_boot, y_boot = resample(X_train, y_train)

        boot_model = copy.deepcopy(model)
        # Step 2: Retrain the model on the bootstrapped dataset
        # model.fit(X_boot, y_boot)
        boot_model.fit(X_boot, y_boot)

        # Step 3: Predict the same input point
        pred = boot_model.predict(X_point)[0]
        preds.append(pred)

    preds = np.array(preds)

    # Step 4: Compute uncertainty statistics
    lower = np.percentile(preds, 100 * (alpha / 2))
    upper = np.percentile(preds, 100 * (1 - alpha / 2))

    results = {
        "mean": np.mean(preds),
        "median": np.median(preds),
        "std": np.std(preds),
        "min": np.min(preds),
        "max": np.max(preds),
        "lower_95ci": lower,
        "upper_95ci": upper,
        "model": model_name
    }

    return results



def get_feature_importance(rf_model, feature_names):
    importances = rf_model.feature_importances_
    feature_importances = [(feature, round(score, 2)) for feature, score in zip(feature_names, importances)]
    feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)
    print("\nFeature importance")
    [print('Variable: {:30} : {}'.format(*pair)) for pair in feature_importances]
    return feature_importances


def get_coeficients_reg_models(model_coef, feature_names):
    for name, coef in sorted(zip(feature_names, model_coef), key=lambda x: abs(x[1]), reverse=True):
        print(f"{name}: {coef:.4f}")


#--------------------------------------- Cross Validation ------------------------------------#

def evaluate_cv(model, X, y, cv_folds=5):
    r2_scores = cross_val_score(model, X, y, cv=cv_folds, scoring='r2')
    rmse_scores = -cross_val_score(model, X, y, cv=cv_folds,
        scoring=make_scorer(lambda yt, yp: np.sqrt(mean_squared_error(yt, yp)), greater_is_better=False))
    mae_scores = -cross_val_score(model, X, y, cv=cv_folds,
        scoring=make_scorer(mean_absolute_error, greater_is_better=False))

    print(f"{model.__class__.__name__} ({cv_folds}-Fold CV) => R²: {r2_scores.mean():.4f} ± {r2_scores.std():.4f}, "
      f"RMSE: {rmse_scores.mean():.4f} ± {rmse_scores.std():.4f}, "
      f"MAE: {mae_scores.mean():.4f} ± {mae_scores.std():.4f}")


#---------------------------------------Predict & Evaluate -------------------------------------#
def evaluate_regression_model(model, X_test, y_test):
    y_pred_test = model.predict(X_test)
    plot_dir = get_plot_dir()
    calibration_curves(y_test, y_pred_test, "XGBoost", save_dir=plot_dir)

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
    return metrics


def print_stats_inline(name, stats):
    print(f"{name} => Mean: {stats['mean']:.4f}, Median: {stats['median']:.4f}, "
          f"95% CI: ({stats['lower_95ci']:.4f}, {stats['upper_95ci']:.4f})")


def predict_and_evaluate_model(model, X_train, y_train, X_test, y_test, Z, model_name, feature_names):
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
    metrics = evaluate_regression_model(model, X_test, y_test)
    print(f"RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}, R2: {metrics['r2']:.4f}")

    # Nonparametric bootstrap prediction statistics
    boot_stats = bootstrap_uncertainty(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_point=Z,
        n_bootstrap=500,
        model_name=model_name
    )

    print_stats_inline(f"{model_name} Prediction Stats", boot_stats)

    # Interpretation
    if model_name in ["RandomForest", "XGBoost"]:
        get_feature_importance(model, feature_names)

    elif model_name in ["Lasso", "Ridge"]:
        get_coeficients_reg_models(model.coef_, feature_names)

    return {
        "metrics": metrics,
        "bootstrap_stats": boot_stats
    }






# -----------------------------------
# XGBoost
# -----------------------------------

# def predict_xgboost(model, Z_scaled, model_lower, model_upper):
#
#     booster = model.get_booster()
#     num_rounds = booster.num_boosted_rounds()
#     tree_preds = np.array([
#         model.predict(Z_scaled, iteration_range=(i, i + 1))
#         for i in range(num_rounds)
#     ]).reshape(-1)
#
#     mean_pred = np.mean(tree_preds)
#     median_pred = np.median(tree_preds)
#
#     if model_lower is not None and model_upper is not None:
#         lower = model_lower.predict(Z_scaled)
#         upper = model_upper.predict(Z_scaled)
#
#         # Enforce order (in case quantiles flip)
#         lower, upper = np.minimum(lower, upper), np.maximum(lower, upper)
#
#         # Clip mean and median to stay within the bounds
#         mean_pred = np.clip(mean_pred, lower, upper)
#         median_pred = np.clip(median_pred, lower, upper)
#
#         predictions = {
#             "mean": mean_pred.item(),
#             "median": median_pred.item(),
#             "lower_95ci": lower.item(),
#             "upper_95ci": upper.item()
#         }
#
#     return predictions






# # ---- Bootstrap-Based Prediction Statistics ---- #
# def predict_with_stats(model, X_train, y_train, new_sample, n_bootstrap=1000, alpha=0.05):
#     predictions = []
#     for _ in range(n_bootstrap):
#         X_boot, y_boot = resample(X_train, y_train)
#         model.fit(X_boot, y_boot)
#         pred = model.predict(new_sample)[0]
#         predictions.append(pred)
#     predictions = np.array(predictions)
#     lower = np.percentile(predictions, 100 * (alpha / 2))
#     upper = np.percentile(predictions, 100 * (1 - alpha / 2))
#     return {
#         'mean': np.mean(predictions),
#         'median': np.median(predictions),
#         'min': np.min(predictions),
#         'max': np.max(predictions),
#         'lower_95ci': lower,
#         'upper_95ci': upper
#     }
