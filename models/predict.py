# Prediction using trained models
import numpy as np
import joblib
import os
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.utils import resample

feature_names = ['Gametic_equilibrium', 'Mlocus_homozegosity_mean', 'Mlocus_homozegosity_variance', 'Fix_index', 'Emean_exhyt']

# -----------------------------------
# Random Forest Regression
# -----------------------------------    

def predict_random_forest(model, X_predict):
    # Confidence Interval Prediction on X_predict
    tree_preds = np.array([tree.predict(X_predict) for tree in model.estimators_])
    median_pred = np.median(tree_preds)
    mean_pred = np.mean(tree_preds)
    min_pred = np.min(tree_preds)
    max_pred = np.max(tree_preds)
    lower_ci = np.percentile(tree_preds, 2.5)
    upper_ci = np.percentile(tree_preds, 97.5)

    predictions = {
        "mean": mean_pred,
        "median": median_pred,
        "min": min_pred,
        "max": max_pred,
        "lower_95ci": lower_ci,
        "upper_95ci": upper_ci
    }
    return predictions


def evaluate_random_forest(model, X_test, y_test):
    # Evaluation metrics
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    metrics = {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2
    }
    return metrics


def get_feature_importance(rf_model, feature_names):
    importances = rf_model.feature_importances_
    feature_importances = [(feature, round(score, 2)) for feature, score in zip(feature_names, importances)]
    feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)
    print("\nFeature importance")
    [print('Variable: {:30} : {}'.format(*pair)) for pair in feature_importances]
    return feature_importances


# -----------------------------------
# XGBoost
# -----------------------------------

def predict_xgboost(model, Z_scaled, model_lower, model_upper):

    booster = model.get_booster()
    num_rounds = booster.num_boosted_rounds()
    tree_preds = np.array([
        model.predict(Z_scaled, iteration_range=(i, i + 1))
        for i in range(num_rounds)
    ]).reshape(-1)

    mean_pred = np.mean(tree_preds)
    median_pred = np.median(tree_preds)

    predictions = {
        "mean": mean_pred.item(),
        "median": median_pred.item()
    }

    # Add quantile predictions if provided
    if model_lower is not None and model_upper is not None:
        lower = model_lower.predict(Z_scaled)
        upper = model_upper.predict(Z_scaled)
        predictions["lower_95ci"] = lower.item()
        predictions["upper_95ci"] = upper.item()

    return predictions



def evaluate_xgboost(model, X_test, y_test):
    # Predict on test set
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
    return metrics

# -----------------------------------
# Lasso and Ridge Models
# -----------------------------------


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return rmse, mae, r2


# ---- Bootstrap-Based Prediction Statistics ---- #
def predict_with_stats(model, X_train, y_train, new_sample, n_bootstrap=1000, alpha=0.05):
    predictions = []
    for _ in range(n_bootstrap):
        X_boot, y_boot = resample(X_train, y_train)
        model.fit(X_boot, y_boot)
        pred = model.predict(new_sample)[0]
        predictions.append(pred)
    predictions = np.array(predictions)
    lower = np.percentile(predictions, 100 * (alpha / 2))
    upper = np.percentile(predictions, 100 * (1 - alpha / 2))
    return {
        'mean': np.mean(predictions),
        'median': np.median(predictions),
        'min': np.min(predictions),
        'max': np.max(predictions),
        'lower_95ci': lower,
        'upper_95ci': upper
    }

def print_stats_inline(name, stats):
    print(f"{name} => Mean: {stats['mean']:.4f}, Median: {stats['median']:.4f}, "
          f"95% CI: ({stats['lower_95ci']:.4f}, {stats['upper_95ci']:.4f})")


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

def predict_and_evaluate_rf(model, X_test, y_test, Z_scaled):
    predictions = predict_random_forest(model, Z_scaled)
    metrics = evaluate_random_forest(model, X_test, y_test)
    print(f"RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}, R2: {metrics['r2']:.4f}")
    print_stats_inline("RF Prediction Stats", predictions)
    print("\nFeature importance")
    get_feature_importance(model, feature_names)


def predict_and_evaluate_xgb(model, X_test, y_test, Z_scaled, model_lower, model_upper):
    predictions = predict_xgboost(model, Z_scaled, model_lower, model_upper)
    metrics = evaluate_xgboost(model, X_test, y_test)
    print(f"RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}, R2: {metrics['r2']:.4f}")
    print_stats_inline("XGB Prediction Stats", predictions)
    get_feature_importance(model, feature_names)

    if model_lower is not None and model_upper is not None:
        print(f"Lower 95% CI: {predictions['lower_95ci']:.4f}, Upper 95% CI: {predictions['upper_95ci']:.4f}")
    else:
        print("No quantile models provided for confidence intervals.")


def predict_and_evaluate_lasso(model, X_test, y_test, Z_scaled):
    rmse, mae, r2 = evaluate_model(model, X_test, y_test)
    print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
    stats = predict_with_stats(model, X_test, y_test, Z_scaled)
    print_stats_inline("Lasso Prediction Stats", stats)
    get_coeficients_reg_models(model.coef_, feature_names)


def predict_and_evaluate_ridge(model, X_test, y_test, Z_scaled):
    rmse, mae, r2 = evaluate_model(model, X_test, y_test)
    print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
    stats = predict_with_stats(model, X_test, y_test, Z_scaled)
    print_stats_inline("Ridge Prediction Stats", stats)
    get_coeficients_reg_models(model.coef_, feature_names)




