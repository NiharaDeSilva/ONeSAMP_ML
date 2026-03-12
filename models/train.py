#Training Functions
import os
import time
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, KFold, cross_validate
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.utils import resample  
from statistics import statisticsClass
from sklearn.pipeline import Pipeline
from models.predict import predict_and_evaluate_rf, predict_and_evaluate_xgb, predict_and_evaluate_lasso, predict_and_evaluate_ridge
from models.calibration import calibration_curves
import config as cfg

loci = cfg.config.numLoci
sampleSize = cfg.config.sampleSize
output_path = cfg.config.OUTPUT_PATH

def get_output_path():
    path = os.path.join(cfg.config.BASE_PATH, "output_100_4400/")
    os.makedirs(path, exist_ok=True)
    return path

def get_loci():
    if cfg.config.numLoci is None:
        raise ValueError("cfg.config.numLoci not set yet")
    return cfg.config.numLoci

def get_sample_size():
    if cfg.config.sampleSize is None:
        raise ValueError("cfg.config.sampleSize not set yet")
    return cfg.config.sampleSize

def get_plot_dir():
    plot_dir = os.path.join(cfg.config.PLOT_DIR, f"{sampleSize}x{loci}")
    os.makedirs(plot_dir, exist_ok=True)
    return plot_dir

def get_rf_path():
    return os.path.join(output_path, f"rf_model_{sampleSize}x{loci}.joblib")

def get_xgb_path():
    return os.path.join(output_path, f"xgb_model_{sampleSize}x{loci}.joblib")

def get_lasso_path():
    return os.path.join(output_path, f"lasso_model_{sampleSize}x{loci}.joblib")

def get_ridge_path():
    return os.path.join(output_path, f"ridge_model_{sampleSize}x{loci}.joblib")

def get_train_data_paths():
    X_train_path = os.path.join(output_path, f"X_train_scaled_{sampleSize}x{loci}.joblib")
    y_train_path = os.path.join(output_path, f"y_train_{sampleSize}x{loci}.joblib")
    return X_train_path, y_train_path




# -----------------------------------
# Out of fold calibration data
# -----------------------------------
def get_oof_predictions(model_constructor, X_train, y_train, model_name):
    """
    Performs 5-fold CV on the training set.
    Produces:
        - OOF predictions (same length as training set)
        - Fold IDs for each point
        - Calibration curves saved to plot_dir
    """

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    oof_pred = np.zeros(len(y_train))
    oof_true = y_train.copy()
    oof_folds = np.zeros(len(y_train), dtype=int)

    for fold_id, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr = y_train[train_idx]

        model = model_constructor()
        model.fit(X_tr, y_tr)

        preds = model.predict(X_val)

        oof_pred[val_idx] = preds
        oof_folds[val_idx] = fold_id

    plot_dir = get_plot_dir()

  # Save OOF calibration curves
    calibration_curves(
        true=oof_true,
        pred=oof_pred,
        model_name=f"{model_name}_OOF",
        save_dir=plot_dir,
        folds=oof_folds,
    )

    return oof_pred, oof_true, oof_folds



# -----------------------------------
# CV results
# -----------------------------------


def rmse_scorer(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def get_cv_results_all_models(X_train, y_train, cv_folds=5):
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scoring = {
        "rmse": make_scorer(rmse_scorer, greater_is_better=False),
        "mae": make_scorer(mean_absolute_error, greater_is_better=False),
        "r2": "r2"
    }
    results = {}

    models = {
        "Lasso": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Lasso(alpha=0.01, max_iter=10000))
        ]),
        "Ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=10))
        ]),
        "RandomForest": RandomForestRegressor(
            n_estimators=5000,
            max_depth=40,
            min_samples_split=2,
            min_samples_leaf=2,
            max_features="log2",
            bootstrap=True,
            random_state=42,
            n_jobs=1
        ),
        "XGBoost": XGBRegressor(
            objective="reg:squarederror",
            n_estimators=800,
            learning_rate=0.01,
            max_depth=8,
            min_child_weight=3,
            subsample=0.6,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=1
        )
    }

    for name, model in models.items():
        cv = cross_validate(model, X_train, y_train, cv=kf, scoring=scoring, n_jobs=1)
        rmse = -cv["test_rmse"]
        mae = -cv["test_mae"]
        r2 = cv["test_r2"]
        results[name] = {
            "rmse_mean": rmse.mean(),
            "rmse_std": rmse.std(),
            "mae_mean": mae.mean(),
            "mae_std": mae.std(),
            "r2_mean": r2.mean(),
            "r2_std": r2.std(),
        }
        print(f"{name} ({cv_folds}-Fold CV) => "
              f"RMSE: {rmse.mean():.4f} ± {rmse.std():.4f}, "
              f"MAE: {mae.mean():.4f} ± {mae.std():.4f}, "
              f"R²: {r2.mean():.4f} ± {r2.std():.4f}")

    return results


# -----------------------------------
# Random Forest Regression
# -----------------------------------

"""
    Train a Random Forest regressor with fixed hyperparameters and save both model.

    Parameters:
    - X_train: numpy or pandas DataFrame of training features (scaled)
    - y_train: numpy array of training labels
    - model_path: filepath to save the trained model
"""
def train_random_forest(X_train, y_train_np, rf_path):
    def rf_constructor():
        return RandomForestRegressor(
            n_estimators=5000,
            max_depth=40,
            min_samples_split=2,
            min_samples_leaf=2,
            max_features='log2',
            bootstrap=True,
            random_state=42,
            n_jobs=-1
        )

    get_oof_predictions(rf_constructor, X_train, y_train_np.ravel(), "RandomForest")
    rf_model = rf_constructor()
    rf_model.fit(X_train, y_train_np.ravel())
    joblib.dump(rf_model, rf_path)
    return rf_model


# -----------------------------------
# XGBoost
# -----------------------------------

#xgb_path = os.path.join(output_path, f"xgb_model_{sampleSize}x{loci}.joblib")
# model_lower_path = os.path.join(output_path, f"xgb_model_lower_{sampleSize}x{loci}.joblib")
# model_upper_path = os.path.join(output_path, f"xgb_model_upper_{sampleSize}x{loci}.joblib")

def train_xgboost(X_train, y_train_np, xgb_path):

    def xgb_constructor():
        return XGBRegressor(
            objective='reg:squarederror',
            n_estimators=800,
            learning_rate=0.01,
            max_depth=8,
            min_child_weight=3,
            subsample=0.6,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
    get_oof_predictions(xgb_constructor, X_train, y_train_np.ravel(), "XGBoost")

    xgb_model = xgb_constructor()
    xgb_model.fit(X_train, y_train_np.ravel())
    joblib.dump(xgb_model, xgb_path)
    return xgb_model


# def train_xgboost_quantile(X_train_scaled, y_train_np, model_lower_path, model_upper_path, quantile_alpha):
#     base_params = {
#         'n_estimators': 2000,
#         'learning_rate': 0.01,
#         'max_depth': 8,
#         'min_child_weight': 5,
#         'subsample': 0.6,
#         'colsample_bytree': 0.8,
#         'random_state': 42,
#         'n_jobs': -1
#     }
#
#     model_lower = XGBRegressor(objective='reg:quantileerror', quantile_alpha=0.025, **base_params)
#     model_upper = XGBRegressor(objective='reg:quantileerror', quantile_alpha=0.975, **base_params)
#     model_lower.fit(X_train_scaled, y_train_np.ravel())
#     model_upper.fit(X_train_scaled, y_train_np.ravel())
#     joblib.dump(model_lower, model_lower_path)
#     joblib.dump(model_upper, model_upper_path)
#     return model_lower, model_upper



# -------------------------------
# Lasso & Ridge Regression
# -------------------------------

#X_train_path = os.path.join(output_path, f"X_train_scaled_{sampleSize}x{loci}.joblib")
#y_train_path = os.path.join(output_path, f"y_train_{sampleSize}x{loci}.joblib")

#ridge_path = os.path.join(output_path, f"ridge_model_{sampleSize}x{loci}.joblib")
#lasso_path = os.path.join(output_path, f"lasso_model_{sampleSize}x{loci}.joblib")

def train_lasso(X_train_scaled, y_train, lasso_path):
    def lasso_constructor():
        return Lasso(alpha=0.01, max_iter=10000)

    get_oof_predictions(lasso_constructor, X_train_scaled, y_train.ravel(), "Lasso")

    lasso_model = lasso_constructor()
    lasso_model.fit(X_train_scaled, y_train.ravel())
    joblib.dump(lasso_model, lasso_path)
    return lasso_model

def train_ridge(X_train_scaled, y_train, ridge_path):

    def ridge_constructor():
        return Ridge(alpha=10)

    get_oof_predictions(ridge_constructor, X_train_scaled, y_train.ravel(), "Ridge")
    ridge_model = ridge_constructor()
    ridge_model.fit(X_train_scaled, y_train.ravel())
    joblib.dump(ridge_model, ridge_path)
    return ridge_model



# -------------------------------
# Run Model Training
# -------------------------------

def run_model_training(model_selection, allPopStatistics, inputStatsList, loci, sampleSize):
    feature_cols = ['Gametic_equilibrium', 'Mlocus_homozegosity_mean', 'Mlocus_homozegosity_variance', 'Fix_index', 'Emean_exhyt']
    target_col = 'Ne'
    print(cfg.config.OUTPUT_PATH)

    Z = np.array(inputStatsList[feature_cols].astype(float).to_numpy())

    df = allPopStatistics.copy()
    df = df[df[target_col].astype(float) > 0].copy()
    X = df[feature_cols].astype(float).to_numpy()
    y = df[target_col].astype(float).to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)
    cv_results = get_cv_results_all_models(X_train, y_train, cv_folds=5)

    # --- Scale only for Lasso / Ridge ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    Z_scaled = scaler.transform(Z)
    
    # loci = get_loci()
    # sampleSize = get_sample_size()
    joblib.dump(scaler, os.path.join(output_path, f"scaler_{sampleSize}x{loci}.joblib"))

    # --- Define model runners ---
    def run_rf():
        print("\n------------- RANDOM FOREST -------------")
        start = time.time()
        rf_path = get_rf_path()
        model = train_random_forest(X_train, y_train, rf_path=rf_path)
        predict_and_evaluate_rf(model, X_train, y_train, X_test, y_test, Z)
        print(f"Time taken: {time.time() - start:.2f} seconds")
        return model

    def run_xgb():
        print("\n------------- XGBOOST ------------------")
        start = time.time()
        xgb_path = get_xgb_path()
        model = train_xgboost(X_train, y_train, xgb_path=xgb_path)
        predict_and_evaluate_xgb(model, X_train, y_train, X_test, y_test, Z)
        # model_lower, model_upper = train_xgboost_quantile(X_train_scaled, y_train, model_lower_path, model_upper_path, quantile_alpha=0.05)
        print(f"Time taken: {time.time() - start:.2f} seconds")
        return model

    def run_lasso():
        print("\n------------- LASSO REGRESSION ------------------")
        start = time.time()
        lasso_path = get_lasso_path()
        model = train_lasso(X_train_scaled, y_train, lasso_path=lasso_path)
        predict_and_evaluate_lasso(model, X_train_scaled, y_train, X_test_scaled, y_test, Z_scaled)
        print(f"Time taken: {time.time() - start:.2f} seconds")
        return model

    def run_ridge():
        print("\n------------- RIDGE REGRESSION ------------------")
        start = time.time()
        ridge_path = get_ridge_path()
        model = train_ridge(X_train_scaled, y_train, ridge_path=ridge_path)
        predict_and_evaluate_ridge(model, X_train_scaled, y_train, X_test_scaled, y_test, Z_scaled)
        print(f"Time taken: {time.time() - start:.2f} seconds")
        return model

    # --- Run selected models ---
    if model_selection == 0:
        rf_model = run_rf()
        return rf_model

    elif model_selection == 1:
        xgb_model = run_xgb()
        return xgb_model

    elif model_selection == 2:
        lasso_model = run_lasso()
        return lasso_model

    elif model_selection == 3:
        ridge_model = run_ridge()
        return ridge_model

    else:
        rf_model = run_rf()
        xgb_model = run_xgb()
        lasso_model = run_lasso()
        ridge_model = run_ridge()
        return rf_model, xgb_model, lasso_model, ridge_model
