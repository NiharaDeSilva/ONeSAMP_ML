#Training Functions
import os
import time
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.utils import resample  
from statistics import statisticsClass
import config
from models.predict import predict_and_evaluate_rf, predict_and_evaluate_xgb, predict_and_evaluate_lasso, predict_and_evaluate_ridge
from models.calibration import calibration_curves

inputFileStatistics = statisticsClass()
loci = config.numLoci
sampleSize = config.sampleSize

BASE_PATH = os.path.dirname(os.path.realpath(__file__))
#output_path = (f"/blue/boucher/suhashidesilva/Nihara/ONeSAMP_ML/output/genePop{sampleSize}x{loci}")
output_path = os.path.join(BASE_PATH, "./output/")
os.makedirs(output_path, exist_ok=True)
scalar_path = os.path.join(output_path, f"scaler_{sampleSize}x{loci}.joblib")
dir_name = f"{sampleSize}x{loci}"
plot_dir = os.path.join(BASE_PATH, f"./plots/{dir_name}")
os.makedirs(plot_dir, exist_ok=True)


# -----------------------------------
# Out of fold calibration data
# -----------------------------------
def get_oof_predictions(model_constructor, X_train_scaled, y_train, model_name):
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

    for fold_id, (train_idx, val_idx) in enumerate(kf.split(X_train_scaled)):
        X_tr, X_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
        y_tr = y_train[train_idx]

        model = model_constructor()
        model.fit(X_tr, y_tr)

        preds = model.predict(X_val)

        oof_pred[val_idx] = preds
        oof_folds[val_idx] = fold_id

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
# Random Forest Regression
# -----------------------------------

rf_path = os.path.join(output_path, f"rf_model_{sampleSize}x{loci}.joblib")

"""
    Train a Random Forest regressor with fixed hyperparameters and save both model and scaler.

    Parameters:
    - X_train: numpy or pandas DataFrame of training features (scaled)
    - y_train: numpy array of training labels
    - scaler: fitted scaler object used to scale X_train
    - model_path: filepath to save the trained model
    - scaler_path: filepath to save the scaler
"""
def train_random_forest(X_train_scaled, y_train_np, rf_path):
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

    get_oof_predictions(rf_constructor, X_train_scaled, y_train_np.ravel(), "RandomForest_oof")
    rf_model = rf_constructor()
    rf_model.fit(X_train_scaled, y_train_np.ravel())
    joblib.dump(rf_model, rf_path)
    return rf_model


# -----------------------------------
# XGBoost
# -----------------------------------

xgb_path = os.path.join(output_path, f"xgb_model_{sampleSize}x{loci}.joblib")
# model_lower_path = os.path.join(output_path, f"xgb_model_lower_{sampleSize}x{loci}.joblib")
# model_upper_path = os.path.join(output_path, f"xgb_model_upper_{sampleSize}x{loci}.joblib")

def train_xgboost(X_train_scaled, y_train_np, xgb_path):
    def xgb_constructor():
        return XGBRegressor(
            objective='reg:squarederror',
            n_estimators=2000,
            learning_rate=0.01,
            max_depth=8,
            min_child_weight=3,
            subsample=0.6,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
    get_oof_predictions(xgb_constructor, X_train_scaled, y_train_np.ravel(), "XGBoost_oof")

    xgb_model = xgb_constructor()
    xgb_model.fit(X_train_scaled, y_train_np.ravel())
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

X_train_path = os.path.join(output_path, f"X_train_scaled_{sampleSize}x{loci}.joblib")
y_train_path = os.path.join(output_path, f"y_train_{sampleSize}x{loci}.joblib")

ridge_path = os.path.join(output_path, f"ridge_model_{sampleSize}x{loci}.joblib")
lasso_path = os.path.join(output_path, f"lasso_model_{sampleSize}x{loci}.joblib")

feature_names = ['Gametic_equilibrium', 'Mlocus_homozegosity_mean', 'Mlocus_homozegosity_variance', 'Fix_index', 'Emean_exhyt']


def train_lasso(X_train_scaled, y_train, lasso_path):
    def lasso_constructor():
        return Lasso(alpha=0.01, max_iter=10000)

    get_oof_predictions(lasso_constructor, X_train_scaled, y_train.ravel(), "Lasso_oof")

    lasso_model = lasso_constructor()
    lasso_model.fit(X_train_scaled, y_train.ravel())
    joblib.dump(lasso_model, lasso_path)
    return lasso_model

def train_ridge(X_train_scaled, y_train, ridge_path):

    def ridge_constructor():
        return Ridge(alpha=10)

    get_oof_predictions(ridge_constructor, X_train_scaled, y_train.ravel(), "Ridge_oof")
    ridge_model = ridge_constructor()
    ridge_model.fit(X_train_scaled, y_train.ravel())
    joblib.dump(ridge_model, ridge_path)
    return ridge_model



# -------------------------------
# Run Model Training
# -------------------------------

def run_model_training(model_selection, allPopStatistics, inputStatsList):
    feature_cols = ['Gametic_equilibrium', 'Mlocus_homozegosity_mean', 'Mlocus_homozegosity_variance', 'Fix_index', 'Emean_exhyt']
    target_col = 'Ne'

    Z = np.array(inputStatsList[feature_cols].astype(float).to_numpy())
    X = np.array(allPopStatistics[feature_cols].astype(float).to_numpy())
    y = np.array([float(value) for value in allPopStatistics[target_col] if float(value) > 0])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)

    # --- Normalize ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    Z_scaled = scaler.transform(Z)

    joblib.dump(scaler, os.path.join(output_path, f"scaler_{sampleSize}x{loci}.joblib"))

    # --- Define model runners ---
    def run_rf():
        print("\n------------- RANDOM FOREST -------------")
        start = time.time()
        model = train_random_forest(X_train_scaled, y_train, rf_path=rf_path)
        predict_and_evaluate_rf(model, X_train_scaled, y_train, X_test_scaled, y_test, Z_scaled)
        print(f"Time taken: {time.time() - start:.2f} seconds")
        return model

    def run_xgb():
        print("\n------------- XGBOOST ------------------")
        start = time.time()
        model = train_xgboost(X_train_scaled, y_train, xgb_path=xgb_path)
        # model_lower, model_upper = train_xgboost_quantile(X_train_scaled, y_train, model_lower_path, model_upper_path, quantile_alpha=0.05)
        predict_and_evaluate_xgb(model, X_train_scaled, y_train, X_test_scaled, y_test, Z_scaled)
        print(f"Time taken: {time.time() - start:.2f} seconds")
        return model

    def run_lasso():
        print("\n------------- LASSO REGRESSION ------------------")
        start = time.time()
        model = train_lasso(X_train_scaled, y_train, lasso_path=lasso_path)
        predict_and_evaluate_lasso(model, X_train_scaled, y_train, X_test_scaled, y_test, Z_scaled)
        print(f"Time taken: {time.time() - start:.2f} seconds")
        return model

    def run_ridge():
        print("\n------------- RIDGE REGRESSION ------------------")
        start = time.time()
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
