import os
import time
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split
import joblib
from sklearn.preprocessing import StandardScaler
from models.predict import bootstrap_uncertainty

def get_output_path():
    path = os.path.join(config.BASE_PATH, "output_100_4400/")
    os.makedirs(path, exist_ok=True)
    return path


def get_loci():
    if config.numLoci is None:
        raise ValueError("config.numLoci not set yet")
    return config.numLoci

def get_sample_size():
    if config.sampleSize is None:
        raise ValueError("config.sampleSize not set yet")
    return config.sampleSize

def get_plot_dir():
    loci = get_loci()
    sampleSize = get_sample_size()

    plot_dir = f"/blue/boucher/suhashidesilva/2025/Revision/ONeSAMP_ML/plots_100_4400/{sampleSize}x{loci}"
    os.makedirs(plot_dir, exist_ok=True)
    return plot_dir


FEATURES = [
    'Gametic_equilibrium',
    'Mlocus_homozegosity_mean',
    'Mlocus_homozegosity_variance',
    'Fix_index',
    'Emean_exhyt'
]

def load_training_data(train_path):
    with open(train_path, "r") as f:
        first = f.readline()
    sep = "\t" if "\t" in first else ","
    df = pd.read_csv(train_path, sep=sep, header=None)

    y_train = df.iloc[:, 0].astype(float).values
    X_train = df.iloc[:, 1:6].astype(float).values
    return X_train, y_train


output_path = "/blue/boucher/suhashidesilva/2025/Revision/ONeSAMP_ML/output_100_4400"
# Predict again using new Z
#Z_scaled = scaler.transform(Z)
#xgb_prediction = xgb_model.predict(Z_scaled)

#train_path = os.path.join(output_path,f'allPopStats_genePop{sampleSize}x{loci}')
#scaler_path = os.path.join(output_path, f'scaler_{sampleSize}x{loci}.joblib')
#rf_path = os.path.join(output_path, f"rf_model_{sampleSize}x{loci}.joblib")
#xgb_path = os.path.join(output_path, f"xgb_model_{sampleSize}x{loci}.joblib")
#lasso_path = os.path.join(output_path, f"lasso_model_{sampleSize}x{loci}.joblib")
#ridge_path = os.path.join(output_path, f"ridge_model_{sampleSize}x{loci}.joblib")



def load_scaler(scaler_path):
    if os.path.exists(scaler_path):
        return joblib.load(scaler_path)
    else:
        raise FileNotFoundError(f"Scaler file {scaler_path} does not exist.")


def load_rf_model(rf_path, Z, scaler_path, train_path):
    if not os.path.exists(rf_path):
        raise FileNotFoundError(f"Model file {rf_path} does not exist.")

    scaler = load_scaler(scaler_path)
    rf_model = joblib.load(rf_path)

    X_train, y_train = load_training_data(train_path)
    X_train_s = scaler.transform(X_train)
    Z_scaled = scaler.transform(Z.values)

    rf_prediction = bootstrap_uncertainty(
        model=rf_model,
        X_train=X_train_s,
        y_train=y_train,
        X_point=Z_scaled,
        n_bootstrap=500,
        model_name="RandomForest"
    )
    return rf_prediction


def load_xgb_model(xgb_path, Z, scaler_path, train_path):
    if not os.path.exists(xgb_path):
        raise FileNotFoundError(f"Model file {xgb_path} does not exist.")

    scaler = load_scaler(scaler_path)
    xgb_model = joblib.load(xgb_path)

    X_train, y_train = load_training_data(train_path)
    X_train_s = scaler.transform(X_train)
    Z_scaled = scaler.transform(Z.values)

    xgb_prediction = bootstrap_uncertainty(
        model=xgb_model,
        X_train=X_train_s,
        y_train=y_train,
        X_point=Z_scaled,
        n_bootstrap=500,
        model_name="XGBoost"
    )
    return xgb_prediction


def load_lasso_model(lasso_path, Z, scaler_path, train_path):
    if not os.path.exists(lasso_path):
        raise FileNotFoundError(f"Model file {lasso_path} does not exist.")

    scaler = load_scaler(scaler_path)
    lasso_model = joblib.load(lasso_path)

    X_train, y_train = load_training_data(train_path)
    X_train_s = scaler.transform(X_train)
    Z_scaled = scaler.transform(Z.values)

    lasso_prediction = bootstrap_uncertainty(
        model=lasso_model,
        X_train=X_train_s,
        y_train=y_train,
        X_point=Z_scaled,
        n_bootstrap=500,
        alpha=0.01,
        model_name="Lasso"
    )
    return lasso_prediction


def load_ridge_model(ridge_path, Z, scaler_path, train_path):
    if not os.path.exists(ridge_path):
        raise FileNotFoundError(f"Model file {ridge_path} does not exist.")

    scaler = load_scaler(scaler_path)
    ridge_model = joblib.load(ridge_path)

    X_train, y_train = load_training_data(train_path)
    X_train_s = scaler.transform(X_train)
    Z_scaled = scaler.transform(Z.values)

    ridge_prediction = bootstrap_uncertainty(
        model=ridge_model,
        X_train=X_train_s,
        y_train=y_train,
        X_point=Z_scaled,
        n_bootstrap=500,
        alpha=10,
        model_name="Ridge"
    )
    return ridge_prediction


def run_all_models(output_path, sampleSize, loci, Z, train_path):
    scaler_path = os.path.join(output_path, f"scaler_{sampleSize}x{loci}.joblib")
    rf_path     = os.path.join(output_path, f"rf_model_{sampleSize}x{loci}.joblib")
    xgb_path    = os.path.join(output_path, f"xgb_model_{sampleSize}x{loci}.joblib")
    lasso_path  = os.path.join(output_path, f"lasso_model_{sampleSize}x{loci}.joblib")
    ridge_path  = os.path.join(output_path, f"ridge_model_{sampleSize}x{loci}.joblib")

    results = []
    for fn, path in [
        (load_rf_model, rf_path),
        (load_xgb_model, xgb_path),
        (load_lasso_model, lasso_path),
        (load_ridge_model, ridge_path),
    ]:
        try:
            pred = fn(path, Z, scaler_path, train_path)
            results.append(pred)
            print(f"{pred['model']}: median={pred['median']:.4f}  95%CI=({pred['lower_95ci']:.4f},{pred['upper_95ci']:.4f})")
        except FileNotFoundError as e:
            print(f"[Skip] {e}")

    return results

