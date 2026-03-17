import os
import time
import json
import uuid
import joblib
import numpy as np

from datetime import datetime
from contextlib import redirect_stdout, redirect_stderr
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context

from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.metrics import mean_squared_error, mean_absolute_error, make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from models.predict import predict_and_evaluate_model
from models.calibration import calibration_curves
from config import OUTPUT_PATH, PLOT_DIR

output_path = OUTPUT_PATH


# -----------------------------------
# Basic path helpers
# -----------------------------------

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def set_size(cfg):
    loci = cfg.numLoci
    sampleSize = cfg.sampleSize
    return loci, sampleSize


def make_run_dir(cfg):
    loci, sampleSize = set_size(cfg)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    return ensure_dir(os.path.join(output_path, f"{sampleSize}x{loci}", run_id))


def get_model_dir(run_dir, model_name):
    return ensure_dir(os.path.join(run_dir, model_name))

def get_lgbm_path(cfg):
    loci, sampleSize = set_size(cfg)
    return os.path.join(output_path, f"lgbm_model_{sampleSize}x{loci}.joblib")

def get_catboost_path(cfg):
    loci, sampleSize = set_size(cfg)
    return os.path.join(output_path, f"catboost_model_{sampleSize}x{loci}.joblib")

def get_model_filename(model_name):
    mapping = {
        "RandomForest": "rf_model.joblib",
        "XGBoost": "xgb_model.joblib",
        "Lasso": "lasso_model.joblib",
        "Ridge": "ridge_model.joblib",
        "LightGBM": "lgbm_model.joblib",
        "CatBoost": "catboost_model.joblib",
    }
    return mapping[model_name]




def get_plot_dir(cfg, run_dir=None, model_name=None):
    if run_dir is not None and model_name is not None:
        return get_model_dir(run_dir, model_name)

    loci, sampleSize = set_size(cfg)
    return ensure_dir(os.path.join(PLOT_DIR, f"{sampleSize}x{loci}"))


# -----------------------------------
# OOF calibration data
# -----------------------------------

def get_oof_predictions(model_constructor, X_train, y_train, model_name, save_dir):
    """
    Performs 5-fold CV on the training set and saves calibration outputs
    into a model-specific directory.
    """
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    y_train = np.asarray(y_train).ravel()
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

    calibration_curves(
        true=oof_true,
        pred=oof_pred,
        model_name=f"{model_name}_OOF",
        save_dir=save_dir,
        folds=oof_folds,
    )

    return oof_pred, oof_true, oof_folds


# -----------------------------------
# CV results
# -----------------------------------

def rmse_scorer(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def get_model_builders():
    return {
        "Lasso": lambda: Lasso(alpha=0.003, max_iter=10000),
        "Ridge": lambda: Ridge(alpha=10),
        "RandomForest": lambda: RandomForestRegressor(
            n_estimators=1000,
            max_depth=20,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='log2',
            bootstrap=True,
            random_state=42,
            n_jobs=1,   # keep inner parallelism off
        ),
        "XGBoost": lambda: XGBRegressor(
            objective='reg:squarederror',
            n_estimators=1000,
            learning_rate=0.04,
            max_depth=11,
            min_child_weight=1,
            subsample=0.8,
            colsample_bytree=0.9,
            reg_alpha=6.97,
            reg_lambda=0.03,
            gamma=4,
            max_delta_step=9,
            random_state=42,
            n_jobs=1,   # keep inner parallelism off
        ),
        "LightGBM": lambda: LGBMRegressor(
            objective="regression",
            n_estimators=1000,
            learning_rate=0.04,
            num_leaves=31,
            max_depth=-1,
            subsample=0.8,
            colsample_bytree=0.9,
            reg_alpha=0.0,
            reg_lambda=0.0,
            random_state=42,
            n_jobs=1,
            verbose=-1
        ),

        "CatBoost": lambda: CatBoostRegressor(
            loss_function="RMSE",
            iterations=1000,
            learning_rate=0.04,
            depth=8,
            random_seed=42,
            thread_count=1,
            verbose=False,
            allow_writing_files=False
        )
    }


def build_cv_model(model_name, model_constructor):
    if model_name in ["Lasso", "Ridge"]:
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", model_constructor())
        ])
    return model_constructor()


def get_cv_results_all_models(X_train, y_train, cv_folds=5):
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scoring = {
        "rmse": make_scorer(rmse_scorer, greater_is_better=False),
        "mae": make_scorer(mean_absolute_error, greater_is_better=False),
        "r2": "r2",
    }

    results = {}
    model_builders = get_model_builders()

    for name, constructor in model_builders.items():
        model = build_cv_model(name, constructor)
        cv = cross_validate(model, X_train, y_train, cv=kf, scoring=scoring, n_jobs=1)

        rmse = -cv["test_rmse"]
        mae = -cv["test_mae"]
        r2 = cv["test_r2"]

        results[name] = {
            "rmse_mean": float(rmse.mean()),
            "rmse_std": float(rmse.std()),
            "mae_mean": float(mae.mean()),
            "mae_std": float(mae.std()),
            "r2_mean": float(r2.mean()),
            "r2_std": float(r2.std()),
        }

        print(
            f"{name} ({cv_folds}-Fold CV) => "
            f"RMSE: {rmse.mean():.4f} ± {rmse.std():.4f}, "
            f"MAE: {mae.mean():.4f} ± {mae.std():.4f}, "
            f"R²: {r2.mean():.4f} ± {r2.std():.4f}"
        )

    return results


def train_model(cfg, model_name, X_train, y_train, model_path, plot_dir):
    model_builders = get_model_builders()
    model_constructor = model_builders[model_name]

    get_oof_predictions(cfg, model_constructor, X_train, y_train, model_name, save_dir=plot_dir)

    model = model_constructor()
    model.fit(X_train, np.asarray(y_train).ravel())
    joblib.dump(model, model_path)
    return model


# -----------------------------------
# Top-level worker for multiprocessing
# -----------------------------------

def _run_single_model_task(task):
    """
    Must stay at module scope for process-based parallelism.
    """
    cfg = task["cfg"]
    model_name = task["model_name"]
    X_train = task["X_train"]
    y_train = task["y_train"]
    X_test = task["X_test"]
    y_test = task["y_test"]
    Z = task["Z"]
    feature_cols = task["feature_cols"]
    model_dir = task["model_dir"]
    model_path = task["model_path"]

    log_path = os.path.join(model_dir, "train.log")

    with open(log_path, "w") as log_fp, redirect_stdout(log_fp), redirect_stderr(log_fp):
        print(f"\n------------- {model_name.upper()} -------------")
        start = time.time()

        model = train_model(
            cfg=cfg,
            model_name=model_name,
            X_train=X_train,
            y_train=y_train,
            model_path=model_path,
            plot_dir=model_dir,
        )

        # IMPORTANT:
        # This is safe only if predict_and_evaluate_model writes model-specific filenames
        # or accepts a save_dir internally.
        predict_and_evaluate_model(
            cfg, model, X_train, y_train, X_test, y_test, Z, model_name, feature_cols, model_dir)

        print(f"Time taken: {time.time() - start:.2f} seconds")

    return {
        "model_name": model_name,
        "model_path": model_path,
        "model_dir": model_dir,
        "log_path": log_path,
    }


# -----------------------------------
# Run Model Training
# -----------------------------------

def run_model_training(cfg, model_selection, allPopStatistics, inputStatsList):
    feature_cols = [
        'Gametic_equilibrium',
        'Mlocus_homozegosity_mean',
        'Mlocus_homozegosity_variance',
        'Fix_index',
        'Emean_exhyt'
    ]
    target_col = 'Ne'

    run_dir = make_run_dir(cfg)

    Z = np.array(inputStatsList[feature_cols].astype(float).to_numpy())

    df = allPopStatistics.copy()
    df = df[df[target_col].astype(float) > 0].copy()
    X = df[feature_cols].astype(float).to_numpy()
    y = df[target_col].astype(float).to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=40
    )

    cv_results = get_cv_results_all_models(X_train, y_train, cv_folds=5)
    with open(os.path.join(run_dir, "cv_results.json"), "w") as f:
        json.dump(cv_results, f, indent=2)

    # --- Scale only for Lasso / Ridge ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    Z_scaled = scaler.transform(Z)

    joblib.dump(scaler, os.path.join(run_dir, "scaler.joblib"))

    # --- Prepare task descriptions ---
    task_map = {
        0: {
            "cfg": cfg,
            "model_name": "RandomForest",
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
            "Z": Z,
            "feature_cols": feature_cols,
        },
        1: {
            "cfg": cfg,
            "model_name": "XGBoost",
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
            "Z": Z,
            "feature_cols": feature_cols,
        },
        2: {
            "cfg": cfg,
            "model_name": "Lasso",
            "X_train": X_train_scaled,
            "y_train": y_train,
            "X_test": X_test_scaled,
            "y_test": y_test,
            "Z": Z_scaled,
            "feature_cols": feature_cols,
        },
        3: {
            "cfg": cfg,
            "model_name": "Ridge",
            "X_train": X_train_scaled,
            "y_train": y_train,
            "X_test": X_test_scaled,
            "y_test": y_test,
            "Z": Z_scaled,
            "feature_cols": feature_cols,
        },
         4: {
            "cfg": cfg,
            "model_name": "LightGBM",
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
            "Z": Z,
            "feature_cols": feature_cols,
        },
        5: {
            "cfg": cfg,
            "model_name": "CatBoost",
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
            "Z": Z,
            "feature_cols": feature_cols,
        },
    }

    for idx, task in task_map.items():
        model_dir = get_model_dir(run_dir, task["model_name"])
        task["model_dir"] = model_dir
        task["model_path"] = os.path.join(model_dir, get_model_filename(task["model_name"]))

    # --- Single model path ---
    if model_selection in [0, 1, 2, 3, 4, 5]:
        result = _run_single_model_task(task_map[model_selection])
        model = joblib.load(result["model_path"])
        return model

    # --- Parallel path: train all 4 in separate processes ---
    max_workers = min(4, os.cpu_count() or 1)
    results = {}

    # spawn is the safest cross-platform start method
    ctx = get_context("spawn")

    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
        futures = {
            executor.submit(_run_single_model_task, task): task["model_name"]
            for task in task_map.values()
        }

        for future in as_completed(futures):
            result = future.result()
            results[result["model_name"]] = result

    rf_model = joblib.load(results["RandomForest"]["model_path"])
    xgb_model = joblib.load(results["XGBoost"]["model_path"])
    lasso_model = joblib.load(results["Lasso"]["model_path"])
    ridge_model = joblib.load(results["Ridge"]["model_path"])
    lgbm_model = joblib.load(results["LightGBM"]["model_path"])
    catboost_model = joblib.load(results["CatBoost"]["model_path"])

    return rf_model, xgb_model, lasso_model, ridge_model, lgbm_model, catboost_model
