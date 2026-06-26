# Released under the GNU GPLv3; see LICENSE for details.
# Developed by Boucher Lab,
import os
import pandas as pd
import joblib
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from config import OUTPUT_PATH
from sklearn.preprocessing import StandardScaler
from models.predict import bootstrap_uncertainty

output_path = OUTPUT_PATH
MODEL_ORDER = ["RandomForest", "XGBoost", "Lasso", "Ridge"]
MODEL_ALIASES = {
    "rf": "RandomForest",
    "xb": "XGBoost",
    "ls": "Lasso",
    "rd": "Ridge",
}


def normalize_model_selection(model_selection):
    if model_selection is None or model_selection == "all":
        return MODEL_ORDER

    if isinstance(model_selection, str):
        selections = model_selection.split(",")
    else:
        selections = list(model_selection)

    selected_models = []
    for selection in selections:
        model_name = MODEL_ALIASES.get(str(selection).strip().lower(), str(selection).strip())
        if model_name not in MODEL_ORDER:
            raise ValueError(f"Unknown model selection: {selection}")
        if model_name not in selected_models:
            selected_models.append(model_name)

    return selected_models

def set_size(cfg):
    loci = cfg.numLoci
    sampleSize = cfg.sampleSize
    return loci, sampleSize

def load_training_data(train_path):
    with open(train_path, "r") as f:
        first = f.readline()
    sep = "\t" if "\t" in first else ","
    df = pd.read_csv(train_path, sep=sep, header=None)

    y_train = df.iloc[:, 0].astype(float).values
    X_train = df.iloc[:, 1:6].astype(float).values
    return X_train, y_train



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


def load_rf_model(rf_path, Z, train_path):
    if not os.path.exists(rf_path):
        raise FileNotFoundError(f"Model file {rf_path} does not exist.")

    rf_model = joblib.load(rf_path)

    X_train, y_train = load_training_data(train_path)
    # X_train_s = scaler.transform(X_train)
    # Z_scaled = scaler.transform(Z.values)

    rf_prediction = bootstrap_uncertainty(
        model=rf_model,
        X_train=X_train,
        y_train=y_train,
        X_point=Z,
        n_bootstrap=500,
        model_name="RandomForest"
    )
    return rf_prediction


def load_xgb_model(xgb_path, Z, train_path):
    if not os.path.exists(xgb_path):
        raise FileNotFoundError(f"Model file {xgb_path} does not exist.")

    X_train, y_train = load_training_data(train_path)
    # X_train_s = scaler.transform(X_train)
    # Z_scaled = scaler.transform(Z.values)

    xgb_model = joblib.load(xgb_path)
    xgb_prediction = bootstrap_uncertainty(
        model=xgb_model,
        X_train=X_train,
        y_train=y_train,
        X_point=Z,
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
        alpha=0.05,
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
        alpha=0.05,
        model_name="Ridge"
    )
    return ridge_prediction


def _run_inference_task(task):
    model_name = task["model_name"]
    model_path = task["model_path"]
    train_path = task["train_path"]
    scaler_path = task.get("scaler_path")
    Z = task["Z"]

    if model_name == "RandomForest":
        return load_rf_model(model_path, Z, train_path)
    elif model_name == "XGBoost":
        return load_xgb_model(model_path, Z, train_path)
    elif model_name == "Lasso":
        return load_lasso_model(model_path, Z, scaler_path, train_path)
    elif model_name == "Ridge":
        return load_ridge_model(model_path, Z, scaler_path, train_path)
    else:
        raise ValueError(f"Unknown inference model: {model_name}")


def run_all_models(cfg, Z, train_path, model_selection=None):
    selected_model_names = normalize_model_selection(model_selection)
    loci, sampleSize = set_size(cfg)
    folder_path = os.path.join(OUTPUT_PATH, f"{sampleSize}x{loci}")
    scaler_path = os.path.join(folder_path, f"scaler.joblib")
    rf_path     = os.path.join(folder_path, "RandomForest", f"rf_model.joblib")
    xgb_path    = os.path.join(folder_path, "XGBoost", f"xgb_model.joblib")
    lasso_path  = os.path.join(folder_path, "Lasso", f"lasso_model.joblib")
    ridge_path  = os.path.join(folder_path, "Ridge", f"ridge_model.joblib")
    results = []

    tasks = []
    task_map = {
        "RandomForest": {
            "model_name": "RandomForest",
            "model_path": rf_path,
            "train_path": train_path,
            "Z": Z,
        },
        "XGBoost": {
            "model_name": "XGBoost",
            "model_path": xgb_path,
            "train_path": train_path,
            "Z": Z,
        },
        "Lasso": {
            "model_name": "Lasso",
            "model_path": lasso_path,
            "train_path": train_path,
            "scaler_path": scaler_path,
            "Z": Z,
        },
        "Ridge": {
            "model_name": "Ridge",
            "model_path": ridge_path,
            "train_path": train_path,
            "scaler_path": scaler_path,
            "Z": Z,
        },
    }

    for model_name in selected_model_names:
        task = task_map[model_name]
        if os.path.exists(task["model_path"]):
            tasks.append(task)
        else:
            print(f"[Skip] {model_name} model not found at {task['model_path']}")

    if not tasks:
        print("No models available for inference.")
        return results

    max_workers = min(len(tasks), os.cpu_count() or 1)
    ctx = get_context("spawn")
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
        futures = {executor.submit(_run_inference_task, task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            model_name = task["model_name"]
            try:
                pred = future.result()
                results.append(pred)
                print(f"{pred['model']}: median={pred['median']:.4f}  95%CI=({pred['lower_95ci']:.4f},{pred['upper_95ci']:.4f})")
            except Exception as exc:
                print(f"[Skip] {model_name} failed: {exc}")

    return results
