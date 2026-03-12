import os
import json
import warnings
import numpy as np
import pandas as pd
import optuna

from sklearn.model_selection import KFold, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import joblib

warnings.filterwarnings("ignore")


def train_and_tune_models(
    allPopStatistics, input_text_list, input_sample_size, output_dir, n_splits=5,random_state=42):
    """
    Train and tune Ridge, Lasso, Random Forest, and XGBoost
    for predicting Ne from summary statistics + sampleSize.

    Parameters
    ----------
    allPopStats_path : str
        Path to training TSV file.
    input_text_list : list
        One input sample containing 5 summary statistics in this order:
        [Gametic_equilibrium,
         Mlocus_homozegosity_mean,
         Mlocus_homozegosity_variance,
         Fix_index,
         Emean_exhyt]
    input_sample_size : int or float
        sampleSize corresponding to the inference input row.
    output_dir : str
        Directory to save tuning results, predictions, and trained models.
    n_splits : int
        Number of CV folds.
    random_state : int
        Random seed.

    Returns
    -------
    dict
        Dictionary containing trained models and results dataframe.
    """

    os.makedirs(output_dir, exist_ok=True)

    # =========================================================
    # 1. LOAD TRAINING DATA
    # =========================================================
    feature_cols = [
        'Gametic_equilibrium',
        'Mlocus_homozegosity_mean',
        'Mlocus_homozegosity_variance',
        'Fix_index',
        'Emean_exhyt'
    ]

    target_col = 'Ne'

    X = allPopStatistics[feature_cols].copy()
    y = allPopStatistics[target_col].copy()

    # =========================================================
    # 2. BUILD INFERENCE INPUT ROW
    # =========================================================
    inputStatsList = pd.DataFrame(
        [[
            input_text_list[0],
            input_text_list[1],
            input_text_list[2],
            input_text_list[3],
            input_text_list[4]
        ]],
        columns=feature_cols
    )
    inputStatsList = inputStatsList.apply(pd.to_numeric, errors='raise')

    # =========================================================
    # 3. CV SETUP
    # =========================================================
    effective_splits = min(n_splits, len(X))
    if effective_splits < 2:
        raise ValueError("Need at least 2 training rows for cross-validation.")

    cv = KFold(n_splits=effective_splits, shuffle=True, random_state=random_state)

    # =========================================================
    # 4. RIDGE
    # =========================================================
    ridge_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', Ridge())
    ])

    ridge_param_grid = {
        'model__alpha': np.logspace(-4, 4, 30)
    }
    print("starting ridge tuning")
    ridge_search = GridSearchCV(
        estimator=ridge_pipe,
        param_grid=ridge_param_grid,
        cv=cv,
        scoring='neg_root_mean_squared_error',
        n_jobs=5
    )
    ridge_search.fit(X, y)
    best_ridge = ridge_search.best_estimator_

    # =========================================================
    # 5. LASSO
    # =========================================================
    lasso_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', Lasso(max_iter=20000))
    ])

    lasso_param_grid = {
        'model__alpha': np.logspace(-4, 2, 30)
    }

    print("starting lasso tuning")
    lasso_search = GridSearchCV(
        estimator=lasso_pipe,
        param_grid=lasso_param_grid,
        cv=cv,
        scoring='neg_root_mean_squared_error',
        n_jobs=5
    )
    lasso_search.fit(X, y)
    best_lasso = lasso_search.best_estimator_

    # =========================================================
    # 6. RANDOM FOREST
    # =========================================================
    rf_model = RandomForestRegressor(random_state=random_state, n_jobs=1)

    rf_param_dist = {
        'n_estimators': [100, 500, 1000, 2000, 5000, 6000, 8000],
        'max_depth': [None, 5, 10, 20, 40, 60],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': [2, 4, 6, 'log2', 'sqrt']
    }

    print("starting rf tuning")
    rf_search = RandomizedSearchCV(
        estimator=rf_model,
        param_distributions=rf_param_dist,
        n_iter=100,
        cv=cv,
        scoring='neg_root_mean_squared_error',
        random_state=random_state,
        n_jobs=5
    )
    rf_search.fit(X, y)
    best_rf = rf_search.best_estimator_

    # =========================================================
    # 7. XGBOOST + OPTUNA
    # =========================================================
    def xgb_objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 8000),
            'max_depth': trial.suggest_int('max_depth', 2, 16),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'max_delta_step': trial.suggest_int('max_delta_step', 0, 10),
            'objective': 'reg:squarederror',
            'random_state': random_state,
            'n_jobs':-1
        }

        model = XGBRegressor(**params, )

        scores = cross_val_score(
            model,
            X,
            y,
            cv=cv,
            scoring='neg_root_mean_squared_error',
            n_jobs=5
        )

        return -scores.mean()

    print("starting xgb tuning")
    study = optuna.create_study(direction='minimize')
    study.optimize(xgb_objective, n_trials=5, show_progress_bar=False)

    best_xgb = XGBRegressor(
        **study.best_params,
        objective='reg:squarederror',
        random_state=random_state,
        n_jobs=1
    )
    best_xgb.fit(X, y)

    # =========================================================
    # 8. REFIT BEST MODELS ON FULL DATA
    # =========================================================
    best_ridge.fit(X, y)
    best_lasso.fit(X, y)
    best_rf.fit(X, y)
    best_xgb.fit(X, y)

    # =========================================================
    # 9. PREDICT FOR THE INPUT ROW
    # =========================================================
    pred_ridge = float(best_ridge.predict(inputStatsList)[0])
    pred_lasso = float(best_lasso.predict(inputStatsList)[0])
    pred_rf = float(best_rf.predict(inputStatsList)[0])
    pred_xgb = float(best_xgb.predict(inputStatsList)[0])

    # =========================================================
    # 10. COLLECT RESULTS
    # =========================================================
    results_df = pd.DataFrame({
        'Model': ['Ridge', 'Lasso', 'RandomForest', 'XGBoost'],
        #'Model': ['XGBoost'],

        'Best_CV_RMSE': [
            -ridge_search.best_score_,
            -lasso_search.best_score_,
            -rf_search.best_score_,
            study.best_value
        ],
        'Prediction_for_input': [
            pred_ridge,
            pred_lasso,
            pred_rf,
            pred_xgb
        ]
    }).sort_values('Best_CV_RMSE').reset_index(drop=True)

    best_model_name = results_df.loc[0, 'Model']

    # =========================================================
    # 11. SAVE RESULTS TO FILES
    # =========================================================
    # Main results table
    results_txt_path = os.path.join(output_dir, "model_tuning_results.txt")
    results_csv_path = os.path.join(output_dir, "model_tuning_results.csv")

    with open(results_txt_path, "w") as f:
        f.write("=== MODEL TUNING RESULTS ===\n\n")
        f.write("Inference input row:\n")
        f.write(inputStatsList.to_string(index=False))
        f.write("\n\n")

        f.write("Cross-validation results:\n")
        f.write(results_df.to_string(index=False))
        f.write("\n\n")

        f.write(f"Best model by CV RMSE: {best_model_name}\n\n")

        f.write("Best hyperparameters:\n\n")
        f.write(f"Ridge: {ridge_search.best_params_}\n")
        f.write(f"Lasso: {lasso_search.best_params_}\n")
        f.write(f"RandomForest: {rf_search.best_params_}\n")
        f.write(f"XGBoost: {study.best_params}\n")

    results_df.to_csv(results_csv_path, index=False)

    # Detailed params JSON
    params_json = {
        "Ridge": ridge_search.best_params_,
        "Lasso": lasso_search.best_params_,
        "RandomForest": rf_search.best_params_,
        "XGBoost": study.best_params
    }

    with open(os.path.join(output_dir, "best_params.json"), "w") as f:
        json.dump(params_json, f, indent=4)

    # Save models
    joblib.dump(best_ridge, os.path.join(output_dir, "ridge_model.pkl"))
    joblib.dump(best_lasso, os.path.join(output_dir, "lasso_model.pkl"))
    joblib.dump(best_rf, os.path.join(output_dir, "random_forest_model.pkl"))
    joblib.dump(best_xgb, os.path.join(output_dir, "xgboost_model.pkl"))

    return {
        "results_df": results_df,
        "trained_models": {
            "ridge": best_ridge,
            "lasso": best_lasso,
            "random_forest": best_rf,
            "xgboost": best_xgb
        },
        "inputStatsList": inputStatsList
    }



