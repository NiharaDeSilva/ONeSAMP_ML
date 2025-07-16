import os
import time
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split
import joblib
from sklearn.preprocessing import StandardScaler
from predict import predict_random_forest, predict_xgboost, preidct_with_stats

# Predict again using new Z
Z_scaled = scaler.transform(Z)
xgb_prediction = xgb_model.predict(Z_scaled)


scaler_path = os.path.join(output_path, f'scaler_{sampleSize}x{loci}.joblib')
rf_path = os.path.join(output_path, f"rf_model_{sampleSize}x{loci}.joblib")
xgb_path = os.path.join(output_path, f"xgb_model_{sampleSize}x{loci}.joblib")
lasso_path = os.path.join(output_path, f"lasso_model_{sampleSize}x{loci}.joblib")
ridge_path = os.path.join(output_path, f"ridge_model_{sampleSize}x{loci}.joblib")



def load_scaler(scaler_path):
    if os.path.exists(scaler_path):
        return joblib.load(scaler_path)
    else:
        raise FileNotFoundError(f"Scaler file {scaler_path} does not exist.")


def load_rf_model(rf_path, Z, scaler_path):
    if os.path.exists(rf_path):
        scaler = load_scaler(scaler_path)
        rf_model = joblib.load(rf_path)
        Z_scaled = scaler.transform(Z)
        rf_prediction = rf_model.predict(Z_scaled)
        print(f"Mean: {rf_prediction['mean']:.4f}, Median: {rf_prediction['median']:.4f}, "
              f"Min: {rf_prediction['min']:.4f}, Max: {rf_prediction['max']:.4f}, "
              f"95% CI: ({rf_prediction['95% CI'][0]:.4f}, {rf_prediction['95% CI'][1]:.4f})")
    else:
        raise FileNotFoundError(f"Model file {rf_path} does not exist.")



def load_xgb_model(xgb_path, Z, scaler_path):
    if os.path.exists(xgb_path):
        scaler = load_scaler(scaler_path)
        xgb_model = joblib.load(xgb_path)
        Z_scaled = scaler.transform(Z)
        xgb_prediction = xgb_model.predict(Z_scaled)
        print(f"Mean: {xgb_prediction['mean']:.4f}, Median: {xgb_prediction['median']:.4f},"
              f"95% CI: ({xgb_prediction['95% CI'][0]:.4f}, {xgb_prediction['95% CI'][1]:.4f})")
    else:
        raise FileNotFoundError(f"Model file {xgb_path} does not exist.")



def load_lasso_model(lasso_path, Z, scaler_path):
    if os.path.exists(lasso_path):
        lasso_model = joblib.load(lasso_path)
        scaler = load_scaler(scaler_path)
        Z_scaled = scaler.transform(Z)
        lasso_prediction = lasso_model.predict(Z_scaled)
        print(f"Mean: {lasso_prediction['mean']:.4f}, Median: {lasso_prediction['median']:.4f}, "
              f"Min: {lasso_prediction['min']:.4f}, Max: {lasso_prediction['max']:.4f}, "
              f"95% CI: ({lasso_prediction['95% CI'][0]:.4f}, {lasso_prediction['95% CI'][1]:.4f})")
    else:
        raise FileNotFoundError(f"Model file {lasso_path} does not exist.")



def load_ridge_model(ridge_path, Z, scaler_path):
    if os.path.exists(ridge_path):
        ridge_model = joblib.load(ridge_path)
        scaler = load_scaler(scaler_path)
        Z_scaled = scaler.transform(Z)
        ridge_prediction = ridge_model.predict(Z_scaled)
        print(f"Mean: {ridge_prediction['mean']:.4f}, Median: {ridge_prediction['median']:.4f}, "
              f"Min: {ridge_prediction['min']:.4f}, Max: {ridge_prediction['max']:.4f}, "
              f"95% CI: ({ridge_prediction['95% CI'][0]:.4f}, {ridge_prediction['95% CI'][1]:.4f})")    
    else:
        raise FileNotFoundError(f"Model file {ridge_path} does not exist.")


