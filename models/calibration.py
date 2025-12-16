import matplotlib
matplotlib.use("Agg")  # Safe for HiPerGator (no display)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from statsmodels.nonparametric.smoothers_lowess import lowess
import os


def ensure_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)


def calibration_curves(true, pred, model_name, save_dir="../../blue/boucher/suhashidesilva/2025/Revision/ONeSAMP_ML/plots", bins=10, folds=None):
    """
    Generate:
        1. Pred vs True
        2. Residuals vs Pred
        3. Binned calibration curve
        4. LOWESS smooth calibration curve
        5. Linear calibration fit + R²
        6. Stratified calibration curves by validation fold (optional)
    """

    ensure_folder(save_dir)
    prefix = os.path.join(save_dir, f"{model_name}")

    true = np.array(true)
    pred = np.array(pred)

    # ---------------------------------------------------------
    # (1) Predicted vs True
    # ---------------------------------------------------------
    plt.figure()
    plt.scatter(pred, true, s=8, alpha=0.4)
    lims = [min(pred.min(), true.min()), max(pred.max(), true.max())]
    plt.plot(lims, lims, "k--")

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"{model_name}: Predicted vs True")
    plt.tight_layout()
    plt.savefig(f"{prefix}_pred_vs_true.png", dpi=300)
    plt.close()


    # ---------------------------------------------------------
    # (2) Residuals vs Predicted
    # ---------------------------------------------------------
    residuals = pred - true

    plt.figure()
    plt.scatter(pred, residuals, s=8, alpha=0.4)
    plt.axhline(0, color="black")
    plt.xlabel("Predicted")
    plt.ylabel("Residual (Pred - True)")
    plt.title(f"{model_name}: Residuals vs Predicted")
    plt.tight_layout()
    plt.savefig(f"{prefix}_residuals.png", dpi=300)
    plt.close()


    # ---------------------------------------------------------
    # (3) Binned Calibration Curve
    # ---------------------------------------------------------
    sorted_idx = np.argsort(pred)
    pred_sorted = pred[sorted_idx]
    true_sorted = true[sorted_idx]

    bin_size = len(pred) // bins
    bin_pred, bin_true = [], []

    for i in range(bins):
        s = i * bin_size
        e = (i + 1) * bin_size
        bin_pred.append(np.mean(pred_sorted[s:e]))
        bin_true.append(np.mean(true_sorted[s:e]))

    plt.figure()
    plt.plot(bin_pred, bin_true, "o-", label="Binned Mean")
    plt.plot([min(bin_pred), max(bin_pred)], [min(bin_pred), max(bin_pred)], "k--", label="Perfect Calibration")

    plt.xlabel("Mean Predicted (bin)")
    plt.ylabel("Mean True (bin)")
    plt.title(f"{model_name}: Binned Calibration Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{prefix}_binned.png", dpi=300)
    plt.close()


    # ---------------------------------------------------------
    # (4) LOWESS Smooth Calibration Curve
    # ---------------------------------------------------------
    lowess_curve = lowess(true, pred, frac=0.25, return_sorted=True)

    plt.figure()
    plt.scatter(pred, true, s=5, alpha=0.2, label="Raw")
    plt.plot(lowess_curve[:, 0], lowess_curve[:, 1], "r-", linewidth=2, label="LOWESS")
    plt.plot(lims, lims, "k--", label="Perfect Calibration")

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"{model_name}: LOWESS Calibration Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{prefix}_lowess.png", dpi=300)
    plt.close()


    # ---------------------------------------------------------
    # (5) Linear Calibration Fit + R²
    # ---------------------------------------------------------
    lr = LinearRegression()
    lr.fit(pred.reshape(-1, 1), true)
    pred_fit = lr.predict(pred.reshape(-1, 1))
    r2 = r2_score(true, pred_fit)

    coef = lr.coef_[0]
    intercept = lr.intercept_

    plt.figure()
    plt.scatter(pred, true, s=5, alpha=0.3)
    plt.plot(pred, pred_fit, "r-", linewidth=2, label=f"Fit: y = {coef:.3f}x + {intercept:.3f}")
    plt.plot(lims, lims, "k--", label="Perfect Calibration")

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"{model_name}: Linear Calibration Fit (R² = {r2:.3f})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{prefix}_linear_fit.png", dpi=300)
    plt.close()


    # ---------------------------------------------------------
    # (6) Stratification by Fold (optional)
    # folds must be an array same length as true/pred
    # ---------------------------------------------------------
    if folds is not None:
        unique_folds = np.unique(folds)

        for f in unique_folds:
            idx = (folds == f)
            true_f = true[idx]
            pred_f = pred[idx]

            # LOWESS per fold
            fold_lowess = lowess(true_f, pred_f, frac=0.4, return_sorted=True)

            plt.figure()
            plt.scatter(pred_f, true_f, s=5, alpha=0.3)
            plt.plot(fold_lowess[:, 0], fold_lowess[:, 1], "r-", label="LOWESS")
            plt.plot(lims, lims, "k--")

            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title(f"{model_name}: Fold {f} Calibration Curve")
            plt.tight_layout()
            plt.savefig(f"{prefix}_fold_{f}.png", dpi=300)
            plt.close()

    print(f"[Saved] Calibration plots for {model_name} → {save_dir}/")
