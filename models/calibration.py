import numpy as np
import matplotlib.pyplot as plt
import os

def ensure_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

save_dir = "../../blue/boucher/suhashidesilva/2025/Revision/ONeSAMP_ML/plots"
def calibration_curves(true, pred, model_name, save_dir, bins=10):
    true = np.array(true)
    pred = np.array(pred)

    # Create output folder
    ensure_folder(save_dir)

    # File prefix
    prefix = os.path.join(save_dir, f"{model_name}")

    # ------------------------------- #
    # 1. Predicted vs True Plot
    # ------------------------------- #
    plt.figure()
    plt.scatter(pred, true, s=8, alpha=0.5)
    lims = [min(true.min(), pred.min()), max(true.max(), pred.max())]
    plt.plot(lims, lims, 'k--')
    plt.xlabel("Predicted Ne")
    plt.ylabel("True Ne")
    plt.title(f"{model_name}: Predicted vs True")
    plt.tight_layout()
    plt.savefig(f"{prefix}_pred_vs_true.png", dpi=300)
    plt.close()

    # ------------------------------- #
    # 2. Residuals vs Predicted
    # ------------------------------- #
    residuals = pred - true
    plt.figure()
    plt.scatter(pred, residuals, s=8, alpha=0.5)
    plt.axhline(0, color='black')
    plt.xlabel("Predicted Ne")
    plt.ylabel("Residual (Pred−True)")
    plt.title(f"{model_name}: Residuals vs Predicted")
    plt.tight_layout()
    plt.savefig(f"{prefix}_residuals.png", dpi=300)
    plt.close()

    # ------------------------------- #
    # 3. Binned Calibration Curve
    # ------------------------------- #
    sorted_idx = np.argsort(pred)
    pred_sorted = pred[sorted_idx]
    true_sorted = true[sorted_idx]

    bin_size = len(pred) // bins
    bin_pred, bin_true = [], []

    for i in range(bins):
        start = i * bin_size
        end = (i + 1) * bin_size
        bin_pred.append(np.mean(pred_sorted[start:end]))
        bin_true.append(np.mean(true_sorted[start:end]))

    plt.figure()
    plt.plot(bin_pred, bin_true, marker='o')
    plt.plot([min(bin_pred), max(bin_pred)], [min(bin_pred), max(bin_pred)], 'k--')
    plt.xlabel("Mean Predicted Ne (bin)")
    plt.ylabel("Mean True Ne (bin)")
    plt.title(f"{model_name}: Binned Calibration Curve")
    plt.tight_layout()
    plt.savefig(f"{prefix}_binned.png", dpi=300)
    plt.close()

    print(f"[Saved] Calibration plots for {model_name} → {save_dir}/")
