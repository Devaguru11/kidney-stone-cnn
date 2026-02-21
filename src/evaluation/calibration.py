# src/evaluation/calibration.py
# src/evaluation/calibration.py
import torch, numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import fbeta_score, precision_recall_curve

from sklearn.calibration import CalibratedClassifierCV, calibration_curve

def find_optimal_threshold(all_labels, all_probs,
                            beta: float = 2.0) -> dict:
    """
    Find threshold that maximises F-beta score.
    beta=2 weights recall twice as much as precision.
    """
    thresholds = np.arange(0.05, 0.95, 0.01)
    best_thresh = 0.5
    best_f2     = 0.0
    results     = []

    for t in thresholds:
        preds = (np.array(all_probs) >= t).astype(int)
        f2    = fbeta_score(all_labels, preds, beta=beta, zero_division=0)
        sens  = sum((p==1 and l==1) for p,l in zip(preds,all_labels)) / max(sum(all_labels),1)
        results.append({'threshold': round(t,2), 'f2': round(f2,4), 'sensitivity': round(sens,4)})
        if f2 > best_f2:
            best_f2     = f2
            best_thresh = t

    print(f'Optimal threshold (F2): {best_thresh:.2f}')
    print(f'Best F2-score:          {best_f2:.4f}')
    return {'threshold': best_thresh, 'f2': best_f2, 'all_results': results}

def plot_threshold_curve(all_labels, all_probs, save_path):
    """Plot F2, Precision, Recall vs threshold."""
    thresholds = np.arange(0.05, 0.95, 0.01)
    f2s, sens_list, prec_list = [], [], []

    for t in thresholds:
        preds = (np.array(all_probs) >= t).astype(int)
        f2  = fbeta_score(all_labels, preds, beta=2, zero_division=0)
        tp  = sum((p==1 and l==1) for p,l in zip(preds,all_labels))
        fn  = sum((p==0 and l==1) for p,l in zip(preds,all_labels))
        fp  = sum((p==1 and l==0) for p,l in zip(preds,all_labels))
        sens = tp / max(tp+fn, 1)
        prec = tp / max(tp+fp, 1)
        f2s.append(f2); sens_list.append(sens); prec_list.append(prec)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(thresholds, f2s,        label='F2-score',    color='#1E88E5', lw=2.5)
    ax.plot(thresholds, sens_list,   label='Sensitivity', color='#34C98B', lw=2)
    ax.plot(thresholds, prec_list,   label='Precision',   color='#F59E42', lw=2)
    ax.axhline(0.92, color='red', linestyle='--', alpha=0.6, label='Sens target 0.92')
    best_t = thresholds[np.argmax(f2s)]
    ax.axvline(best_t, color='purple', linestyle=':', lw=2, label=f'Optimal t={best_t:.2f}')
    ax.set_xlabel('Decision Threshold'); ax.set_ylabel('Score')
    ax.set_title('Threshold vs Metrics — Validation Set')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.show()
    print(f'Saved to {save_path}')

def plot_calibration_curve(all_labels, all_probs, save_path):
    """Reliability diagram — is the model well-calibrated?"""
    prob_true, prob_pred = calibration_curve(
        all_labels, all_probs, n_bins=10, strategy='uniform')
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(prob_pred, prob_true, 's-', color='#1E88E5', lw=2, label='Model')
    ax.plot([0,1],[0,1],'k--', label='Perfect calibration')
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')
    ax.set_title('Calibration Curve (Reliability Diagram)')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.show()
    print(f'Saved to {save_path}')