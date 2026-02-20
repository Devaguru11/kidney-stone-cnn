# src/training/metrics.py
import torch
import numpy as np
from sklearn.metrics import (roc_auc_score, confusion_matrix,
                             f1_score, fbeta_score)

def compute_metrics(all_labels, all_probs, threshold=0.5):
    """
    Compute all clinical metrics from collected labels and probabilities.
    Returns a dict of metric_name -> value.
    """
    all_preds = (np.array(all_probs) >= threshold).astype(int)
    all_labels = np.array(all_labels)

    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()

    sensitivity = tp / (tp + fn + 1e-8)   # recall for stone class
    specificity = tn / (tn + fp + 1e-8)
    precision   = tp / (tp + fp + 1e-8)
    f2          = fbeta_score(all_labels, all_preds, beta=2)
    auc         = roc_auc_score(all_labels, all_probs)

    return {
        'sensitivity': round(sensitivity, 4),   # PRIMARY — must be >= 0.92
        'specificity': round(specificity, 4),
        'precision':   round(precision, 4),
        'f2_score':    round(f2, 4),
        'auc_roc':     round(auc, 4),           # PRIMARY — must be >= 0.95
        'tp': int(tp), 'fp': int(fp),
        'tn': int(tn), 'fn': int(fn),
    }

