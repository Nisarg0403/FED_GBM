"""
Evaluation Metrics & Calibration Utilities for FED-GBM v3.
Primary Metric: ROC-AUC.
Secondary Metrics: PR-AUC, Accuracy, F1, Sensitivity, Specificity, Brier Score, ECE.
Concordance Index (C-Index) is strictly EXCLUDED for binary classification.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Union
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    f1_score,
    recall_score,
    confusion_matrix,
    brier_score_loss
)


def logits_to_probs(logits: Union[np.ndarray, float]) -> np.ndarray:
    """
    Converts raw model logits to probabilities using the sigmoid function:
    p = 1 / (1 + exp(-z)).
    """
    z = np.asarray(logits, dtype=np.float64)
    # Clip z to avoid numerical overflow in exp
    z_clipped = np.clip(z, -88.0, 88.0)
    return 1.0 / (1.0 + np.exp(-z_clipped))


def calculate_expected_calibration_error(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    n_bins: int = 10
) -> float:
    """
    Computes Expected Calibration Error (ECE) across binned confidence scores.
    """
    y_true = np.asarray(y_true)
    y_probs = np.asarray(y_probs)
    
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n_samples = len(y_true)
    
    if n_samples == 0:
        return 0.0
        
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        # Mask for samples in current probability bin
        if i == n_bins - 1:
            in_bin = (y_probs >= bin_lower) & (y_probs <= bin_upper)
        else:
            in_bin = (y_probs >= bin_lower) & (y_probs < bin_upper)
            
        bin_size = np.sum(in_bin)
        if bin_size > 0:
            bin_acc = np.mean(y_true[in_bin])
            bin_conf = np.mean(y_probs[in_bin])
            ece += (bin_size / n_samples) * np.abs(bin_acc - bin_conf)
            
    return float(ece)


def evaluate_binary_classification(
    y_true: Union[np.ndarray, pd.Series, list],
    y_probs: Union[np.ndarray, pd.Series, list],
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Calculates primary and secondary binary classification metrics.
    Prohibits C-Index calculation.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_probs = np.asarray(y_probs, dtype=float)
    
    # Binary predictions
    y_pred = (y_probs >= threshold).astype(int)
    
    # ROC-AUC (Primary Metric)
    try:
        roc_auc = float(roc_auc_score(y_true, y_probs))
    except ValueError:
        # Fallback for single class case in synthetic test
        roc_auc = 0.5
        
    # PR-AUC
    try:
        pr_auc = float(average_precision_score(y_true, y_probs))
    except ValueError:
        pr_auc = 0.5
        
    # Accuracy & F1
    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    
    # Sensitivity (Recall) & Specificity
    sens = float(recall_score(y_true, y_pred, zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    
    # Brier Score
    brier = float(brier_score_loss(y_true, y_probs))
    
    # Expected Calibration Error (ECE)
    ece = calculate_expected_calibration_error(y_true, y_probs)
    
    return {
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'accuracy': acc,
        'f1': f1,
        'sensitivity': sens,
        'specificity': spec,
        'brier_score': brier,
        'ece': ece
    }
