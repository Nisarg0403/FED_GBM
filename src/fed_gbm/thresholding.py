"""
Threshold Optimization & Decision Curve Analysis (DCA) Module for FED-GBM v4.

Provides threshold optimization methods:
1. Youden's J statistic: J(t) = Sensitivity(t) + Specificity(t) - 1
2. Cost-sensitive thresholding: min (Cost_FN * FN(t) + Cost_FP * FP(t))
3. Sens@Spec=0.5 and Sens@Spec=0.8 evaluation
4. Decision Curve Analysis (DCA) net benefit calculation
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, confusion_matrix


def find_optimal_threshold_youden(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Finds decision threshold that maximizes Youden's J statistic = Sens + Spec - 1."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j_scores = tpr - fpr  # equivalent to tpr + (1 - fpr) - 1
    best_idx = np.argmax(j_scores)
    best_thresh = float(thresholds[best_idx])
    # Handle infinite threshold boundary returned by roc_curve
    if best_thresh > 1.0:
        best_thresh = 1.0
    return best_thresh


def find_optimal_threshold_cost_sensitive(
    y_true: np.ndarray, y_prob: np.ndarray, cost_fn: float = 3.0, cost_fp: float = 1.0
) -> float:
    """Finds threshold minimizing total misclassification cost (cost_fn * FN + cost_fp * FP)."""
    thresholds = np.linspace(0.01, 0.99, 99)
    best_cost = float("inf")
    best_thresh = 0.5

    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
        total_cost = cost_fn * fn + cost_fp * fp
        if total_cost < best_cost:
            best_cost = total_cost
            best_thresh = float(t)

    return best_thresh


def evaluate_sensitivity_at_fixed_specificity(
    y_true: np.ndarray, y_prob: np.ndarray, target_spec: float = 0.80
) -> float:
    """Finds sensitivity at closest target specificity (e.g. 0.50 or 0.80)."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    specs = 1.0 - fpr
    # Find index where spec is closest to target_spec
    idx = np.argmin(np.abs(specs - target_spec))
    return float(tpr[idx])


def compute_dca_net_benefit(
    y_true: np.ndarray, y_prob: np.ndarray, thresholds: np.ndarray = None
) -> pd.DataFrame:
    """
    Computes Decision Curve Analysis (DCA) Net Benefit across threshold probabilities.
    Net Benefit = (TP / N) - (FP / N) * (p_t / (1 - p_t))
    """
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.50, 50)

    N = len(y_true)
    results = []

    for p_t in thresholds:
        preds = (y_prob >= p_t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()

        weight = p_t / (1.0 - p_t)
        nb_model = (tp / N) - (fp / N) * weight

        # Treat all baseline: TP = all positive cases, FP = all negative cases
        total_pos = np.sum(y_true == 1)
        total_neg = np.sum(y_true == 0)
        nb_all = (total_pos / N) - (total_neg / N) * weight

        results.append({
            "threshold": float(p_t),
            "net_benefit_model": float(nb_model),
            "net_benefit_treat_all": float(nb_all),
            "net_benefit_treat_none": 0.0,
        })

    return pd.DataFrame(results)
