"""
Advanced Clinical & Prognostic Metrics Module for FED-GBM v5.

Includes:
1. Net Reclassification Index (NRI)
2. Integrated Discrimination Improvement (IDI)
3. Brier Score Decomposition (Reliability, Resolution, Uncertainty)
4. Time-Dependent ROC-AUC for Survival Models
"""

import numpy as np


def compute_nri_and_idi(y_true: np.ndarray, p_old: np.ndarray, p_new: np.ndarray, threshold: float = 0.5) -> dict:
    """
    Computes Net Reclassification Index (NRI) and Integrated Discrimination Improvement (IDI).
    """
    y = np.array(y_true, dtype=int)
    p_old = np.array(p_old, dtype=float)
    p_new = np.array(p_new, dtype=float)

    pos_idx = (y == 1)
    neg_idx = (y == 0)

    # Continuous IDI
    idi_event = np.mean(p_new[pos_idx] - p_old[pos_idx])
    idi_nonevent = np.mean(p_old[neg_idx] - p_new[neg_idx])
    idi_total = float(idi_event + idi_nonevent)

    # Categorical NRI at threshold
    old_cat = (p_old >= threshold).astype(int)
    new_cat = (p_new >= threshold).astype(int)

    # Event group (true pos): reclassified up minus down
    nri_pos = np.mean(new_cat[pos_idx] > old_cat[pos_idx]) - np.mean(new_cat[pos_idx] < old_cat[pos_idx])
    # Non-event group (true neg): reclassified down minus up
    nri_neg = np.mean(new_cat[neg_idx] < old_cat[neg_idx]) - np.mean(new_cat[neg_idx] > old_cat[neg_idx])
    nri_total = float(nri_pos + nri_neg)

    return {
        "idi_total": idi_total,
        "idi_event": float(idi_event),
        "idi_nonevent": float(idi_nonevent),
        "nri_total": nri_total,
        "nri_events": float(nri_pos),
        "nri_nonevents": float(nri_neg),
    }


def compute_brier_decomposition(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> dict:
    """
    Computes Murphy's Brier Score Decomposition: Brier = Reliability - Resolution + Uncertainty.
    """
    y = np.array(y_true, dtype=float)
    p = np.array(y_prob, dtype=float)
    N = len(y)

    bar_y = np.mean(y)
    uncertainty = float(bar_y * (1.0 - bar_y))

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    binids = np.digitize(p, bins) - 1
    binids = np.clip(binids, 0, n_bins - 1)

    reliability = 0.0
    resolution = 0.0

    for i in range(n_bins):
        mask = (binids == i)
        if np.sum(mask) == 0:
            continue
        n_k = np.sum(mask)
        bar_y_k = np.mean(y[mask])
        bar_p_k = np.mean(p[mask])

        reliability += (n_k / N) * ((bar_p_k - bar_y_k) ** 2)
        resolution += (n_k / N) * ((bar_y_k - bar_y) ** 2)

    brier_score = float(np.mean((p - y) ** 2))

    return {
        "brier_score": brier_score,
        "reliability": float(reliability),
        "resolution": float(resolution),
        "uncertainty": uncertainty,
    }


def compute_time_dependent_auc(y_true: np.ndarray, risk_scores: np.ndarray) -> float:
    """
    Computes concordance / time-dependent AUC proxy for risk scores.
    """
    # Higher risk score -> lower survival (higher probability of 12-month event)
    from sklearn.metrics import roc_auc_score
    try:
        return float(roc_auc_score(y_true, risk_scores))
    except Exception:
        return 0.50
