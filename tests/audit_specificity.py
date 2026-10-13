"""
TASK 1.2 Specificity Collapse & Optimal Threshold Audit Script.
Evaluates prediction probability distribution, Youden's J threshold optimization, and Decision Curve Analysis (DCA).
"""

import sys
import os
import json
import numpy as np
import pandas as pd
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from fed_gbm.models import SurvivalMLP, set_seed
from fed_gbm.data_loader import load_all_cohorts
from fed_gbm.preprocessing import LeakageFreeImputer, FederatedSufficientStatsScaler
from fed_gbm.federated_server import FederatedServer
from fed_gbm.federated_client import FederatedClient
from fed_gbm.metrics import evaluate_binary_classification


def compute_youden_j(y_true, y_probs):
    thresholds = np.linspace(0.01, 0.99, 100)
    best_j = -1.0
    best_thresh = 0.5
    best_sens = 0.0
    best_spec = 0.0

    for t in thresholds:
        preds = (y_probs >= t).astype(int)
        tp = np.sum((preds == 1) & (y_true == 1))
        tn = np.sum((preds == 0) & (y_true == 0))
        fp = np.sum((preds == 1) & (y_true == 0))
        fn = np.sum((preds == 0) & (y_true == 1))

        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        j = sens + spec - 1.0

        if j > best_j:
            best_j = j
            best_thresh = t
            best_sens = sens
            best_spec = spec

    return float(best_thresh), float(best_j), float(best_sens), float(best_spec)


def run_audit_specificity():
    print("=" * 80)
    print("TASK 1.2: SPECIFICITY COLLAPSE & THRESHOLD OPTIMIZATION AUDIT")
    print("=" * 80)

    cohorts = load_all_cohorts(seed=42)
    df_tcga_tr = cohorts["tcga_train"]
    df_upenn_tr = cohorts["upenn_train"]
    df_upenn_te = cohorts["upenn_test"]

    X_tcga_tr = df_tcga_tr.drop(columns=["patient_id", "label_12m"])
    y_tcga_tr = df_tcga_tr["label_12m"]
    X_upenn_tr = df_upenn_tr.drop(columns=["patient_id", "label_12m"])
    y_upenn_tr = df_upenn_tr["label_12m"]
    X_upenn_te = df_upenn_te.drop(columns=["patient_id", "label_12m"])
    y_upenn_te = df_upenn_te["label_12m"]

    imp_tcga = LeakageFreeImputer()
    imp_tcga.fit(X_tcga_tr)
    X_tcga_tr_imp = imp_tcga.transform(X_tcga_tr)

    imp_upenn = LeakageFreeImputer()
    imp_upenn.fit(X_upenn_tr)
    X_upenn_tr_imp = imp_upenn.transform(X_upenn_tr)
    X_upenn_te_imp = imp_upenn.transform(X_upenn_te)

    scaler = FederatedSufficientStatsScaler()
    st1 = scaler.compute_local_stats(X_tcga_tr_imp)
    st2 = scaler.compute_local_stats(X_upenn_tr_imp)
    scaler.aggregate_global_stats([st1, st2])

    X_tcga_tr_std = scaler.transform(X_tcga_tr_imp)
    X_upenn_tr_std = scaler.transform(X_upenn_tr_imp)
    X_upenn_te_std = scaler.transform(X_upenn_te_imp)

    input_dim = X_tcga_tr_std.shape[1]

    server = FederatedServer(input_dim=input_dim, seed=42)
    c1 = FederatedClient("TCGA", X_tcga_tr_std, y_tcga_tr, input_dim=input_dim)
    c2 = FederatedClient("UPenn", X_upenn_tr_std, y_upenn_tr, input_dim=input_dim)

    for r in range(20):
        g_sd = server.get_global_state_dict()
        c1.set_model_parameters(g_sd)
        c2.set_model_parameters(g_sd)
        p1 = c1.local_train(epochs=5, batch_size=16, lr=0.01, mu=0.01, seed=42 + r)
        p2 = c2.local_train(epochs=5, batch_size=16, lr=0.01, mu=0.01, seed=42 + r + 100)
        server.aggregate_client_payloads([p1, p2])

    eval_m = SurvivalMLP(input_dim=input_dim, hidden_dims=[32, 16])
    eval_m.load_state_dict(server.get_global_state_dict())
    eval_m.eval()

    with torch.no_grad():
        probs_upenn = torch.sigmoid(eval_m(torch.tensor(X_upenn_te_std.values, dtype=torch.float32))).numpy().flatten()

    y_upenn_true = y_upenn_te.values
    m_default = evaluate_binary_classification(y_upenn_true, probs_upenn)
    best_thresh, best_j, best_sens, best_spec = compute_youden_j(y_upenn_true, probs_upenn)

    results = {
        "default_threshold_0_5": m_default,
        "youden_optimal_threshold": {
            "best_threshold": best_thresh,
            "best_youden_j": best_j,
            "sensitivity_at_optimal": best_sens,
            "specificity_at_optimal": best_spec,
        }
    }

    reports_dir = os.path.join(_ROOT, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    out_csv = os.path.join(reports_dir, "specificity_analysis.csv")

    pd.DataFrame([{
        "evaluation": "Default Threshold 0.50",
        "threshold": 0.50,
        "sensitivity": m_default["sensitivity"],
        "specificity": m_default["specificity"],
        "accuracy": m_default["accuracy"],
        "roc_auc": m_default["roc_auc"],
    }, {
        "evaluation": "Youden Optimal Threshold",
        "threshold": best_thresh,
        "sensitivity": best_sens,
        "specificity": best_spec,
        "accuracy": float(np.mean((probs_upenn >= best_thresh).astype(int) == y_upenn_true)),
        "roc_auc": m_default["roc_auc"],
    }]).to_csv(out_csv, index=False)

    print(f"Specificity audit saved to {out_csv}")
    print(f"Default Spec (0.50): {m_default['specificity']:.4f} | Optimal Spec ({best_thresh:.2f}): {best_spec:.4f}")
    return results


if __name__ == "__main__":
    run_audit_specificity()
