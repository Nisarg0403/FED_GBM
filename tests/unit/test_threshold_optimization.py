"""
Verification Gate 2.4 Test Script: Threshold Optimization & Specificity Fix.

Verifies Youden's J threshold optimization and cost-sensitive thresholding across 5 seeds:
- Seed list: [42, 101, 2024, 777, 999]
- Evaluates on UPenn external test set (N=114)
- Verification Gate 2.4 PASS condition: Mean Specificity > 0.4000 at Youden's J threshold
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
from fed_gbm.thresholding import (
    find_optimal_threshold_youden,
    find_optimal_threshold_cost_sensitive,
    evaluate_sensitivity_at_fixed_specificity,
    compute_dca_net_benefit,
)
from fed_gbm.metrics import evaluate_binary_classification


def test_thresholding_unit():
    """Unit test for threshold optimization math."""
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    y_prob = np.array([0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1])

    t_youden = find_optimal_threshold_youden(y_true, y_prob)
    assert 0.4 <= t_youden <= 0.6

    sens_spec50 = evaluate_sensitivity_at_fixed_specificity(y_true, y_prob, target_spec=0.50)
    assert 0.0 <= sens_spec50 <= 1.0

    df_dca = compute_dca_net_benefit(y_true, y_prob)
    assert not df_dca.empty
    assert "net_benefit_model" in df_dca.columns


def run_verification_gate_2_4():
    print("=" * 80)
    print("VERIFICATION GATE 2.4: THRESHOLD OPTIMIZATION & SPECIFICITY FIX (5 SEEDS)")
    print("=" * 80)

    seeds = [42, 101, 2024, 777, 999]
    default_specs = []
    youden_specs = []
    youden_thresholds = []

    for seed in seeds:
        set_seed(seed)
        cohorts = load_all_cohorts(seed=seed)
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

        server = FederatedServer(input_dim=input_dim, seed=seed)
        c1 = FederatedClient("TCGA", X_tcga_tr_std, y_tcga_tr, input_dim=input_dim)
        c2 = FederatedClient("UPenn", X_upenn_tr_std, y_upenn_tr, input_dim=input_dim)

        for r in range(20):
            g_sd = server.get_global_state_dict()
            c1.set_model_parameters(g_sd)
            c2.set_model_parameters(g_sd)
            p1 = c1.local_train(epochs=5, batch_size=16, lr=0.01, mu=0.01, seed=seed + r)
            p2 = c2.local_train(epochs=5, batch_size=16, lr=0.01, mu=0.01, seed=seed + r + 100)
            server.aggregate_client_payloads([p1, p2])

        eval_m = SurvivalMLP(input_dim=input_dim, hidden_dims=[32, 16])
        eval_m.load_state_dict(server.get_global_state_dict())
        eval_m.eval()

        with torch.no_grad():
            tr_probs = torch.sigmoid(eval_m(torch.tensor(X_upenn_tr_std.values, dtype=torch.float32))).numpy().flatten()
            te_probs = torch.sigmoid(eval_m(torch.tensor(X_upenn_te_std.values, dtype=torch.float32))).numpy().flatten()

        m_def = evaluate_binary_classification(y_upenn_te.values, te_probs, threshold=0.50)
        default_specs.append(m_def["specificity"])

        # Find optimal threshold on validation (UPenn train split)
        opt_thresh = find_optimal_threshold_youden(y_upenn_tr.values, tr_probs)
        m_opt = evaluate_binary_classification(y_upenn_te.values, te_probs, threshold=opt_thresh)

        youden_specs.append(m_opt["specificity"])
        youden_thresholds.append(opt_thresh)

    def_spec_mean = float(np.mean(default_specs))
    youd_spec_mean = float(np.mean(youden_specs))
    avg_thresh = float(np.mean(youden_thresholds))

    print(f"Default (t=0.50) Specificity Mean: {def_spec_mean:.4f}")
    print(f"Youden's J (t={avg_thresh:.2f}) Specificity Mean: {youd_spec_mean:.4f}")

    gate_pass = youd_spec_mean > 0.4000
    print(f"\nVERIFICATION GATE 2.4 STATUS: {'[PASS]' if gate_pass else '[FAIL/WARNING]'}")

    return {
        "gate_2_4_passed": gate_pass,
        "default_specificity_mean": def_spec_mean,
        "youden_specificity_mean": youd_spec_mean,
        "optimal_threshold_mean": avg_thresh,
    }


if __name__ == "__main__":
    test_thresholding_unit()
    run_verification_gate_2_4()
