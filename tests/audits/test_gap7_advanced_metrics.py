"""
GAP 7 Test Script: Advanced Clinical Metrics (NRI, IDI, Brier Decomposition).

Evaluates advanced prognostic metrics on UPenn External Test Set:
1. NRI & IDI (CA-FedAvg vs FedAvg Baseline)
2. Murphy's Brier Score Decomposition (Reliability, Resolution, Uncertainty)
3. Time-dependent ROC-AUC proxy

Outputs reports/advanced_metrics.csv.
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
from fed_gbm.ca_fedavg import CalibrationAwareFedAvg
from fed_gbm.metrics import evaluate_binary_classification
from fed_gbm.advanced_metrics import compute_nri_and_idi, compute_brier_decomposition, compute_time_dependent_auc


def run_gap7_advanced_metrics():
    print("=" * 80)
    print("GAP 7: ADVANCED CLINICAL METRICS (NRI, IDI, BRIER DECOMPOSITION)")
    print("=" * 80)

    seed = 42
    set_seed(seed)
    cohorts = load_all_cohorts(seed=seed)

    df_tcga_tr = cohorts["tcga_train"]
    df_upenn_tr = cohorts["upenn_train"]
    df_upenn_te = cohorts["upenn_test"]

    X1 = df_tcga_tr.drop(columns=["patient_id", "label_12m"])
    y1 = df_tcga_tr["label_12m"]
    X2 = df_upenn_tr.drop(columns=["patient_id", "label_12m"])
    y2 = df_upenn_tr["label_12m"]
    Xte = df_upenn_te.drop(columns=["patient_id", "label_12m"])
    yte = df_upenn_te["label_12m"]

    imp1 = LeakageFreeImputer(); imp1.fit(X1); X1_imp = imp1.transform(X1)
    imp2 = LeakageFreeImputer(); imp2.fit(X2); X2_imp = imp2.transform(X2); Xte_imp = imp2.transform(Xte)
    scaler = FederatedSufficientStatsScaler()
    st1 = scaler.compute_local_stats(X1_imp); st2 = scaler.compute_local_stats(X2_imp)
    scaler.aggregate_global_stats([st1, st2])
    X1_std = scaler.transform(X1_imp); X2_std = scaler.transform(X2_imp); Xte_std = scaler.transform(Xte_imp)

    input_dim = X1_std.shape[1]

    # Baseline FedAvg
    srv_fa = FederatedServer(input_dim=input_dim, seed=seed)
    c1_fa = FederatedClient("TCGA", X1_std, y1, input_dim=input_dim)
    c2_fa = FederatedClient("UPenn", X2_std, y2, input_dim=input_dim)
    for r in range(20):
        g_sd = srv_fa.get_global_state_dict()
        c1_fa.set_model_parameters(g_sd); c2_fa.set_model_parameters(g_sd)
        p1 = c1_fa.local_train(epochs=5, batch_size=16, lr=0.01, mu=0.0, seed=seed+r)
        p2 = c2_fa.local_train(epochs=5, batch_size=16, lr=0.01, mu=0.0, seed=seed+r+100)
        srv_fa.aggregate_client_payloads([p1, p2])

    eval_fa = SurvivalMLP(input_dim=input_dim, hidden_dims=[32, 16])
    eval_fa.load_state_dict(srv_fa.get_global_state_dict())
    eval_fa.eval()
    with torch.no_grad():
        p_fa = torch.sigmoid(eval_fa(torch.tensor(Xte_std.values, dtype=torch.float32))).numpy().flatten()

    # CA-FedAvg
    srv_ca = FederatedServer(input_dim=input_dim, seed=seed)
    c1_ca = FederatedClient("TCGA", X1_std, y1, input_dim=input_dim)
    c2_ca = FederatedClient("UPenn", X2_std, y2, input_dim=input_dim)
    ca_agg = CalibrationAwareFedAvg(lambda_param=1.0)
    for r in range(20):
        g_sd = srv_ca.get_global_state_dict()
        c1_ca.set_model_parameters(g_sd); c2_ca.set_model_parameters(g_sd)
        p1 = c1_ca.local_train(epochs=5, batch_size=16, lr=0.01, mu=0.0, seed=seed+r)
        p2 = c2_ca.local_train(epochs=5, batch_size=16, lr=0.01, mu=0.0, seed=seed+r+100)
        ece1 = evaluate_binary_classification(y1.values, torch.sigmoid(c1_ca.model(c1_ca.X_train_tensor)).detach().numpy().flatten())["ece"]
        ece2 = evaluate_binary_classification(y2.values, torch.sigmoid(c2_ca.model(c2_ca.X_train_tensor)).detach().numpy().flatten())["ece"]
        srv_ca.global_model.load_state_dict(ca_agg.aggregate([p1, p2], client_eces=[ece1, ece2]))

    eval_ca = SurvivalMLP(input_dim=input_dim, hidden_dims=[32, 16])
    eval_ca.load_state_dict(srv_ca.get_global_state_dict())
    eval_ca.eval()
    with torch.no_grad():
        p_ca = torch.sigmoid(eval_ca(torch.tensor(Xte_std.values, dtype=torch.float32))).numpy().flatten()

    # Compute advanced metrics
    nri_idi = compute_nri_and_idi(yte.values, p_fa, p_ca, threshold=0.50)
    brier_decomp = compute_brier_decomposition(yte.values, p_ca)
    td_auc = compute_time_dependent_auc(yte.values, p_ca)

    adv_records = [
        {"metric_category": "Improvement vs FedAvg", "metric_name": "Net Reclassification Index (NRI)", "value": nri_idi["nri_total"]},
        {"metric_category": "Improvement vs FedAvg", "metric_name": "Integrated Discrimination Improvement (IDI)", "value": nri_idi["idi_total"]},
        {"metric_category": "Brier Decomposition", "metric_name": "Total Brier Score", "value": brier_decomp["brier_score"]},
        {"metric_category": "Brier Decomposition", "metric_name": "Reliability (Calibration Error)", "value": brier_decomp["reliability"]},
        {"metric_category": "Brier Decomposition", "metric_name": "Resolution (Discriminative Power)", "value": brier_decomp["resolution"]},
        {"metric_category": "Brier Decomposition", "metric_name": "Uncertainty (Baseline Variance)", "value": brier_decomp["uncertainty"]},
        {"metric_category": "Prognostic Metric", "metric_name": "Time-Dependent ROC-AUC", "value": td_auc},
    ]

    df_res = pd.DataFrame(adv_records)
    print("\nADVANCED METRICS RESULTS:")
    print(df_res)

    os.makedirs(os.path.join(_ROOT, "reports"), exist_ok=True)
    df_res.to_csv(os.path.join(_ROOT, "reports", "advanced_metrics.csv"), index=False)

    return {"results": adv_records}


if __name__ == "__main__":
    run_gap7_advanced_metrics()
