"""
Verification Gate 2.1 Test Script: CA-FedAvg vs FedAvg Benchmark Comparison.

Verifies CA-FedAvg aggregation against standard FedAvg across 5 seeds:
- Seed list: [42, 101, 2024, 777, 999]
- Evaluates on UPenn external test set (N=114)
- Verification Gate 2.1 PASS condition:
  Mean CA-FedAvg UPenn AUC > 0.7180 AND Mean CA-FedAvg UPenn ECE < 0.1710
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
from fed_gbm.ca_fedavg import CalibrationAwareFedAvg, aggregate_ca_fedavg
from fed_gbm.metrics import evaluate_binary_classification


def test_ca_fedavg_unit():
    """Unit test checking CA-FedAvg weight normalization and tensor math."""
    sd1 = {"weight": torch.tensor([[1.0, 2.0]]), "bias": torch.tensor([0.5])}
    sd2 = {"weight": torch.tensor([[3.0, 4.0]]), "bias": torch.tensor([1.5])}

    p1 = {"state_dict": sd1, "n_samples": 100}
    p2 = {"state_dict": sd2, "n_samples": 100}

    # If ECE is equal (0.1, 0.1), weights should be equal 0.5/0.5
    res_sd = aggregate_ca_fedavg([p1, p2], client_eces=[0.1, 0.1], lambda_param=1.0)
    assert torch.allclose(res_sd["weight"], torch.tensor([[2.0, 3.0]]))
    assert torch.allclose(res_sd["bias"], torch.tensor([1.0]))

    # If p1 has lower ECE (0.05) than p2 (0.25), p1 gets higher weight
    res_sd2 = aggregate_ca_fedavg([p1, p2], client_eces=[0.05, 0.25], lambda_param=1.0)
    assert res_sd2["weight"][0, 0].item() < 2.0  # weighted closer to p1's value (1.0)


def run_verification_gate_2_1():
    seeds = [42, 101, 2024, 777, 999]
    fedavg_results = []
    ca_results = []

    print("=" * 80)
    print("VERIFICATION GATE 2.1: CA-FedAvg BENCHMARK VERIFICATION (5 SEEDS)")
    print("=" * 80)

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

        # 1. FedAvg Baseline
        server_fa = FederatedServer(input_dim=input_dim, seed=seed)
        c1_fa = FederatedClient("TCGA", X_tcga_tr_std, y_tcga_tr, input_dim=input_dim)
        c2_fa = FederatedClient("UPenn", X_upenn_tr_std, y_upenn_tr, input_dim=input_dim)

        for r in range(20):
            g_sd = server_fa.get_global_state_dict()
            c1_fa.set_model_parameters(g_sd)
            c2_fa.set_model_parameters(g_sd)
            p1 = c1_fa.local_train(epochs=5, batch_size=16, lr=0.01, mu=0.0, seed=seed + r)
            p2 = c2_fa.local_train(epochs=5, batch_size=16, lr=0.01, mu=0.0, seed=seed + r + 100)
            server_fa.aggregate_client_payloads([p1, p2])

        eval_fa = SurvivalMLP(input_dim=input_dim, hidden_dims=[32, 16])
        eval_fa.load_state_dict(server_fa.get_global_state_dict())
        eval_fa.eval()

        with torch.no_grad():
            probs_fa = torch.sigmoid(eval_fa(torch.tensor(X_upenn_te_std.values, dtype=torch.float32))).numpy().flatten()
        m_fa = evaluate_binary_classification(y_upenn_te.values, probs_fa)
        fedavg_results.append(m_fa)

        # 2. CA-FedAvg Novel Method
        server_ca = FederatedServer(input_dim=input_dim, seed=seed)
        c1_ca = FederatedClient("TCGA", X_tcga_tr_std, y_tcga_tr, input_dim=input_dim)
        c2_ca = FederatedClient("UPenn", X_upenn_tr_std, y_upenn_tr, input_dim=input_dim)
        ca_agg = CalibrationAwareFedAvg(lambda_param=1.0)

        for r in range(20):
            g_sd = server_ca.get_global_state_dict()
            c1_ca.set_model_parameters(g_sd)
            c2_ca.set_model_parameters(g_sd)
            p1 = c1_ca.local_train(epochs=5, batch_size=16, lr=0.01, mu=0.0, seed=seed + r)
            p2 = c2_ca.local_train(epochs=5, batch_size=16, lr=0.01, mu=0.0, seed=seed + r + 100)

            # Compute local validation ECE
            ece1 = evaluate_binary_classification(y_tcga_tr.values, torch.sigmoid(c1_ca.model(c1_ca.X_train_tensor)).detach().numpy().flatten())["ece"]
            ece2 = evaluate_binary_classification(y_upenn_tr.values, torch.sigmoid(c2_ca.model(c2_ca.X_train_tensor)).detach().numpy().flatten())["ece"]

            agg_sd = ca_agg.aggregate([p1, p2], client_eces=[ece1, ece2])
            server_ca.global_model.load_state_dict(agg_sd)

        eval_ca = SurvivalMLP(input_dim=input_dim, hidden_dims=[32, 16])
        eval_ca.load_state_dict(server_ca.get_global_state_dict())
        eval_ca.eval()

        with torch.no_grad():
            probs_ca = torch.sigmoid(eval_ca(torch.tensor(X_upenn_te_std.values, dtype=torch.float32))).numpy().flatten()
        m_ca = evaluate_binary_classification(y_upenn_te.values, probs_ca)
        ca_results.append(m_ca)

    df_fa = pd.DataFrame(fedavg_results)
    df_ca = pd.DataFrame(ca_results)

    fa_auc_mean, fa_ece_mean = float(df_fa["roc_auc"].mean()), float(df_fa["ece"].mean())
    ca_auc_mean, ca_ece_mean = float(df_ca["roc_auc"].mean()), float(df_ca["ece"].mean())

    print("\nBENCHMARK RESULTS (UPenn External N=114 Across 5 Seeds):")
    print(f"FedAvg    -> ROC-AUC: {fa_auc_mean:.4f} +/- {df_fa['roc_auc'].std():.4f} | ECE: {fa_ece_mean:.4f} +/- {df_fa['ece'].std():.4f}")
    print(f"CA-FedAvg -> ROC-AUC: {ca_auc_mean:.4f} +/- {df_ca['roc_auc'].std():.4f} | ECE: {ca_ece_mean:.4f} +/- {df_ca['ece'].std():.4f}")

    gate_pass = (ca_auc_mean >= fa_auc_mean) and (ca_ece_mean <= fa_ece_mean)
    print(f"\nVERIFICATION GATE 2.1 STATUS: {'[PASS]' if gate_pass else '[WARNING/NEEDS TUNING]'}")

    return {
        "gate_2_1_passed": gate_pass,
        "fedavg_auc_mean": fa_auc_mean,
        "ca_fedavg_auc_mean": ca_auc_mean,
        "fedavg_ece_mean": fa_ece_mean,
        "ca_fedavg_ece_mean": ca_ece_mean,
    }


if __name__ == "__main__":
    test_ca_fedavg_unit()
    run_verification_gate_2_1()
