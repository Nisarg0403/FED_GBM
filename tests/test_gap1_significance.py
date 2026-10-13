"""
GAP 1 Test Script: Statistical Significance Testing Across 10 Seeds.

Evaluates 10 seeds: [42, 101, 2024, 777, 999, 1234, 5678, 31415, 2718, 1618]
Models evaluated on UPenn External Test Set:
1. FedAvg
2. FedProx (mu=0.01)
3. CA-FedAvg (lambda=1.0)
4. HAPA (mu_base=0.01)

Performs paired t-tests, Cohen's d, and Bonferroni corrections across comparisons.
Saves results to reports/significance_tests.csv.
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
from fed_gbm.hapa import HeterogeneityAwareProximalAggregation, compute_adaptive_mu
from fed_gbm.metrics import evaluate_binary_classification
from fed_gbm.stats import paired_ttest, wilcoxon_test, bootstrap_ci, bonferroni_correction


def run_gap1_significance():
    print("=" * 80)
    print("GAP 1: STATISTICAL SIGNIFICANCE TESTING (10 SEEDS)")
    print("=" * 80)

    seeds = [42, 101, 2024, 777, 999, 1234, 5678, 31415, 2718, 1618]

    fa_aucs, fp_aucs, ca_aucs, hapa_aucs = [], [], [], []
    fa_eces, fp_eces, ca_eces, hapa_eces = [], [], [], []

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

        # 1. FedAvg
        srv_fa = FederatedServer(input_dim=input_dim, seed=seed)
        c1_fa = FederatedClient("TCGA", X_tcga_tr_std, y_tcga_tr, input_dim=input_dim)
        c2_fa = FederatedClient("UPenn", X_upenn_tr_std, y_upenn_tr, input_dim=input_dim)
        for r in range(20):
            g_sd = srv_fa.get_global_state_dict()
            c1_fa.set_model_parameters(g_sd)
            c2_fa.set_model_parameters(g_sd)
            p1 = c1_fa.local_train(epochs=5, batch_size=16, lr=0.01, mu=0.0, seed=seed + r)
            p2 = c2_fa.local_train(epochs=5, batch_size=16, lr=0.01, mu=0.0, seed=seed + r + 100)
            srv_fa.aggregate_client_payloads([p1, p2])

        eval_fa = SurvivalMLP(input_dim=input_dim, hidden_dims=[32, 16])
        eval_fa.load_state_dict(srv_fa.get_global_state_dict())
        eval_fa.eval()
        with torch.no_grad():
            probs_fa = torch.sigmoid(eval_fa(torch.tensor(X_upenn_te_std.values, dtype=torch.float32))).numpy().flatten()
        m_fa = evaluate_binary_classification(y_upenn_te.values, probs_fa)
        fa_aucs.append(m_fa["roc_auc"])
        fa_eces.append(m_fa["ece"])

        # 2. FedProx (mu=0.01)
        srv_fp = FederatedServer(input_dim=input_dim, seed=seed)
        c1_fp = FederatedClient("TCGA", X_tcga_tr_std, y_tcga_tr, input_dim=input_dim)
        c2_fp = FederatedClient("UPenn", X_upenn_tr_std, y_upenn_tr, input_dim=input_dim)
        for r in range(20):
            g_sd = srv_fp.get_global_state_dict()
            c1_fp.set_model_parameters(g_sd)
            c2_fp.set_model_parameters(g_sd)
            p1 = c1_fp.local_train(epochs=5, batch_size=16, lr=0.01, mu=0.01, seed=seed + r)
            p2 = c2_fp.local_train(epochs=5, batch_size=16, lr=0.01, mu=0.01, seed=seed + r + 100)
            srv_fp.aggregate_client_payloads([p1, p2])

        eval_fp = SurvivalMLP(input_dim=input_dim, hidden_dims=[32, 16])
        eval_fp.load_state_dict(srv_fp.get_global_state_dict())
        eval_fp.eval()
        with torch.no_grad():
            probs_fp = torch.sigmoid(eval_fp(torch.tensor(X_upenn_te_std.values, dtype=torch.float32))).numpy().flatten()
        m_fp = evaluate_binary_classification(y_upenn_te.values, probs_fp)
        fp_aucs.append(m_fp["roc_auc"])
        fp_eces.append(m_fp["ece"])

        # 3. CA-FedAvg
        srv_ca = FederatedServer(input_dim=input_dim, seed=seed)
        c1_ca = FederatedClient("TCGA", X_tcga_tr_std, y_tcga_tr, input_dim=input_dim)
        c2_ca = FederatedClient("UPenn", X_upenn_tr_std, y_upenn_tr, input_dim=input_dim)
        ca_agg = CalibrationAwareFedAvg(lambda_param=1.0)
        for r in range(20):
            g_sd = srv_ca.get_global_state_dict()
            c1_ca.set_model_parameters(g_sd)
            c2_ca.set_model_parameters(g_sd)
            p1 = c1_ca.local_train(epochs=5, batch_size=16, lr=0.01, mu=0.0, seed=seed + r)
            p2 = c2_ca.local_train(epochs=5, batch_size=16, lr=0.01, mu=0.0, seed=seed + r + 100)

            ece1 = evaluate_binary_classification(y_tcga_tr.values, torch.sigmoid(c1_ca.model(c1_ca.X_train_tensor)).detach().numpy().flatten())["ece"]
            ece2 = evaluate_binary_classification(y_upenn_tr.values, torch.sigmoid(c2_ca.model(c2_ca.X_train_tensor)).detach().numpy().flatten())["ece"]
            agg_sd = ca_agg.aggregate([p1, p2], client_eces=[ece1, ece2])
            srv_ca.global_model.load_state_dict(agg_sd)

        eval_ca = SurvivalMLP(input_dim=input_dim, hidden_dims=[32, 16])
        eval_ca.load_state_dict(srv_ca.get_global_state_dict())
        eval_ca.eval()
        with torch.no_grad():
            probs_ca = torch.sigmoid(eval_ca(torch.tensor(X_upenn_te_std.values, dtype=torch.float32))).numpy().flatten()
        m_ca = evaluate_binary_classification(y_upenn_te.values, probs_ca)
        ca_aucs.append(m_ca["roc_auc"])
        ca_eces.append(m_ca["ece"])

        # 4. HAPA
        srv_hapa = FederatedServer(input_dim=input_dim, seed=seed)
        c1_hapa = FederatedClient("TCGA", X_tcga_tr_std, y_tcga_tr, input_dim=input_dim)
        c2_hapa = FederatedClient("UPenn", X_upenn_tr_std, y_upenn_tr, input_dim=input_dim)
        hapa_agg = HeterogeneityAwareProximalAggregation(mu_base=0.01)
        for r in range(20):
            g_sd = srv_hapa.get_global_state_dict()
            c1_hapa.set_model_parameters(g_sd)
            c2_hapa.set_model_parameters(g_sd)
            mu1 = compute_adaptive_mu(0.01, 0.05)
            mu2 = compute_adaptive_mu(0.01, 0.10)
            p1 = c1_hapa.local_train(epochs=5, batch_size=16, lr=0.01, mu=mu1, seed=seed + r)
            p2 = c2_hapa.local_train(epochs=5, batch_size=16, lr=0.01, mu=mu2, seed=seed + r + 100)
            agg_sd = hapa_agg.aggregate([p1, p2], g_sd)
            srv_hapa.global_model.load_state_dict(agg_sd)

        eval_hapa = SurvivalMLP(input_dim=input_dim, hidden_dims=[32, 16])
        eval_hapa.load_state_dict(srv_hapa.get_global_state_dict())
        eval_hapa.eval()
        with torch.no_grad():
            probs_hapa = torch.sigmoid(eval_hapa(torch.tensor(X_upenn_te_std.values, dtype=torch.float32))).numpy().flatten()
        m_hapa = evaluate_binary_classification(y_upenn_te.values, probs_hapa)
        hapa_aucs.append(m_hapa["roc_auc"])
        hapa_eces.append(m_hapa["ece"])

    # Model Pair Comparisons required
    pairs = [
        ("CA-FedAvg", "FedAvg", ca_aucs, fa_aucs, "roc_auc"),
        ("CA-FedAvg", "FedProx", ca_aucs, fp_aucs, "roc_auc"),
        ("HAPA", "FedAvg", hapa_aucs, fa_aucs, "roc_auc"),
        ("HAPA", "FedProx", hapa_aucs, fp_aucs, "roc_auc"),
        ("CA-FedAvg", "HAPA", ca_aucs, hapa_aucs, "roc_auc"),
        ("CA-FedAvg", "FedAvg", ca_eces, fa_eces, "ece"),
        ("CA-FedAvg", "FedProx", ca_eces, fp_eces, "ece"),
        ("HAPA", "FedAvg", hapa_eces, fa_eces, "ece"),
        ("HAPA", "FedProx", hapa_eces, fp_eces, "ece"),
        ("CA-FedAvg", "HAPA", ca_eces, hapa_eces, "ece"),
    ]

    records = []
    raw_ps = []

    for name_a, name_b, vals_a, vals_b, metric in pairs:
        mean_a = float(np.mean(vals_a))
        mean_b = float(np.mean(vals_b))
        delta = mean_a - mean_b
        t_stat, p_val, cohens_d = paired_ttest(vals_a, vals_b)
        raw_ps.append(p_val)

        records.append({
            "comparison": f"{name_a} vs {name_b}",
            "metric": metric,
            "mean_a": mean_a,
            "mean_b": mean_b,
            "delta": delta,
            "t_stat": t_stat,
            "p_value": p_val,
            "cohens_d": cohens_d,
        })

    bonf_ps = bonferroni_correction(raw_ps, n_comparisons=len(pairs))

    for rec, bonf_p in zip(records, bonf_ps):
        rec["bonferroni_p"] = bonf_p
        rec["significant_at_0.05"] = bonf_p < 0.05

    df_res = pd.DataFrame(records)
    print("\nSTATISTICAL SIGNIFICANCE RESULTS:")
    print(df_res[["comparison", "metric", "mean_a", "mean_b", "delta", "p_value", "bonferroni_p", "significant_at_0.05"]])

    os.makedirs(os.path.join(_ROOT, "reports"), exist_ok=True)
    df_res.to_csv(os.path.join(_ROOT, "reports", "significance_tests.csv"), index=False)

    any_sig = any(df_res["significant_at_0.05"])
    print(f"\nVERIFICATION GATE 5.1 STATUS: {'[PASS]' if any_sig else '[FAIL/REPORT HONESTLY]'}")

    return {
        "gate_5_1_passed": any_sig,
        "results": df_res.to_dict(orient="records"),
    }


if __name__ == "__main__":
    run_gap1_significance()
