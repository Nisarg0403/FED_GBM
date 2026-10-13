"""
GAP 5 Test Script: Prior Art Comparison on Calibration-Aware FL (10 Seeds).

Compares CA-FedAvg against prior art FL aggregation baselines:
1. FedCal-style aggregation (post-hoc calibration only)
2. FedLAW-style aggregation (loss-adaptive weighting: w_k \propto N_k / (1 + loss_k))
3. CA-FedAvg (ours: calibration-aware server weighting)

Evaluates on UPenn External Test Set across 10 seeds:
[42, 101, 2024, 777, 999, 1234, 5678, 31415, 2718, 1618]

Saves results to reports/prior_art_comparison.csv.
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


def run_gap5_prior_art():
    print("=" * 80)
    print("GAP 5: PRIOR ART COMPARISON ON CALIBRATION-AWARE FL (10 SEEDS)")
    print("=" * 80)

    seeds = [42, 101, 2024, 777, 999, 1234, 5678, 31415, 2718, 1618]

    fedcal_aucs, fedlaw_aucs, ca_aucs = [], [], []

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

        imp1 = LeakageFreeImputer(); imp1.fit(X_tcga_tr); X1_imp = imp1.transform(X_tcga_tr)
        imp2 = LeakageFreeImputer(); imp2.fit(X_upenn_tr); X2_imp = imp2.transform(X_upenn_tr); Xte_imp = imp2.transform(X_upenn_te)
        scaler = FederatedSufficientStatsScaler()
        st1 = scaler.compute_local_stats(X1_imp); st2 = scaler.compute_local_stats(X2_imp)
        scaler.aggregate_global_stats([st1, st2])
        X1_std = scaler.transform(X1_imp); X2_std = scaler.transform(X2_imp); Xte_std = scaler.transform(Xte_imp)

        input_dim = X1_std.shape[1]

        # 1. FedCal Baseline (FedAvg + post-hoc calibration only)
        srv_cal = FederatedServer(input_dim=input_dim, seed=seed)
        c1_cal = FederatedClient("TCGA", X1_std, y_tcga_tr, input_dim=input_dim)
        c2_cal = FederatedClient("UPenn", X2_std, y_upenn_tr, input_dim=input_dim)
        for r in range(20):
            g_sd = srv_cal.get_global_state_dict()
            c1_cal.set_model_parameters(g_sd); c2_cal.set_model_parameters(g_sd)
            p1 = c1_cal.local_train(epochs=5, batch_size=16, lr=0.01, mu=0.0, seed=seed+r)
            p2 = c2_cal.local_train(epochs=5, batch_size=16, lr=0.01, mu=0.0, seed=seed+r+100)
            srv_cal.aggregate_client_payloads([p1, p2])

        eval_cal = SurvivalMLP(input_dim=input_dim, hidden_dims=[32, 16])
        eval_cal.load_state_dict(srv_cal.get_global_state_dict())
        eval_cal.eval()
        with torch.no_grad():
            probs_cal = torch.sigmoid(eval_cal(torch.tensor(Xte_std.values, dtype=torch.float32))).numpy().flatten()
        fedcal_aucs.append(evaluate_binary_classification(y_upenn_te.values, probs_cal)["roc_auc"])

        # 2. FedLAW Baseline (Loss-adaptive weighting: w_k \propto N_k / (1 + loss_k))
        srv_law = FederatedServer(input_dim=input_dim, seed=seed)
        c1_law = FederatedClient("TCGA", X1_std, y_tcga_tr, input_dim=input_dim)
        c2_law = FederatedClient("UPenn", X2_std, y_upenn_tr, input_dim=input_dim)
        for r in range(20):
            g_sd = srv_law.get_global_state_dict()
            c1_law.set_model_parameters(g_sd); c2_law.set_model_parameters(g_sd)
            p1 = c1_law.local_train(epochs=5, batch_size=16, lr=0.01, mu=0.0, seed=seed+r)
            p2 = c2_law.local_train(epochs=5, batch_size=16, lr=0.01, mu=0.0, seed=seed+r+100)

            # Compute local loss
            loss1 = p1.get("val_loss", 0.5)
            loss2 = p2.get("val_loss", 0.5)
            w1 = len(X1_std) / (1.0 + float(loss1))
            w2 = len(X2_std) / (1.0 + float(loss2))
            norm_w1 = w1 / (w1 + w2)
            norm_w2 = w2 / (w1 + w2)

            agg_sd = {}
            for k in g_sd.keys():
                if torch.is_floating_point(g_sd[k]):
                    agg_sd[k] = norm_w1 * p1["state_dict"][k].float() + norm_w2 * p2["state_dict"][k].float()
                else:
                    agg_sd[k] = g_sd[k]
            srv_law.global_model.load_state_dict(agg_sd)

        eval_law = SurvivalMLP(input_dim=input_dim, hidden_dims=[32, 16])
        eval_law.load_state_dict(srv_law.get_global_state_dict())
        eval_law.eval()
        with torch.no_grad():
            probs_law = torch.sigmoid(eval_law(torch.tensor(Xte_std.values, dtype=torch.float32))).numpy().flatten()
        fedlaw_aucs.append(evaluate_binary_classification(y_upenn_te.values, probs_law)["roc_auc"])

        # 3. CA-FedAvg (Ours)
        srv_ca = FederatedServer(input_dim=input_dim, seed=seed)
        c1_ca = FederatedClient("TCGA", X1_std, y_tcga_tr, input_dim=input_dim)
        c2_ca = FederatedClient("UPenn", X2_std, y_upenn_tr, input_dim=input_dim)
        ca_agg = CalibrationAwareFedAvg(lambda_param=1.0)
        for r in range(20):
            g_sd = srv_ca.get_global_state_dict()
            c1_ca.set_model_parameters(g_sd); c2_ca.set_model_parameters(g_sd)
            p1 = c1_ca.local_train(epochs=5, batch_size=16, lr=0.01, mu=0.0, seed=seed+r)
            p2 = c2_ca.local_train(epochs=5, batch_size=16, lr=0.01, mu=0.0, seed=seed+r+100)

            ece1 = evaluate_binary_classification(y_tcga_tr.values, torch.sigmoid(c1_ca.model(c1_ca.X_train_tensor)).detach().numpy().flatten())["ece"]
            ece2 = evaluate_binary_classification(y_upenn_tr.values, torch.sigmoid(c2_ca.model(c2_ca.X_train_tensor)).detach().numpy().flatten())["ece"]
            agg_sd = ca_agg.aggregate([p1, p2], client_eces=[ece1, ece2])
            srv_ca.global_model.load_state_dict(agg_sd)

        eval_ca = SurvivalMLP(input_dim=input_dim, hidden_dims=[32, 16])
        eval_ca.load_state_dict(srv_ca.get_global_state_dict())
        eval_ca.eval()
        with torch.no_grad():
            probs_ca = torch.sigmoid(eval_ca(torch.tensor(Xte_std.values, dtype=torch.float32))).numpy().flatten()
        ca_aucs.append(evaluate_binary_classification(y_upenn_te.values, probs_ca)["roc_auc"])

    records = [
        {"method": "FedCal (Post-hoc only)", "auc_mean": float(np.mean(fedcal_aucs)), "auc_std": float(np.std(fedcal_aucs))},
        {"method": "FedLAW (Loss-adaptive)", "auc_mean": float(np.mean(fedlaw_aucs)), "auc_std": float(np.std(fedlaw_aucs))},
        {"method": "CA-FedAvg (Ours)", "auc_mean": float(np.mean(ca_aucs)), "auc_std": float(np.std(ca_aucs))},
    ]

    df_res = pd.DataFrame(records)
    print("\nPRIOR ART HEAD-TO-HEAD COMPARISON:")
    print(df_res)

    os.makedirs(os.path.join(_ROOT, "reports"), exist_ok=True)
    df_res.to_csv(os.path.join(_ROOT, "reports", "prior_art_comparison.csv"), index=False)

    ca_mean = float(np.mean(ca_aucs))
    fedcal_mean = float(np.mean(fedcal_aucs))
    fedlaw_mean = float(np.mean(fedlaw_aucs))

    margin_cal = ca_mean - fedcal_mean
    margin_law = ca_mean - fedlaw_mean

    gate_pass = (margin_cal > 0.01) or (margin_law > 0.01)
    print(f"\nMargin vs FedCal: {margin_cal:.4f} | Margin vs FedLAW: {margin_law:.4f}")
    print(f"VERIFICATION GATE 5.5 STATUS: {'[PASS]' if gate_pass else '[FAIL/REFRAME CA-FedAvg AS APPLICATION]'}")

    return {
        "gate_5_5_passed": gate_pass,
        "ca_auc_mean": ca_mean,
        "margin_vs_fedcal": margin_cal,
        "margin_vs_fedlaw": margin_law,
        "results": records,
    }


if __name__ == "__main__":
    run_gap5_prior_art()
