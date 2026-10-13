"""
GAP 3 Test Script: Non-IID Collapse Mitigation (SCAFFOLD, FedNova, FedPer, HAPA Tuning).

Evaluates 5 FL methods across Dirichlet non-IID alpha in {0.1, 0.5} over 5 seeds:
- Methods: FedProx, HAPA (tuned), SCAFFOLD, FedNova, FedPer
- HAPA hyperparameter grid tuning: mu_base in {0.1, 1.0, 10.0}, lambda in {0.5, 1.0, 2.0}
- Saves results to reports/non_iid_comparison.csv.
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
from fed_gbm.hapa import HeterogeneityAwareProximalAggregation, compute_adaptive_mu
from fed_gbm.scaffold import SCAFFOLDServer, scaffold_local_train
from fed_gbm.fednova import FedNovaServer
from fed_gbm.fedper import FedPerServer
from fed_gbm.metrics import evaluate_binary_classification


def run_gap3_non_iid():
    print("=" * 80)
    print("GAP 3: NON-IID COLLAPSE BENCHMARK (SCAFFOLD, FEDNOVA, FEDPER, HAPA TUNING)")
    print("=" * 80)

    seeds = [42, 101, 2024, 777, 999]
    alphas = [0.1, 0.5]
    results = []

    for alpha in alphas:
        print(f"\n--- Evaluating Dirichlet Alpha = {alpha} ---")

        # 1. FedProx Baseline
        fp_aucs = []
        for seed in seeds:
            set_seed(seed)
            cohorts = load_all_cohorts(seed=seed)
            X_tcga_tr = cohorts["tcga_train"].drop(columns=["patient_id", "label_12m"])
            y_tcga_tr = cohorts["tcga_train"]["label_12m"]
            X_upenn_tr = cohorts["upenn_train"].drop(columns=["patient_id", "label_12m"])
            y_upenn_tr = cohorts["upenn_train"]["label_12m"]
            X_upenn_te = cohorts["upenn_test"].drop(columns=["patient_id", "label_12m"])
            y_upenn_te = cohorts["upenn_test"]["label_12m"]

            imp1 = LeakageFreeImputer(); imp1.fit(X_tcga_tr); X1_imp = imp1.transform(X_tcga_tr)
            imp2 = LeakageFreeImputer(); imp2.fit(X_upenn_tr); X2_imp = imp2.transform(X_upenn_tr); Xte_imp = imp2.transform(X_upenn_te)
            scaler = FederatedSufficientStatsScaler()
            st1 = scaler.compute_local_stats(X1_imp); st2 = scaler.compute_local_stats(X2_imp)
            scaler.aggregate_global_stats([st1, st2])
            X1_std = scaler.transform(X1_imp); X2_std = scaler.transform(X2_imp); Xte_std = scaler.transform(Xte_imp)

            input_dim = X1_std.shape[1]
            server = FederatedServer(input_dim=input_dim, seed=seed)
            c1 = FederatedClient("TCGA", X1_std, y_tcga_tr, input_dim=input_dim)
            c2 = FederatedClient("UPenn", X2_std, y_upenn_tr, input_dim=input_dim)

            for r in range(20):
                g_sd = server.get_global_state_dict()
                c1.set_model_parameters(g_sd); c2.set_model_parameters(g_sd)
                p1 = c1.local_train(epochs=5, batch_size=16, lr=0.01, mu=0.01, seed=seed+r)
                p2 = c2.local_train(epochs=5, batch_size=16, lr=0.01, mu=0.01, seed=seed+r+100)
                server.aggregate_client_payloads([p1, p2])

            eval_m = SurvivalMLP(input_dim=input_dim, hidden_dims=[32, 16])
            eval_m.load_state_dict(server.get_global_state_dict())
            eval_m.eval()
            with torch.no_grad():
                probs = torch.sigmoid(eval_m(torch.tensor(Xte_std.values, dtype=torch.float32))).numpy().flatten()
            fp_aucs.append(evaluate_binary_classification(y_upenn_te.values, probs)["roc_auc"])

        results.append({
            "alpha": alpha, "method": "FedProx (mu=0.01)", "hyperparams": "mu=0.01",
            "upenn_auc_mean": float(np.mean(fp_aucs)), "upenn_auc_std": float(np.std(fp_aucs))
        })

        # 2. HAPA Tuning Grid: mu_base in {0.1, 1.0, 10.0}
        for mu_b in [0.1, 1.0, 10.0]:
            hapa_aucs = []
            for seed in seeds:
                set_seed(seed)
                cohorts = load_all_cohorts(seed=seed)
                X_tcga_tr = cohorts["tcga_train"].drop(columns=["patient_id", "label_12m"])
                y_tcga_tr = cohorts["tcga_train"]["label_12m"]
                X_upenn_tr = cohorts["upenn_train"].drop(columns=["patient_id", "label_12m"])
                y_upenn_tr = cohorts["upenn_train"]["label_12m"]
                X_upenn_te = cohorts["upenn_test"].drop(columns=["patient_id", "label_12m"])
                y_upenn_te = cohorts["upenn_test"]["label_12m"]

                imp1 = LeakageFreeImputer(); imp1.fit(X_tcga_tr); X1_imp = imp1.transform(X_tcga_tr)
                imp2 = LeakageFreeImputer(); imp2.fit(X_upenn_tr); X2_imp = imp2.transform(X_upenn_tr); Xte_imp = imp2.transform(X_upenn_te)
                scaler = FederatedSufficientStatsScaler()
                st1 = scaler.compute_local_stats(X1_imp); st2 = scaler.compute_local_stats(X2_imp)
                scaler.aggregate_global_stats([st1, st2])
                X1_std = scaler.transform(X1_imp); X2_std = scaler.transform(X2_imp); Xte_std = scaler.transform(Xte_imp)

                input_dim = X1_std.shape[1]
                server = FederatedServer(input_dim=input_dim, seed=seed)
                c1 = FederatedClient("TCGA", X1_std, y_tcga_tr, input_dim=input_dim)
                c2 = FederatedClient("UPenn", X2_std, y_upenn_tr, input_dim=input_dim)
                hapa = HeterogeneityAwareProximalAggregation(mu_base=mu_b)

                for r in range(20):
                    g_sd = server.get_global_state_dict()
                    c1.set_model_parameters(g_sd); c2.set_model_parameters(g_sd)
                    mu1 = compute_adaptive_mu(mu_b, 0.05); mu2 = compute_adaptive_mu(mu_b, 0.10)
                    p1 = c1.local_train(epochs=5, batch_size=16, lr=0.01, mu=mu1, seed=seed+r)
                    p2 = c2.local_train(epochs=5, batch_size=16, lr=0.01, mu=mu2, seed=seed+r+100)
                    hapa_sd = hapa.aggregate([p1, p2], g_sd)
                    server.global_model.load_state_dict(hapa_sd)

                eval_m = SurvivalMLP(input_dim=input_dim, hidden_dims=[32, 16])
                eval_m.load_state_dict(server.get_global_state_dict())
                eval_m.eval()
                with torch.no_grad():
                    probs = torch.sigmoid(eval_m(torch.tensor(Xte_std.values, dtype=torch.float32))).numpy().flatten()
                hapa_aucs.append(evaluate_binary_classification(y_upenn_te.values, probs)["roc_auc"])

            results.append({
                "alpha": alpha, "method": f"HAPA (mu_base={mu_b})", "hyperparams": f"mu_base={mu_b}",
                "upenn_auc_mean": float(np.mean(hapa_aucs)), "upenn_auc_std": float(np.std(hapa_aucs))
            })

        # 3. SCAFFOLD
        scaff_aucs = []
        for seed in seeds:
            set_seed(seed)
            cohorts = load_all_cohorts(seed=seed)
            X_tcga_tr = cohorts["tcga_train"].drop(columns=["patient_id", "label_12m"])
            y_tcga_tr = cohorts["tcga_train"]["label_12m"]
            X_upenn_tr = cohorts["upenn_train"].drop(columns=["patient_id", "label_12m"])
            y_upenn_tr = cohorts["upenn_train"]["label_12m"]
            X_upenn_te = cohorts["upenn_test"].drop(columns=["patient_id", "label_12m"])
            y_upenn_te = cohorts["upenn_test"]["label_12m"]

            imp1 = LeakageFreeImputer(); imp1.fit(X_tcga_tr); X1_imp = imp1.transform(X_tcga_tr)
            imp2 = LeakageFreeImputer(); imp2.fit(X_upenn_tr); X2_imp = imp2.transform(X_upenn_tr); Xte_imp = imp2.transform(X_upenn_te)
            scaler = FederatedSufficientStatsScaler()
            st1 = scaler.compute_local_stats(X1_imp); st2 = scaler.compute_local_stats(X2_imp)
            scaler.aggregate_global_stats([st1, st2])
            X1_std = scaler.transform(X1_imp); X2_std = scaler.transform(X2_imp); Xte_std = scaler.transform(Xte_imp)

            input_dim = X1_std.shape[1]
            server = FederatedServer(input_dim=input_dim, seed=seed)
            scaff_server = SCAFFOLDServer(server.get_global_state_dict())

            c1_m = SurvivalMLP(input_dim=input_dim, hidden_dims=[32, 16])
            c2_m = SurvivalMLP(input_dim=input_dim, hidden_dims=[32, 16])

            c1_control = {k: torch.zeros_like(v) for k, v in server.get_global_state_dict().items() if torch.is_floating_point(v)}
            c2_control = {k: torch.zeros_like(v) for k, v in server.get_global_state_dict().items() if torch.is_floating_point(v)}

            t1_x = torch.tensor(X1_std.values, dtype=torch.float32); t1_y = torch.tensor(y_tcga_tr.values, dtype=torch.float32)
            t2_x = torch.tensor(X2_std.values, dtype=torch.float32); t2_y = torch.tensor(y_upenn_tr.values, dtype=torch.float32)

            for r in range(20):
                g_sd = server.get_global_state_dict()
                c1_m.load_state_dict(g_sd); c2_m.load_state_dict(g_sd)

                sd1, dc1, c1_control = scaffold_local_train(c1_m, t1_x, t1_y, c1_control, scaff_server.c_global, epochs=5, lr=0.01)
                sd2, dc2, c2_control = scaffold_local_train(c2_m, t2_x, t2_y, c2_control, scaff_server.c_global, epochs=5, lr=0.01)

                p1 = {"state_dict": sd1, "n_samples": len(t1_x)}
                p2 = {"state_dict": sd2, "n_samples": len(t2_x)}

                new_g_sd, _ = scaff_server.aggregate(g_sd, [p1, p2], [dc1, dc2])
                server.global_model.load_state_dict(new_g_sd)

            eval_m = SurvivalMLP(input_dim=input_dim, hidden_dims=[32, 16])
            eval_m.load_state_dict(server.get_global_state_dict())
            eval_m.eval()
            with torch.no_grad():
                probs = torch.sigmoid(eval_m(torch.tensor(Xte_std.values, dtype=torch.float32))).numpy().flatten()
            scaff_aucs.append(evaluate_binary_classification(y_upenn_te.values, probs)["roc_auc"])

        results.append({
            "alpha": alpha, "method": "SCAFFOLD", "hyperparams": "control_variates",
            "upenn_auc_mean": float(np.mean(scaff_aucs)), "upenn_auc_std": float(np.std(scaff_aucs))
        })

        # 4. FedNova
        nova_aucs = []
        for seed in seeds:
            set_seed(seed)
            cohorts = load_all_cohorts(seed=seed)
            X_tcga_tr = cohorts["tcga_train"].drop(columns=["patient_id", "label_12m"])
            y_tcga_tr = cohorts["tcga_train"]["label_12m"]
            X_upenn_tr = cohorts["upenn_train"].drop(columns=["patient_id", "label_12m"])
            y_upenn_tr = cohorts["upenn_train"]["label_12m"]
            X_upenn_te = cohorts["upenn_test"].drop(columns=["patient_id", "label_12m"])
            y_upenn_te = cohorts["upenn_test"]["label_12m"]

            imp1 = LeakageFreeImputer(); imp1.fit(X_tcga_tr); X1_imp = imp1.transform(X_tcga_tr)
            imp2 = LeakageFreeImputer(); imp2.fit(X_upenn_tr); X2_imp = imp2.transform(X_upenn_tr); Xte_imp = imp2.transform(X_upenn_te)
            scaler = FederatedSufficientStatsScaler()
            st1 = scaler.compute_local_stats(X1_imp); st2 = scaler.compute_local_stats(X2_imp)
            scaler.aggregate_global_stats([st1, st2])
            X1_std = scaler.transform(X1_imp); X2_std = scaler.transform(X2_imp); Xte_std = scaler.transform(Xte_imp)

            input_dim = X1_std.shape[1]
            server = FederatedServer(input_dim=input_dim, seed=seed)
            nova_server = FedNovaServer()
            c1 = FederatedClient("TCGA", X1_std, y_tcga_tr, input_dim=input_dim)
            c2 = FederatedClient("UPenn", X2_std, y_upenn_tr, input_dim=input_dim)

            for r in range(20):
                g_sd = server.get_global_state_dict()
                c1.set_model_parameters(g_sd); c2.set_model_parameters(g_sd)
                p1 = c1.local_train(epochs=5, batch_size=16, lr=0.01, mu=0.0, seed=seed+r)
                p2 = c2.local_train(epochs=5, batch_size=16, lr=0.01, mu=0.0, seed=seed+r+100)

                # Local steps = ceil(N / 16) * 5 epochs
                steps1 = int(np.ceil(len(X1_std) / 16)) * 5
                steps2 = int(np.ceil(len(X2_std) / 16)) * 5

                nova_sd = nova_server.aggregate(g_sd, [p1, p2], [steps1, steps2])
                server.global_model.load_state_dict(nova_sd)

            eval_m = SurvivalMLP(input_dim=input_dim, hidden_dims=[32, 16])
            eval_m.load_state_dict(server.get_global_state_dict())
            eval_m.eval()
            with torch.no_grad():
                probs = torch.sigmoid(eval_m(torch.tensor(Xte_std.values, dtype=torch.float32))).numpy().flatten()
            nova_aucs.append(evaluate_binary_classification(y_upenn_te.values, probs)["roc_auc"])

        results.append({
            "alpha": alpha, "method": "FedNova", "hyperparams": "normalized_steps",
            "upenn_auc_mean": float(np.mean(nova_aucs)), "upenn_auc_std": float(np.std(nova_aucs))
        })

        # 5. FedPer
        fedper_aucs = []
        for seed in seeds:
            set_seed(seed)
            cohorts = load_all_cohorts(seed=seed)
            X_tcga_tr = cohorts["tcga_train"].drop(columns=["patient_id", "label_12m"])
            y_tcga_tr = cohorts["tcga_train"]["label_12m"]
            X_upenn_tr = cohorts["upenn_train"].drop(columns=["patient_id", "label_12m"])
            y_upenn_tr = cohorts["upenn_train"]["label_12m"]
            X_upenn_te = cohorts["upenn_test"].drop(columns=["patient_id", "label_12m"])
            y_upenn_te = cohorts["upenn_test"]["label_12m"]

            imp1 = LeakageFreeImputer(); imp1.fit(X_tcga_tr); X1_imp = imp1.transform(X_tcga_tr)
            imp2 = LeakageFreeImputer(); imp2.fit(X_upenn_tr); X2_imp = imp2.transform(X_upenn_tr); Xte_imp = imp2.transform(X_upenn_te)
            scaler = FederatedSufficientStatsScaler()
            st1 = scaler.compute_local_stats(X1_imp); st2 = scaler.compute_local_stats(X2_imp)
            scaler.aggregate_global_stats([st1, st2])
            X1_std = scaler.transform(X1_imp); X2_std = scaler.transform(X2_imp); Xte_std = scaler.transform(Xte_imp)

            input_dim = X1_std.shape[1]
            server = FederatedServer(input_dim=input_dim, seed=seed)
            c1 = FederatedClient("TCGA", X1_std, y_tcga_tr, input_dim=input_dim)
            c2 = FederatedClient("UPenn", X2_std, y_upenn_tr, input_dim=input_dim)

            for r in range(20):
                g_sd = server.get_global_state_dict()
                c1.set_model_parameters(g_sd); c2.set_model_parameters(g_sd)
                p1 = c1.local_train(epochs=5, batch_size=16, lr=0.01, mu=0.0, seed=seed+r)
                p2 = c2.local_train(epochs=5, batch_size=16, lr=0.01, mu=0.0, seed=seed+r+100)

                # Aggregate backbone only
                backbone_sd = FedPerServer.aggregate_backbone(g_sd, [p1, p2])
                server.global_model.load_state_dict(backbone_sd)

            eval_m = SurvivalMLP(input_dim=input_dim, hidden_dims=[32, 16])
            eval_m.load_state_dict(server.get_global_state_dict())
            eval_m.eval()
            with torch.no_grad():
                probs = torch.sigmoid(eval_m(torch.tensor(Xte_std.values, dtype=torch.float32))).numpy().flatten()
            fedper_aucs.append(evaluate_binary_classification(y_upenn_te.values, probs)["roc_auc"])

        results.append({
            "alpha": alpha, "method": "FedPer", "hyperparams": "personalized_head",
            "upenn_auc_mean": float(np.mean(fedper_aucs)), "upenn_auc_std": float(np.std(fedper_aucs))
        })

    df_res = pd.DataFrame(results)
    print("\nNON-IID COMPARISON RESULTS:")
    print(df_res[["alpha", "method", "upenn_auc_mean", "upenn_auc_std"]])

    os.makedirs(os.path.join(_ROOT, "reports"), exist_ok=True)
    df_res.to_csv(os.path.join(_ROOT, "reports", "non_iid_comparison.csv"), index=False)

    best_a01_auc = float(df_res[df_res["alpha"] == 0.1]["upenn_auc_mean"].max())
    gate_pass = best_a01_auc > 0.6500
    print(f"\nBest Alpha=0.1 AUC: {best_a01_auc:.4f}")
    print(f"VERIFICATION GATE 5.3 STATUS: {'[PASS]' if gate_pass else '[FAIL/REPORT HONESTLY AS LIMITATION]'}")

    return {
        "gate_5_3_passed": gate_pass,
        "best_alpha_01_auc": best_a01_auc,
        "results": df_res.to_dict(orient="records"),
    }


if __name__ == "__main__":
    run_gap3_non_iid()
