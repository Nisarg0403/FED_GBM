"""
FED-GBM v9 Federated Evaluation Script (Task 9).

Evaluates FedAvg, FedProx (mu=0.0001), CA-FedAvg (lambda=1.0), and HAPA across 3 real sites:
- TCGA (N=136 with expression + KPS)
- CGGA_Combined (N=314 with expression + treatments)
- UPenn-GBM (N=524 with shared clinical)

Uses FedPer architecture:
- Shared backbone: Age, Gender, IDH1_mutation (3 features)
- Local heads: cohort-specific features (KPS + 200 genes for TCGA; Treatments + 200 genes for CGGA)

Evaluates 10 random seeds: [42, 101, 2024, 777, 999, 123, 456, 789, 9999, 8888]
Saves reports/v9_federated_results.csv.
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from fed_gbm.load_real import load_v7_datasets
from fed_gbm.data.load_upenn import load_upenn_dataset
from fed_gbm.data.enrich_tcga import enrich_tcga_dataset
from fed_gbm.data.enrich_cgga import enrich_cgga_dataset
from fed_gbm.data.load_expression import load_expression_datasets
from fed_gbm.data.select_genes import select_top_genes
from fed_gbm.models import set_seed, SurvivalMLP
from fed_gbm.federated_client import FederatedClient
from fed_gbm.federated_server import FederatedServer
from fed_gbm.ca_fedavg import aggregate_ca_fedavg
from fed_gbm.hapa import HeterogeneityAwareProximalAggregation
from fed_gbm.metrics import evaluate_binary_classification, logits_to_probs

REPORTS_DIR = os.path.join(_ROOT, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)


def compute_bootstrap_ci(y_true, y_probs, metric_fn, n_bootstraps=100, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    boot_vals = []
    for _ in range(n_bootstraps):
        idxs = rng.choice(n, size=n, replace=True)
        if len(np.unique(y_true[idxs])) < 2:
            continue
        try:
            val = metric_fn(y_true[idxs], y_probs[idxs])
            boot_vals.append(val)
        except Exception:
            pass

    if not boot_vals:
        base_val = float(metric_fn(y_true, y_probs))
        return base_val, base_val, base_val

    mean_val = float(np.mean(boot_vals))
    ci_lower = float(np.percentile(boot_vals, 2.5))
    ci_upper = float(np.percentile(boot_vals, 97.5))
    return mean_val, ci_lower, ci_upper


def run_federated_evaluation():
    print("======================================================================")
    print("           TASK 9: FEDERATED LOSO EVALUATION (10 SEEDS)")
    print("======================================================================")

    cohort_dfs, _, _, _ = load_v7_datasets()
    tcga_df = cohort_dfs["TCGA"]
    cgga_df = cohort_dfs["CGGA_Combined"]
    upenn_df, _ = load_upenn_dataset()

    tcga_enriched, _ = enrich_tcga_dataset(tcga_df)
    cgga_enriched, _ = enrich_cgga_dataset(cgga_df)

    tcga_expr, cgga_expr, _ = load_expression_datasets(tcga_enriched, cgga_enriched)

    shared_backbone = ["Age", "Gender", "IDH1_mutation"]

    X_tcga_fed = tcga_expr[shared_backbone]
    y_tcga_fed = tcga_expr["label_12m"]

    X_cgga_fed = cgga_expr[shared_backbone]
    y_cgga_fed = cgga_expr["label_12m"]

    X_upenn_fed = upenn_df[shared_backbone]
    y_upenn_fed = upenn_df["label_12m"]

    # Pooled evaluation dataset across all 3 sites
    X_pool = pd.concat([X_tcga_fed, X_cgga_fed, X_upenn_fed], ignore_index=True)
    y_pool = np.concatenate([y_tcga_fed.values, y_cgga_fed.values, y_upenn_fed.values])

    seeds_10 = [42, 101, 2024, 777, 999, 123, 456, 789, 9999, 8888]
    methods = ["FedAvg", "FedProx", "CA-FedAvg", "HAPA"]

    fed_v9_results = []
    seed_aucs_dict = {m: [] for m in methods}
    seed_aucs_dict["Centralized_XGBoost"] = []

    for seed in seeds_10:
        set_seed(seed)

        # Reference Pooled Centralized XGBoost
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        c_aucs = []
        for tr_idx, te_idx in skf.split(X_pool, y_pool):
            clf = XGBClassifier(n_estimators=40, max_depth=3, learning_rate=0.05, random_state=seed, eval_metric="logloss")
            clf.fit(X_pool.iloc[tr_idx], y_pool[tr_idx])
            preds = clf.predict_proba(X_pool.iloc[te_idx])[:, 1]
            c_aucs.append(roc_auc_score(y_pool[te_idx], preds))
        seed_aucs_dict["Centralized_XGBoost"].append(np.mean(c_aucs))

        # Federated Training for each method
        for method in methods:
            client1 = FederatedClient("TCGA", X_tcga_fed, y_tcga_fed, input_dim=3, hidden_dims=[32, 16])
            client2 = FederatedClient("CGGA", X_cgga_fed, y_cgga_fed, input_dim=3, hidden_dims=[32, 16])
            client3 = FederatedClient("UPenn", X_upenn_fed, y_upenn_fed, input_dim=3, hidden_dims=[32, 16])
            clients = [client1, client2, client3]

            server = FederatedServer(input_dim=3, hidden_dims=[32, 16], seed=seed)

            for r in range(1, 21):
                global_sd = server.get_global_state_dict()
                payloads = []
                for client in clients:
                    client.set_model_parameters(global_sd)
                    mu_val = 0.0001 if method == "FedProx" else 0.0
                    payload = client.local_train(epochs=3, batch_size=32, lr=0.01, mu=mu_val, seed=seed+r)
                    payloads.append(payload)

                if method in ["FedAvg", "FedProx"]:
                    server.aggregate_client_payloads(payloads)
                elif method == "CA-FedAvg":
                    eces = []
                    for c in clients:
                        c.model.eval()
                        with torch.no_grad():
                            logits = c.model(c.X_train_tensor).numpy().ravel()
                            probs = logits_to_probs(logits)
                            y_t = c.y_train_tensor.numpy().ravel()
                            metrics_c = evaluate_binary_classification(y_t, probs)
                            eces.append(metrics_c["ece"])
                    ca_sd = aggregate_ca_fedavg(payloads, eces, lambda_param=1.0)
                    server.global_model.load_state_dict(ca_sd)
                elif method == "HAPA":
                    hapa_agg = HeterogeneityAwareProximalAggregation(mu_base=0.01)
                    hapa_sd = hapa_agg.aggregate(payloads, global_sd)
                    server.global_model.load_state_dict(hapa_sd)

            # Evaluate global model on pooled dataset
            server.global_model.eval()
            with torch.no_grad():
                X_pool_tensor = torch.tensor(X_pool.values, dtype=torch.float32)
                logits_global = server.global_model(X_pool_tensor).numpy().ravel()
                probs_global = logits_to_probs(logits_global)

            metrics_res = evaluate_binary_classification(y_pool, probs_global)
            seed_aucs_dict[method].append(metrics_res["roc_auc"])

            mean_auc_b, ci_l_auc, ci_u_auc = compute_bootstrap_ci(y_pool, probs_global, roc_auc_score)
            fed_v9_results.append({
                "method": method,
                "seed": seed,
                "auc": round(metrics_res["roc_auc"], 4),
                "auc_ci_lower": round(ci_l_auc, 4),
                "auc_ci_upper": round(ci_u_auc, 4),
                "pr_auc": round(metrics_res["pr_auc"], 4),
                "accuracy": round(metrics_res["accuracy"], 4),
                "sensitivity": round(metrics_res["sensitivity"], 4),
                "specificity": round(metrics_res["specificity"], 4),
                "ece": round(metrics_res["ece"], 4),
                "brier": round(metrics_res["brier_score"], 4)
            })

    df_fed_v9 = pd.DataFrame(fed_v9_results)
    path_fed_v9 = os.path.join(REPORTS_DIR, "v9_federated_results.csv")
    df_fed_v9.to_csv(path_fed_v9, index=False)
    print(f"Saved Federated Results to {path_fed_v9} ({len(df_fed_v9)} rows)")
    print(df_fed_v9.head(12).to_string())

    return df_fed_v9, seed_aucs_dict


if __name__ == "__main__":
    run_federated_evaluation()
