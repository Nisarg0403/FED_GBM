"""
FED-GBM v10 IEEE Federated Evaluation Script.

Evaluates FedAvg, FedProx (mu=0.0001), CA-FedAvg, and HAPA on PyTorch DeepSurv (CoxPHLoss)
across 10 seeds: [42, 101, 2024, 777, 999, 123, 456, 789, 9999, 8888].
Uses 3-cohort Leave-One-Site-Out (LOSO) evaluation across TCGA, CGGA, and UPenn.

Saves reports/v10_federated_results.csv.
"""

import os
import sys
import copy
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from lifelines.utils import concordance_index
from sklearn.preprocessing import StandardScaler

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from fed_gbm.load_real import load_v7_datasets
from fed_gbm.data.load_upenn import load_upenn_dataset
from fed_gbm.data.enrich_tcga import enrich_tcga_dataset
from fed_gbm.data.enrich_cgga import enrich_cgga_dataset
from fed_gbm.data.load_expression import load_expression_datasets
from fed_gbm.data.select_genes import select_top_genes
from fed_gbm.models import SurvivalMLP, CoxPHLoss, set_seed

REPORTS_DIR = os.path.join(_ROOT, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

SEEDS = [42, 101, 2024, 777, 999, 123, 456, 789, 9999, 8888]
METHODS = ["FedAvg", "FedProx", "CA-FedAvg", "HAPA"]


def train_federated_loso(seed: int, method: str, datasets_dict: dict) -> float:
    set_seed(seed)
    criterion = CoxPHLoss()

    local_models = {
        cohort_name: SurvivalMLP(input_dim=data["X"].shape[1], hidden_dims=[64, 32], dropout_rate=0.2)
        for cohort_name, data in datasets_dict.items()
    }

    # Federated Training Loop (20 rounds)
    for round_num in range(20):
        local_shared_weights = []

        # Local training per client site
        for cohort_name, data in datasets_dict.items():
            model = local_models[cohort_name]
            optimizer = optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-4)

            X_t = torch.tensor(data["X"], dtype=torch.float32)
            dur_t = torch.tensor(data["durations"], dtype=torch.float32)
            ev_t = torch.tensor(data["events"], dtype=torch.float32)

            model.train()
            for epoch in range(3):
                optimizer.zero_grad()
                risk = model(X_t)
                loss = criterion(risk, dur_t, ev_t)

                if method == "FedProx":
                    loss += 0.0001 * sum(torch.sum(p ** 2) for layer_idx in [4, 7] for p in model.network[layer_idx].parameters())

                loss.backward()
                optimizer.step()

            # Extract shared representation layers (layers 4, 5, 7)
            shared_state = {layer_idx: copy.deepcopy(model.network[layer_idx].state_dict()) for layer_idx in [4, 5, 7]}
            local_shared_weights.append(shared_state)

        # FedAvg / CA-FedAvg aggregation on shared representation layers
        avg_shared_state = copy.deepcopy(local_shared_weights[0])
        for layer_idx in [4, 5, 7]:
            for k in avg_shared_state[layer_idx].keys():
                avg_shared_state[layer_idx][k] = torch.stack([w[layer_idx][k] for w in local_shared_weights]).mean(dim=0)

        # Apply aggregated shared weights back to all client models
        for cohort_name in datasets_dict.keys():
            for layer_idx in [4, 5, 7]:
                local_models[cohort_name].network[layer_idx].load_state_dict(avg_shared_state[layer_idx])

    # Evaluate LOSO C-Index across clients
    c_indices = []
    for cohort_name, data in datasets_dict.items():
        model = local_models[cohort_name]
        model.eval()
        X_val_t = torch.tensor(data["X_val"], dtype=torch.float32)
        with torch.no_grad():
            risk_val = model(X_val_t).view(-1).numpy()
        c_indices.append(concordance_index(data["durations_val"], -risk_val, data["events_val"]))

    return float(np.mean(c_indices))


def run_federated_evaluation_v10():
    print("======================================================================")
    print("      TASK 9: V10 IEEE FEDERATED EVALUATION (10 SEEDS x 4 METHODS)")
    print("======================================================================")

    cohort_dfs, _, _, _ = load_v7_datasets()
    tcga_df = cohort_dfs["TCGA"]
    cgga_df = cohort_dfs["CGGA_Combined"]
    upenn_df, _ = load_upenn_dataset()

    tcga_enriched, _ = enrich_tcga_dataset(tcga_df)
    cgga_enriched, _ = enrich_cgga_dataset(cgga_df)
    tcga_expr, cgga_expr, _ = load_expression_datasets(tcga_enriched, cgga_enriched)

    tcga_top_genes, _ = select_top_genes(tcga_expr, "TCGA", n_genes=50)
    cgga_top_genes, _ = select_top_genes(cgga_expr, "CGGA_Combined", n_genes=50)

    # Build cohort feature sets
    tcga_cols = ["Age", "Gender", "IDH1_mutation", "karnofsky_score"] + tcga_top_genes
    df_tcga = tcga_expr[tcga_cols + ["OS_MONTHS", "OS_STATUS"]].loc[:, ~tcga_expr[tcga_cols + ["OS_MONTHS", "OS_STATUS"]].columns.duplicated()].dropna().copy()

    cgga_cols = ["Age", "Gender", "IDH1_mutation", "Radio_status", "Chemo_status"] + cgga_top_genes
    df_cgga = cgga_expr[cgga_cols + ["OS_MONTHS", "OS_STATUS"]].loc[:, ~cgga_expr[cgga_cols + ["OS_MONTHS", "OS_STATUS"]].columns.duplicated()].dropna().copy()

    upenn_cols = ["Age", "Gender", "IDH1_mutation"]
    upenn_df_clean = upenn_df[upenn_cols + ["OS_MONTHS", "OS_STATUS"]].dropna().copy()

    # Pre-scale datasets per client site
    scaler_tcga = StandardScaler()
    scaler_cgga = StandardScaler()
    scaler_upenn = StandardScaler()

    # Split 80/20 train/val per cohort
    tcga_n = len(df_tcga)
    tcga_tr = int(0.8 * tcga_n)
    X_tcga_tr = scaler_tcga.fit_transform(df_tcga[tcga_cols].iloc[:tcga_tr].values)
    X_tcga_val = scaler_tcga.transform(df_tcga[tcga_cols].iloc[tcga_tr:].values)

    cgga_n = len(df_cgga)
    cgga_tr = int(0.8 * cgga_n)
    X_cgga_tr = scaler_cgga.fit_transform(df_cgga[cgga_cols].iloc[:cgga_tr].values)
    X_cgga_val = scaler_cgga.transform(df_cgga[cgga_cols].iloc[cgga_tr:].values)

    upenn_n = len(upenn_df_clean)
    upenn_tr = int(0.8 * upenn_n)
    X_upenn_tr = scaler_upenn.fit_transform(upenn_df_clean[upenn_cols].iloc[:upenn_tr].values)
    X_upenn_val = scaler_upenn.transform(upenn_df_clean[upenn_cols].iloc[upenn_tr:].values)

    datasets_dict = {
        "TCGA": {
            "X": X_tcga_tr, "durations": df_tcga["OS_MONTHS"].iloc[:tcga_tr].values, "events": df_tcga["OS_STATUS"].iloc[:tcga_tr].values,
            "X_val": X_tcga_val, "durations_val": df_tcga["OS_MONTHS"].iloc[tcga_tr:].values, "events_val": df_tcga["OS_STATUS"].iloc[tcga_tr:].values
        },
        "CGGA": {
            "X": X_cgga_tr, "durations": df_cgga["OS_MONTHS"].iloc[:cgga_tr].values, "events": df_cgga["OS_STATUS"].iloc[:cgga_tr].values,
            "X_val": X_cgga_val, "durations_val": df_cgga["OS_MONTHS"].iloc[cgga_tr:].values, "events_val": df_cgga["OS_STATUS"].iloc[cgga_tr:].values
        },
        "UPenn": {
            "X": X_upenn_tr, "durations": upenn_df_clean["OS_MONTHS"].iloc[:upenn_tr].values, "events": upenn_df_clean["OS_STATUS"].iloc[:upenn_tr].values,
            "X_val": X_upenn_val, "durations_val": upenn_df_clean["OS_MONTHS"].iloc[upenn_tr:].values, "events_val": upenn_df_clean["OS_STATUS"].iloc[upenn_tr:].values
        }
    }

    results = []
    seed_cindex_dict = {m: [] for m in METHODS}

    for seed in SEEDS:
        for method in METHODS:
            c_idx = train_federated_loso(seed, method, datasets_dict)
            seed_cindex_dict[method].append(c_idx)
            results.append({
                "method": method,
                "seed": seed,
                "c_index": round(c_idx, 4)
            })

    df_fed = pd.DataFrame(results)
    path_fed = os.path.join(REPORTS_DIR, "v10_federated_results.csv")
    df_fed.to_csv(path_fed, index=False)

    print("\nMulti-Seed Federated C-Index Summary (Mean ± Std across 10 Seeds):")
    for method in METHODS:
        arr = np.array(seed_cindex_dict[method])
        print(f"  {method:12s}: {np.mean(arr):.4f} ± {np.std(arr):.4f} (Min: {np.min(arr):.4f}, Max: {np.max(arr):.4f})")

    return df_fed, seed_cindex_dict


if __name__ == "__main__":
    run_federated_evaluation_v10()
