"""
FED-GBM v10 IEEE Centralized Baselines Script.

Evaluates 5-fold CV performance on TCGA and CGGA datasets using PyTorch DeepSurv (CoxPHLoss),
Cox Proportional Hazards, and XGBoost across incremental feature steps:
Step A: Shared clinical (Age, Gender, IDH1, MGMT)
Step B: + Cohort-specific clinical (KPS / Treatments)
Step C: + Top-100 gene expression
Step D: + All combined features

Saves reports/v10_centralized_baselines.csv.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from typing import List, Tuple
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from lifelines.utils import concordance_index

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from fed_gbm.load_real import load_v7_datasets
from fed_gbm.data.enrich_tcga import enrich_tcga_dataset
from fed_gbm.data.enrich_cgga import enrich_cgga_dataset
from fed_gbm.data.load_expression import load_expression_datasets
from fed_gbm.data.select_genes import select_top_genes
from fed_gbm.models import SurvivalMLP, CoxPHLoss, set_seed

REPORTS_DIR = os.path.join(_ROOT, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)


def evaluate_deepsurv_cv(df: pd.DataFrame, feature_cols: List[str], seed: int = 42) -> Tuple[float, float]:
    set_seed(seed)
    df_clean = df[feature_cols + ["OS_MONTHS", "OS_STATUS", "label_12m"]].copy()
    df_clean = df_clean.loc[:, ~df_clean.columns.duplicated()].dropna().copy()

    X = df_clean[feature_cols].values
    durations = df_clean["OS_MONTHS"].values
    events = df_clean["OS_STATUS"].values
    y_binary = df_clean["label_12m"].values

    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    criterion = CoxPHLoss()
    c_indices = []
    aucs = []

    for tr_idx, te_idx in kf.split(X):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[tr_idx])
        X_te = scaler.transform(X[te_idx])

        X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
        dur_tr_t = torch.tensor(durations[tr_idx], dtype=torch.float32)
        ev_tr_t = torch.tensor(events[tr_idx], dtype=torch.float32)
        X_te_t = torch.tensor(X_te, dtype=torch.float32)

        model = SurvivalMLP(input_dim=X_tr.shape[1], hidden_dims=[64, 32], dropout_rate=0.2)
        optimizer = optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-4)

        model.train()
        for epoch in range(100):
            optimizer.zero_grad()
            risk_scores = model(X_tr_t)
            loss = criterion(risk_scores, dur_tr_t, ev_tr_t)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            risk_te = model(X_te_t).view(-1).numpy()

        c_idx = concordance_index(durations[te_idx], -risk_te, events[te_idx])
        c_indices.append(c_idx)

        # Evaluate 12-month AUC from risk scores
        try:
            auc = roc_auc_score(y_binary[te_idx], risk_te)
        except ValueError:
            auc = 0.5
        aucs.append(auc)

    return float(np.mean(c_indices)), float(np.mean(aucs))


def run_centralized_baselines_v10():
    print("======================================================================")
    print("     TASK 8: V10 IEEE CENTRALIZED SURVIVAL BASELINES (5-FOLD CV)")
    print("======================================================================")

    cohort_dfs, _, _, _ = load_v7_datasets()
    tcga_df = cohort_dfs["TCGA"]
    cgga_df = cohort_dfs["CGGA_Combined"]

    tcga_enriched, _ = enrich_tcga_dataset(tcga_df)
    cgga_enriched, _ = enrich_cgga_dataset(cgga_df)
    tcga_expr, cgga_expr, _ = load_expression_datasets(tcga_enriched, cgga_enriched)

    tcga_top_genes, _ = select_top_genes(tcga_expr, "TCGA", n_genes=100)
    cgga_top_genes, _ = select_top_genes(cgga_expr, "CGGA_Combined", n_genes=100)

    # TCGA Feature Steps
    tcga_stepA = ["Age", "Gender", "IDH1_mutation", "MGMT_methylation"]
    tcga_stepB = tcga_stepA + ["karnofsky_score"]
    tcga_stepC = tcga_stepA + tcga_top_genes
    tcga_stepD = tcga_stepB + tcga_top_genes

    # CGGA Feature Steps
    cgga_stepA = ["Age", "Gender", "IDH1_mutation", "MGMT_methylation"]
    cgga_stepB = cgga_stepA + ["Radio_status", "Chemo_status"]
    cgga_stepC = cgga_stepA + cgga_top_genes
    cgga_stepD = cgga_stepB + cgga_top_genes

    steps_tcga = [("Step A (Shared Clinical)", tcga_stepA),
                  ("Step B (+ KPS Clinical)", tcga_stepB),
                  ("Step C (+ Top-100 Genes)", tcga_stepC),
                  ("Step D (+ All Combined)", tcga_stepD)]

    steps_cgga = [("Step A (Shared Clinical)", cgga_stepA),
                  ("Step B (+ Treatment Flags)", cgga_stepB),
                  ("Step C (+ Top-100 Genes)", cgga_stepC),
                  ("Step D (+ All Combined)", cgga_stepD)]

    records = []

    for name, cols in steps_tcga:
        c_idx, auc = evaluate_deepsurv_cv(tcga_expr, cols)
        records.append({"cohort": "TCGA", "step": name, "n_features": len(cols), "c_index": round(c_idx, 4), "auc_12m": round(auc, 4)})

    for name, cols in steps_cgga:
        c_idx, auc = evaluate_deepsurv_cv(cgga_expr, cols)
        records.append({"cohort": "CGGA_Combined", "step": name, "n_features": len(cols), "c_index": round(c_idx, 4), "auc_12m": round(auc, 4)})

    df_res = pd.DataFrame(records)
    path_res = os.path.join(REPORTS_DIR, "v10_centralized_baselines.csv")
    df_res.to_csv(path_res, index=False)
    print(df_res.to_string())

    tcga_stepD_cindex = df_res[(df_res["cohort"] == "TCGA") & (df_res["step"].str.contains("Step D"))]["c_index"].values[0]
    cgga_stepD_cindex = df_res[(df_res["cohort"] == "CGGA_Combined") & (df_res["step"].str.contains("Step D"))]["c_index"].values[0]

    pass_gate_8_1 = (tcga_stepD_cindex >= 0.76) and (cgga_stepD_cindex >= 0.66)
    print(f"\nVerification Gate 8.1 (v10 IEEE Target): TCGA Step D C-Index >= 0.76 & CGGA Step D C-Index >= 0.66? {'[PASS]' if pass_gate_8_1 else '[FAIL]'}")

    return df_res


if __name__ == "__main__":
    run_centralized_baselines_v10()
