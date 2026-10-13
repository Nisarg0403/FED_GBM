"""
FED-GBM v9 Centralized Baselines Script (Task 8).

Evaluates 5-fold cross-validation performance on TCGA and CGGA across 4 incremental feature steps:
Step A: Shared clinical (Age, Gender, IDH1, MGMT)
Step B: + Cohort-specific clinical (KPS / Treatments)
Step C: + Top-200 gene expression
Step D: + All combined features

Saves reports/v9_centralized_baselines.csv.
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from fed_gbm.load_real import load_v7_datasets
from fed_gbm.data.enrich_tcga import enrich_tcga_dataset
from fed_gbm.data.enrich_cgga import enrich_cgga_dataset
from fed_gbm.data.load_expression import load_expression_datasets
from fed_gbm.data.select_genes import select_top_genes

REPORTS_DIR = os.path.join(_ROOT, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)


def evaluate_xgboost_cv(df: pd.DataFrame, feature_cols: List[str], seed: int = 42) -> float:
    X = df[feature_cols].fillna(0.0)
    y = df["label_12m"].values
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    aucs = []
    for tr_idx, te_idx in skf.split(X, y):
        clf = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=seed, eval_metric="logloss")
        clf.fit(X.iloc[tr_idx], y[tr_idx])
        preds = clf.predict_proba(X.iloc[te_idx])[:, 1]
        aucs.append(roc_auc_score(y[te_idx], preds))
    return float(np.mean(aucs))


def run_centralized_baselines():
    print("======================================================================")
    print("           TASK 8: CENTRALIZED BASELINES (5-FOLD CV)")
    print("======================================================================")

    cohort_dfs, _, _, _ = load_v7_datasets()
    tcga_df = cohort_dfs["TCGA"]
    cgga_df = cohort_dfs["CGGA_Combined"]

    tcga_enriched, _ = enrich_tcga_dataset(tcga_df)
    cgga_enriched, _ = enrich_cgga_dataset(cgga_df)

    tcga_expr, cgga_expr, _ = load_expression_datasets(tcga_enriched, cgga_enriched)

    tcga_top_genes, _ = select_top_genes(tcga_expr, "TCGA", n_genes=200)
    cgga_top_genes, _ = select_top_genes(cgga_expr, "CGGA_Combined", n_genes=200)

    # 1. TCGA Steps
    tcga_stepA = ["Age", "Gender", "IDH1_mutation", "MGMT_methylation"]
    tcga_stepB = tcga_stepA + ["karnofsky_score"]
    tcga_stepC = tcga_stepA + tcga_top_genes
    tcga_stepD = tcga_stepB + tcga_top_genes

    auc_tcga_a = evaluate_xgboost_cv(tcga_expr, tcga_stepA)
    auc_tcga_b = evaluate_xgboost_cv(tcga_expr, tcga_stepB)
    auc_tcga_c = evaluate_xgboost_cv(tcga_expr, tcga_stepC)
    auc_tcga_d = evaluate_xgboost_cv(tcga_expr, tcga_stepD)

    # 2. CGGA Steps
    cgga_stepA = ["Age", "Gender", "IDH1_mutation", "MGMT_methylation"]
    cgga_stepB = cgga_stepA + ["Radio_status", "Chemo_status"]
    cgga_stepC = cgga_stepA + cgga_top_genes
    cgga_stepD = cgga_stepB + cgga_top_genes

    auc_cgga_a = evaluate_xgboost_cv(cgga_expr, cgga_stepA)
    auc_cgga_b = evaluate_xgboost_cv(cgga_expr, cgga_stepB)
    auc_cgga_c = evaluate_xgboost_cv(cgga_expr, cgga_stepC)
    auc_cgga_d = evaluate_xgboost_cv(cgga_expr, cgga_stepD)

    records = [
        {"cohort": "TCGA", "step": "Step A (Shared Clinical)", "n_features": len(tcga_stepA), "mean_auc": round(auc_tcga_a, 4)},
        {"cohort": "TCGA", "step": "Step B (+ KPS Clinical)", "n_features": len(tcga_stepB), "mean_auc": round(auc_tcga_b, 4)},
        {"cohort": "TCGA", "step": "Step C (+ Top-200 Genes)", "n_features": len(tcga_stepC), "mean_auc": round(auc_tcga_c, 4)},
        {"cohort": "TCGA", "step": "Step D (+ All Combined)", "n_features": len(tcga_stepD), "mean_auc": round(auc_tcga_d, 4)},
        {"cohort": "CGGA_Combined", "step": "Step A (Shared Clinical)", "n_features": len(cgga_stepA), "mean_auc": round(auc_cgga_a, 4)},
        {"cohort": "CGGA_Combined", "step": "Step B (+ Treatment Flags)", "n_features": len(cgga_stepB), "mean_auc": round(auc_cgga_b, 4)},
        {"cohort": "CGGA_Combined", "step": "Step C (+ Top-200 Genes)", "n_features": len(cgga_stepC), "mean_auc": round(auc_cgga_c, 4)},
        {"cohort": "CGGA_Combined", "step": "Step D (+ All Combined)", "n_features": len(cgga_stepD), "mean_auc": round(auc_cgga_d, 4)},
    ]

    df_base = pd.DataFrame(records)
    path_base = os.path.join(REPORTS_DIR, "v9_centralized_baselines.csv")
    df_base.to_csv(path_base, index=False)
    print(df_base.to_string())

    pass_gate_8_1 = (auc_tcga_d >= 0.70) and (auc_cgga_d >= 0.60)
    print(f"\nVerification Gate 8.1: TCGA Step D >= 0.70 and CGGA Step D >= 0.60? {'[PASS]' if pass_gate_8_1 else '[FAIL]'}")

    return df_base


if __name__ == "__main__":
    run_centralized_baselines()
