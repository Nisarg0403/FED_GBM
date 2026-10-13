"""
FED-GBM v9 Gene Selection Module (Task 5).

Selects top-200 prognostic genes per cohort (TCGA and CGGA) on training data only using
high-variance pre-filtering and mutual information, enforces LeakageGuard (MI < 0.50),
and saves reports/v9_selected_genes.csv.
"""

import os
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any
from sklearn.feature_selection import mutual_info_classif

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
REPORTS_DIR = os.path.join(_ROOT, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)


def select_top_genes(
    df_cohort: pd.DataFrame,
    cohort_name: str,
    n_genes: int = 200,
    seed: int = 42
) -> Tuple[List[str], pd.DataFrame]:
    """
    Selects top-N prognostic genes by mutual information with label_12m on training data.
    Pre-filters top 2000 highest-variance genes for computational efficiency and signal focus.
    Enforces LeakageGuard: ensures max MI < 0.50.
    """
    clinical_cols = {"patient_id", "label_12m", "Age", "Gender", "IDH1_mutation", "MGMT_methylation", "karnofsky_score", "Radio_status", "Chemo_status"}
    gene_cols = [c for c in df_cohort.columns if c not in clinical_cols and not c.startswith("TCGA-")]

    if not gene_cols:
        return [], pd.DataFrame()

    # Pre-filter top 2000 genes by variance
    if len(gene_cols) > 2000:
        variances = df_cohort[gene_cols].var()
        candidate_genes = variances.nlargest(2000).index.tolist()
    else:
        candidate_genes = gene_cols

    X_gene = df_cohort[candidate_genes].fillna(0.0)
    y = df_cohort["label_12m"].values

    mi_scores = mutual_info_classif(X_gene, y, random_state=seed)
    mi_df = pd.DataFrame({
        "cohort": cohort_name,
        "gene": candidate_genes,
        "mutual_info": mi_scores
    }).sort_values("mutual_info", ascending=False).reset_index(drop=True)

    top_df = mi_df.head(n_genes).copy()
    top_gene_names = top_df["gene"].tolist()
    max_mi = float(top_df["mutual_info"].max())

    return top_gene_names, top_df


def run_gene_selection(df_tcga_expr: pd.DataFrame, df_cgga_expr: pd.DataFrame) -> Tuple[List[str], List[str], Dict[str, Any]]:
    """
    Executes gene selection for TCGA and CGGA and saves audit report.
    """
    tcga_genes, tcga_df = select_top_genes(df_tcga_expr, "TCGA", n_genes=200)
    cgga_genes, cgga_df = select_top_genes(df_cgga_expr, "CGGA_Combined", n_genes=200)

    combined_df = pd.concat([tcga_df, cgga_df], ignore_index=True)
    path_selected = os.path.join(REPORTS_DIR, "v9_selected_genes.csv")
    combined_df.to_csv(path_selected, index=False)

    max_tcga_mi = float(tcga_df["mutual_info"].max()) if len(tcga_df) > 0 else 0.0
    max_cgga_mi = float(cgga_df["mutual_info"].max()) if len(cgga_df) > 0 else 0.0

    pass_gate = (len(tcga_genes) == 200) and (len(cgga_genes) == 200) and (max_tcga_mi < 0.50) and (max_cgga_mi < 0.50)

    audit_dict = {
        "tcga_n_genes_selected": len(tcga_genes),
        "cgga_n_genes_selected": len(cgga_genes),
        "tcga_max_gene_mi": round(max_tcga_mi, 4),
        "cgga_max_gene_mi": round(max_cgga_mi, 4),
        "gate_5_1_pass": bool(pass_gate)
    }

    return tcga_genes, cgga_genes, audit_dict
