"""
FED-GBM v9 Expression Loader Module (Task 4).

Loads TCGA STAR RNA-seq counts and CGGA mRNAseq expression matrices,
applies log2 transformations, aligns samples to clinical patient barcodes,
and saves reports/v9_expression_coverage.csv.
"""

import os
import requests
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
REPORTS_DIR = os.path.join(_ROOT, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)


def load_expression_datasets(df_tcga: pd.DataFrame, df_cgga: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Loads TCGA and CGGA expression datasets, transforms counts to log2, aligns samples,
    and returns expression DataFrames along with coverage audit stats.
    """
    # 1. TCGA Expression
    tcga_tsv = os.path.join(_ROOT, "data", "real", "tcga", "expression", "TCGA-GBM_STAR_unstranded_TUMOR_ONLY.tsv")
    if not os.path.exists(tcga_tsv):
        raise FileNotFoundError(f"TCGA tumor-only expression file not found at {tcga_tsv}")

    df_tcga_raw = pd.read_csv(tcga_tsv, sep="\t")
    df_tcga_raw["patient_id"] = df_tcga_raw["case_submitter_id"].astype(str).str[:12]
    df_tcga_raw = df_tcga_raw.drop_duplicates(subset=["patient_id"]).copy()

    ensg_cols = [c for c in df_tcga_raw.columns if c.startswith("ENSG")]

    # Extract base ENSG IDs
    ensg_renamed = {c: c.split(".")[0] for c in ensg_cols}

    # Convert numeric counts
    counts_df = df_tcga_raw[ensg_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    counts_log2 = np.log2(counts_df + 1.0)
    counts_log2 = counts_log2.rename(columns=ensg_renamed)

    # Aggregate duplicate gene columns by mean
    counts_log2 = counts_log2.T.groupby(level=0).mean().T
    counts_log2["patient_id"] = df_tcga_raw["patient_id"].values

    tcga_expr_matched = pd.merge(df_tcga, counts_log2, on="patient_id", how="inner")

    # 2. CGGA Expression
    cgga_325_path = os.path.join(_ROOT, "data", "real", "cgga", "expression", "CGGA_325_expression.csv")
    cgga_693_path = os.path.join(_ROOT, "data", "real", "cgga", "expression", "CGGA_693_expression.csv")

    e325 = pd.read_csv(cgga_325_path, index_col=0)
    e693 = pd.read_csv(cgga_693_path, index_col=0)
    df_cgga_raw = pd.concat([e325, e693], axis=1).T
    df_cgga_raw.index.name = "patient_id"
    df_cgga_raw = df_cgga_raw.reset_index().drop_duplicates(subset=["patient_id"]).copy()

    gene_cols_cgga = [c for c in df_cgga_raw.columns if c != "patient_id"]
    cgga_expr_vals = df_cgga_raw[gene_cols_cgga].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    cgga_log2 = np.log2(cgga_expr_vals + 1.0)
    cgga_log2["patient_id"] = df_cgga_raw["patient_id"].values

    cgga_expr_matched = pd.merge(df_cgga, cgga_log2, on="patient_id", how="inner")

    n_tcga_matched = len(tcga_expr_matched)
    n_cgga_matched = len(cgga_expr_matched)

    pass_gate = (n_tcga_matched >= 100) and (n_cgga_matched >= 200)

    audit_dict = {
        "tcga_expression_matched_n": n_tcga_matched,
        "cgga_expression_matched_n": n_cgga_matched,
        "tcga_genes_available": len(ensg_cols),
        "cgga_genes_available": len(gene_cols_cgga),
        "gate_4_1_pass": bool(pass_gate)
    }

    df_audit = pd.DataFrame([audit_dict])
    path_audit = os.path.join(REPORTS_DIR, "v9_expression_coverage.csv")
    df_audit.to_csv(path_audit, index=False)

    return tcga_expr_matched, cgga_expr_matched, audit_dict
