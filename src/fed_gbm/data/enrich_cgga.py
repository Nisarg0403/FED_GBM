"""
FED-GBM v9 CGGA Enrichment Module (Task 3).

Extracts real treatment flags (Radio_status and Chemo_status) from CGGA clinical files,
handles missing data strictly without imputation, and saves reports/v9_cgga_enrichment.csv.
"""

import os
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
REPORTS_DIR = os.path.join(_ROOT, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)


def enrich_cgga_dataset(df_cgga_combined: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Enriches CGGA_Combined dataset with real treatment flags (Radio_status, Chemo_status).
    Drops rows with missing treatment values (no imputation).
    """
    cgga_325_path = os.path.join(_ROOT, "data", "real", "cgga", "raw", "CGGA.mRNAseq_325_clinical.20200506.txt")
    cgga_693_path = os.path.join(_ROOT, "data", "real", "cgga", "raw", "CGGA.mRNAseq_693_clinical.20200506.txt")

    df325 = pd.read_csv(cgga_325_path, sep="\t")
    df693 = pd.read_csv(cgga_693_path, sep="\t")

    radio_col = "Radio_status (treated=1;un-treated=0)"
    chemo_col = "Chemo_status (TMZ treated=1;un-treated=0)"

    df325_sub = df325[["CGGA_ID", radio_col, chemo_col]].rename(columns={
        "CGGA_ID": "patient_id",
        radio_col: "Radio_status",
        chemo_col: "Chemo_status"
    })

    df693_sub = df693[["CGGA_ID", radio_col, chemo_col]].rename(columns={
        "CGGA_ID": "patient_id",
        radio_col: "Radio_status",
        chemo_col: "Chemo_status"
    })

    df_treatments = pd.concat([df325_sub, df693_sub], ignore_index=True).drop_duplicates(subset=["patient_id"])

    n_before = len(df_cgga_combined)
    df_merged = pd.merge(df_cgga_combined, df_treatments, on="patient_id", how="left")

    radio_cov = round(df_merged["Radio_status"].notna().mean() * 100.0, 2)
    chemo_cov = round(df_merged["Chemo_status"].notna().mean() * 100.0, 2)

    # Drop missing treatment status strictly
    df_clean = df_merged.dropna(subset=["Radio_status", "Chemo_status"]).copy()
    n_after = len(df_clean)

    pass_gate = (radio_cov >= 80.0) and (chemo_cov >= 80.0)

    audit_dict = {
        "cohort": "CGGA_Combined",
        "n_before": n_before,
        "n_after_treatment_filter": n_after,
        "radio_coverage_pct": radio_cov,
        "chemo_coverage_pct": chemo_cov,
        "gate_3_1_pass": bool(pass_gate)
    }

    df_audit = pd.DataFrame([audit_dict])
    path_audit = os.path.join(REPORTS_DIR, "v9_cgga_enrichment.csv")
    df_audit.to_csv(path_audit, index=False)

    return df_clean, audit_dict


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(_ROOT, "src"))
    from fed_gbm.load_real import load_v7_datasets
    cohort_dfs, _, _, _ = load_v7_datasets()
    enriched_cgga, audit = enrich_cgga_dataset(cohort_dfs["CGGA_Combined"])
    print("CGGA Enrichment Audit:\n", audit)
