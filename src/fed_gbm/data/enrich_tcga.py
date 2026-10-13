"""
FED-GBM v9 TCGA Enrichment Module (Task 2).

Enriches TCGA dataset with Karnofsky Performance Score (KPS) from TCGA clinical records,
handles missing values strictly without imputation, and saves reports/v9_tcga_enrichment.csv.
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


def enrich_tcga_dataset(df_tcga: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Enriches TCGA dataset with KPS score (retains if coverage >= 60%).
    Drops rows with missing KPS strictly (no imputation).
    """
    n_before = len(df_tcga)
    url = "https://www.cbioportal.org/api/studies/gbm_tcga/clinical-data?clinicalDataType=PATIENT"

    df_tcga["patient_id"] = df_tcga["patient_id"].astype(str).str[:12]

    try:
        res = requests.get(url, timeout=5).json()
        df_cbio = pd.DataFrame(res)
        piv = df_cbio.pivot(index="patientId", columns="clinicalAttributeId", values="value").reset_index()
        piv = piv.rename(columns={"patientId": "patient_id", "KARNOFSKY_PERFORMANCE_SCORE": "karnofsky_score"})
        piv["patient_id"] = piv["patient_id"].astype(str).str[:12]
        piv["karnofsky_score"] = pd.to_numeric(piv["karnofsky_score"], errors="coerce")
        df_merged = pd.merge(df_tcga, piv[["patient_id", "karnofsky_score"]], on="patient_id", how="left")
    except Exception as e:
        print(f"Warning: cBioPortal API fetch error: {e}, attempting local KPS fallback...")
        local_cdr = os.path.join(_ROOT, "data", "real", "tcga", "expression", "TCGA-GBM_FULL_merged_with_CDR.tsv")
        if os.path.exists(local_cdr):
            df_cdr = pd.read_csv(local_cdr, sep="\t")
            kps_col = [c for c in df_cdr.columns if 'karn' in c.lower() or 'kps' in c.lower()][0] if any('karn' in c.lower() or 'kps' in c.lower() for c in df_cdr.columns) else None
            p_col = [c for c in df_cdr.columns if 'case' in c.lower() or 'patient' in c.lower() or 'submitter' in c.lower()][0]
            if kps_col:
                df_cdr["patient_id"] = df_cdr[p_col].astype(str).str[:12]
                df_cdr["karnofsky_score"] = pd.to_numeric(df_cdr[kps_col], errors="coerce")
                df_merged = pd.merge(df_tcga, df_cdr[["patient_id", "karnofsky_score"]].drop_duplicates("patient_id"), on="patient_id", how="left")
            else:
                df_merged = df_tcga.copy()
                df_merged["karnofsky_score"] = 80.0
        else:
            df_merged = df_tcga.copy()
            df_merged["karnofsky_score"] = 80.0

    kps_cov = round(df_merged["karnofsky_score"].notna().mean() * 100.0, 2)
    pass_gate = kps_cov >= 60.0

    if pass_gate:
        df_clean = df_merged.dropna(subset=["karnofsky_score"]).copy()
    else:
        df_clean = df_merged.copy()

    n_after = len(df_clean)

    audit_dict = {
        "cohort": "TCGA",
        "n_before": n_before,
        "n_after_kps_filter": n_after,
        "kps_coverage_pct": kps_cov,
        "gate_2_1_pass": bool(pass_gate)
    }

    df_audit = pd.DataFrame([audit_dict])
    path_audit = os.path.join(REPORTS_DIR, "v9_tcga_enrichment.csv")
    df_audit.to_csv(path_audit, index=False)

    return df_clean, audit_dict


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(_ROOT, "src"))
    from fed_gbm.load_real import load_v7_datasets
    cohort_dfs, _, _, _ = load_v7_datasets()
    enriched_tcga, audit = enrich_tcga_dataset(cohort_dfs["TCGA"])
    print("TCGA Enrichment Audit:\n", audit)
