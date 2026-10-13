"""
FED-GBM v7 Data Loader.

Loads real public Glioblastoma Multiforme (GBM) datasets:
1. TCGA-GBM Real Cohort (N=617 cases from cBioPortal / NCI GDC)
2. CGGA-325 Real Cohort (N=139 Grade IV cases from CGGA Portal)
3. CGGA-693 Real Cohort (N=249 Grade IV cases from CGGA Portal)

Features Panel (Strict Feature Extraction, No Imputation):
- Age (continuous, years)
- Gender (binary: 1=Male, 0=Female)
- IDH1_mutation (binary: 1=Mutant, 0=Wildtype)
- MGMT_methylation (binary: 1=Methylated, 0=Unmethylated)
- label_12m: 12-month survival label (1 = survived >= 365 days, 0 = died < 365 days)

Enforces ProvenanceGuard and LeakageGuard.
"""

import os
import json
import urllib.request
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List
from sklearn.model_selection import train_test_split

from fed_gbm.data.labels import compute_label_12m, audit_label_12m
from fed_gbm.data.features import extract_strict_features, audit_feature_coverage

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))

PROVENANCE_PATH = os.path.join(_ROOT, "data", "provenance.json")
TCGA_REAL_PATH = os.path.join(_ROOT, "data", "real", "tcga", "raw", "tcga_real_merged.csv")
CGGA_325_PATH = os.path.join(_ROOT, "data", "real", "cgga", "raw", "CGGA.mRNAseq_325_clinical.20200506.txt")
CGGA_693_PATH = os.path.join(_ROOT, "data", "real", "cgga", "raw", "CGGA.mRNAseq_693_clinical.20200506.txt")

FEATURE_COLS = ["Age", "Gender", "IDH1_mutation", "MGMT_methylation"]


def load_raw_tcga_dataset() -> pd.DataFrame:
    """Fetches/loads complete TCGA patient clinical and molecular data from cBioPortal."""
    try:
        url_gbm = 'https://www.cbioportal.org/api/studies/gbm_tcga/clinical-data?clinicalDataType=PATIENT'
        req = urllib.request.Request(url_gbm, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as resp:
            data_gbm = json.loads(resp.read().decode('utf-8'))
        df_gbm_pt = pd.DataFrame(data_gbm).drop_duplicates(subset=['patientId', 'clinicalAttributeId']).pivot(index='patientId', columns='clinicalAttributeId', values='value').reset_index()

        url_lgg = 'https://www.cbioportal.org/api/studies/lgggbm_tcga_pub/clinical-data'
        req = urllib.request.Request(url_lgg, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as resp:
            data_lgg = json.loads(resp.read().decode('utf-8'))
        df_lgg = pd.DataFrame(data_lgg).drop_duplicates(subset=['patientId', 'clinicalAttributeId']).pivot(index='patientId', columns='clinicalAttributeId', values='value').reset_index()

        tcga_df = pd.merge(df_gbm_pt, df_lgg[['patientId', 'IDH_STATUS', 'MGMT_PROMOTER_STATUS']], on='patientId', how='inner')
        tcga_df['patient_id'] = tcga_df['patientId']
        return tcga_df
    except Exception as e:
        print(f"Warning: Failed to fetch online cBioPortal TCGA data ({e}), falling back to local files.")
        if os.path.exists(TCGA_REAL_PATH):
            df_local = pd.read_csv(TCGA_REAL_PATH)
            p_col = [c for c in df_local.columns if 'patient' in c.lower() or 'barcode' in c.lower()][0]
            df_local['patient_id'] = df_local[p_col].astype(str).str[:12]
            return df_local
        raise FileNotFoundError(f"Neither online API nor local file {TCGA_REAL_PATH} available.")


def load_raw_cgga_dataset(filepath: str) -> pd.DataFrame:
    """Parses raw CGGA clinical file and filters Grade IV (GBM) cases."""
    sep = "\t" if filepath.endswith(".txt") else ","
    df = pd.read_csv(filepath, sep=sep)
    grade_col = [c for c in df.columns if "grade" in c.lower()][0]
    df_gbm = df[df[grade_col].astype(str).str.contains("IV|4|GBM", case=False, na=False)].copy()
    df_gbm["patient_id"] = df_gbm["CGGA_ID"]
    return df_gbm.reset_index(drop=True)


def process_cohort_v7(
    df_raw: pd.DataFrame,
    cohort_name: str,
    enforce_panel: List[str] = None
) -> Tuple[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Applies Task 1 (labeling) and Task 2 (strict feature extraction) to a raw cohort DataFrame.
    
    Returns:
        (df_clean, label_audit_record, feature_coverage_records, exclusion_log)
    """
    # 1. Task 1 Labeling & Audit
    label_audit_rec = audit_label_12m(df_raw, cohort_name)
    df_raw['label_12m'] = df_raw.apply(compute_label_12m, axis=1)
    df_valid = df_raw.dropna(subset=['label_12m']).copy()

    # 2. Task 2 Feature Extraction & Audit
    df_clean, feat_cols, cov_recs, exclusion_log = extract_strict_features(
        df_valid, cohort_name, enforce_panel=enforce_panel
    )
    exclusion_log["n_dropped_censored_12m"] = label_audit_rec["n_dropped"]

    return df_clean, label_audit_rec, cov_recs, exclusion_log


def load_v7_datasets() -> Tuple[Dict[str, pd.DataFrame], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Master v7 dataset loader for all cohorts (TCGA, CGGA_325, CGGA_693, CGGA_Combined).

    Returns:
        (cohort_dfs, label_audit_list, feature_coverage_list, exclusion_logs)
    """
    # 1. Load raw data
    raw_tcga = load_raw_tcga_dataset()
    raw_c325 = load_raw_cgga_dataset(CGGA_325_PATH)
    raw_c693 = load_raw_cgga_dataset(CGGA_693_PATH)

    # 2. Process each cohort using unified 4-feature panel: Age, Gender, IDH1, MGMT
    common_panel = ['Age', 'Gender', 'IDH1', 'MGMT']
    tcga_clean, tcga_la, tcga_fc, tcga_el = process_cohort_v7(raw_tcga, "TCGA", enforce_panel=common_panel)
    c325_clean, c325_la, c325_fc, c325_el = process_cohort_v7(raw_c325, "CGGA_325", enforce_panel=common_panel)
    c693_clean, c693_la, c693_fc, c693_el = process_cohort_v7(raw_c693, "CGGA_693", enforce_panel=common_panel)

    # 3. Create Combined CGGA Cohort
    cgga_combined_clean = pd.concat([c325_clean, c693_clean], ignore_index=True)
    cgga_comb_la = {
        "cohort": "CGGA_Combined",
        "n_before": c325_la["n_before"] + c693_la["n_before"],
        "n_after": c325_la["n_after"] + c693_la["n_after"],
        "n_dropped": c325_la["n_dropped"] + c693_la["n_dropped"],
        "drop_pct": round((c325_la["n_dropped"] + c693_la["n_dropped"]) / (c325_la["n_before"] + c693_la["n_before"]) * 100.0, 2),
        "alive_at_12m_pct": round((cgga_combined_clean["label_12m"] == 1).sum() / len(cgga_combined_clean) * 100.0, 2),
        "dead_at_12m_pct": round((cgga_combined_clean["label_12m"] == 0).sum() / len(cgga_combined_clean) * 100.0, 2),
    }

    cohort_dfs = {
        "TCGA": tcga_clean,
        "CGGA_325": c325_clean,
        "CGGA_693": c693_clean,
        "CGGA_Combined": cgga_combined_clean
    }

    label_audits = [tcga_la, c325_la, c693_la, cgga_comb_la]
    feature_coverages = tcga_fc + c325_fc + c693_fc
    exclusion_logs = [tcga_el, c325_el, c693_el]

    return cohort_dfs, label_audits, feature_coverages, exclusion_logs
