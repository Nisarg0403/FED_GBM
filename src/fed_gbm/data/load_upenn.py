"""
FED-GBM v8 UPenn-GBM Data Loader Module.

Loads UPenn-GBM clinical and mpMRI radiomic data with strict censoring handling
and explicit IDH1 mutation filtering.
"""

import os
import glob
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
from functools import reduce

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))

DATA_DIR = os.path.join(_ROOT, "data", "real", "upenn")
REPORTS_DIR = os.path.join(_ROOT, "reports")


def upenn_label_12m(row: Any) -> Any:
    """
    Strict 12-month survival binary classification label for UPenn clinical schema.
    Returns:
        1.0  -> confirmed survival >= 365 days
        0.0  -> confirmed death < 365 days
        None -> censored before 365 days (EXCLUDED downstream)
    """
    status = str(row.get('Survival_Status', '')).lower()
    days_val = row.get('Survival_from_surgery_days_UPDATED')
    if pd.isna(days_val) or str(days_val).strip().lower() in ['not available', 'nan', 'none', '']:
        return None
    try:
        days = float(days_val)
    except (ValueError, TypeError):
        return None

    if 'deceased' in status or 'dead' in status:
        return 1.0 if days >= 365.0 else 0.0
    elif 'alive' in status or 'living' in status:
        return 1.0 if days >= 365.0 else None
    return None


def load_upenn_dataset() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Loads raw UPenn clinical and radiomics datasets, applies strict survival labeling,
    filters out unknown IDH1 values, and returns clean merged DataFrame and audit stats.
    """
    os.makedirs(REPORTS_DIR, exist_ok=True)
    clin_csv = os.path.join(DATA_DIR, "clinical", "UPENN-GBM_clinical_info_v2.1.csv")

    if not os.path.exists(clin_csv):
        raise FileNotFoundError(f"UPenn clinical file not found at {clin_csv}")

    df_clin = pd.read_csv(clin_csv)
    n_raw = len(df_clin)

    df_clin['patient_id'] = df_clin['ID'].astype(str).str.split('_').str[0]
    df_clin['label_12m'] = df_clin.apply(upenn_label_12m, axis=1)

    # Filter survival censoring
    df_survival_clean = df_clin.dropna(subset=['label_12m']).copy()
    n_after_survival = len(df_survival_clean)

    # Filter IDH1 strict (Mutated vs Wildtype, drop NOS/NEC)
    df_idh1_clean = df_survival_clean[df_survival_clean['IDH1'].isin(['Mutated', 'Wildtype'])].copy()
    n_after_idh1 = len(df_idh1_clean)

    # Format clinical features
    df_idh1_clean['Age'] = pd.to_numeric(df_idh1_clean['Age_at_scan_years'], errors='coerce')
    df_idh1_clean['Gender'] = df_idh1_clean['Gender'].map({'M': 1.0, 'F': 0.0})
    df_idh1_clean['IDH1_mutation'] = df_idh1_clean['IDH1'].map({'Mutated': 1.0, 'Wildtype': 0.0})
    df_idh1_clean['OS_MONTHS'] = pd.to_numeric(df_idh1_clean['Survival_from_surgery_days_UPDATED'], errors='coerce') / 30.4375
    df_idh1_clean['OS_STATUS'] = df_idh1_clean['Survival_Status'].apply(lambda s: 1.0 if 'deceased' in str(s).lower() or 'dead' in str(s).lower() else 0.0)

    df_clin_clean = df_idh1_clean.dropna(subset=['Age', 'Gender', 'IDH1_mutation', 'OS_MONTHS', 'OS_STATUS']).copy()
    n_after_clin = len(df_clin_clean)

    # Merge radiomics
    rad_dir = os.path.join(DATA_DIR, "radiomics")
    rad_files = glob.glob(os.path.join(rad_dir, "Radiomic_Features_CaPTk_automaticsegm_*.csv"))

    if rad_files:
        rad_dfs = []
        for f in rad_files:
            df_temp = pd.read_csv(f)
            df_temp['patient_id'] = df_temp['SubjectID'].astype(str).str.split('_').str[0]
            df_temp = df_temp.drop(columns=['SubjectID'], errors='ignore')
            df_temp = df_temp.loc[:, ~df_temp.columns.duplicated()]
            rad_dfs.append(df_temp)

        df_rad_all = reduce(lambda left, right: pd.merge(left, right, on='patient_id', how='outer'), rad_dfs)
        df_rad_all = df_rad_all.loc[:, ~df_rad_all.columns.duplicated()]
        merged_df = pd.merge(df_clin_clean, df_rad_all, on='patient_id', how='inner')
    else:
        merged_df = df_clin_clean

    n_final = len(merged_df)
    n_pos = int((merged_df['label_12m'] == 1.0).sum())
    n_neg = int((merged_df['label_12m'] == 0.0).sum())
    pct_pos = round(n_pos / n_final * 100.0, 2)
    pct_neg = round(n_neg / n_final * 100.0, 2)

    audit_dict = {
        "cohort": "UPenn-GBM",
        "n_raw_clinical": n_raw,
        "n_after_survival_filter": n_after_survival,
        "n_after_idh1_filter": n_after_idh1,
        "n_final_matched": n_final,
        "n_survived_12m": n_pos,
        "n_died_12m": n_neg,
        "pct_survived": pct_pos,
        "pct_died": pct_neg,
        "gate_8_1_pass": bool(n_final >= 400 and pct_pos >= 40.0 and pct_neg >= 40.0)
    }

    audit_df = pd.DataFrame([audit_dict])
    path_audit = os.path.join(REPORTS_DIR, "v8_upenn_load_audit.csv")
    audit_df.to_csv(path_audit, index=False)

    return merged_df, audit_dict
