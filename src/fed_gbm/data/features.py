"""
FED-GBM v7 Strict Feature Extraction Module.

Enforces strict feature extraction rules with NO imputation of key biomarkers:
- Age: continuous, kept if present (>95%).
- Gender: binary (1=Male, 0=Female), kept if present (>95%).
- IDH1: binary (1=Mutant, 0=Wildtype). Drop missing/unknowns (NO default to wildtype).
- MGMT: binary (1=Methylated, 0=Unmethylated). Drop missing/unknowns. Drop column if coverage < 40%.
- Treatment_Radiation: binary (1=Treated, 0=Untreated). Drop column if coverage < 30%.
- Treatment_Chemo: binary (1=Treated, 0=Untreated). Drop column if coverage < 30%.
- KPS: continuous (40-100). Drop column if coverage < 40%.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional


def audit_feature_coverage(
    df: pd.DataFrame,
    cohort_name: str,
    feature_mappings: Dict[str, str]
) -> List[Dict[str, Any]]:
    """
    Audits feature coverage across raw/valid cohort.

    Returns a list of dicts with:
        cohort, feature, n_present, n_missing, coverage_pct
    """
    results = []
    n_total = len(df)

    for feat_name, col_name in feature_mappings.items():
        if col_name is not None and col_name in df.columns:
            s = df[col_name]
            valid_mask = s.notna() & (~s.astype(str).str.lower().isin(['nan', 'unknown', 'not reported', 'none', '']))
            n_present = int(valid_mask.sum())
        else:
            n_present = 0

        n_missing = n_total - n_present
        cov_pct = (n_present / n_total * 100.0) if n_total > 0 else 0.0

        results.append({
            "cohort": cohort_name,
            "feature": feat_name,
            "n_present": n_present,
            "n_missing": n_missing,
            "coverage_pct": round(cov_pct, 2)
        })

    return results


def extract_strict_features(
    df: pd.DataFrame,
    cohort_name: str,
    feature_thresholds: Dict[str, float] = None,
    enforce_panel: List[str] = None
) -> Tuple[pd.DataFrame, List[str], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Extracts strict features for a cohort based on coverage thresholds and returns cleaned DataFrame.

    Rules:
    - IDH1: drop patients with missing status.
    - MGMT: drop missing status; drop column if coverage < 40%.
    - Treatment_Radiation / Chemo: drop column if coverage < 30%.
    - KPS: drop column if coverage < 40%.

    Returns:
        (df_cleaned, active_features, coverage_records, exclusion_log)
    """
    if feature_thresholds is None:
        feature_thresholds = {
            "MGMT": 40.0,
            "Treatment_Radiation": 30.0,
            "Treatment_Chemo": 30.0,
            "KPS": 40.0,
        }

    df_work = df.copy()
    n_initial = len(df_work)

    # Standardize column mappings across TCGA / CGGA schemas
    col_map = {}
    
    # Age
    age_cols = [c for c in df_work.columns if c.lower() in ['age', 'age_at_index', 'age_at_diagnosis'] or 'age' in c.lower()]
    col_map['Age'] = age_cols[0] if age_cols else None

    # Gender
    gender_cols = [c for c in df_work.columns if c.lower() in ['gender', 'sex', 'demographic.gender']]
    col_map['Gender'] = gender_cols[0] if gender_cols else None

    # IDH1
    idh_candidates = [c for c in df_work.columns if 'idh' in c.lower()]
    col_map['IDH1'] = idh_candidates[0] if idh_candidates else None

    # MGMT
    mgmt_candidates = [c for c in df_work.columns if 'mgmt' in c.lower()]
    col_map['MGMT'] = mgmt_candidates[0] if mgmt_candidates else None

    # KPS
    kps_candidates = [c for c in df_work.columns if 'kps' in c.lower() or 'karn' in c.lower()]
    col_map['KPS'] = kps_candidates[0] if kps_candidates else None

    # Radiation
    rad_candidates = [c for c in df_work.columns if 'radio' in c.lower() or 'radiation' in c.lower()]
    col_map['Treatment_Radiation'] = rad_candidates[0] if rad_candidates else None

    # Chemo
    chemo_candidates = [c for c in df_work.columns if 'chemo' in c.lower() or 'pharmaceutical' in c.lower()]
    col_map['Treatment_Chemo'] = chemo_candidates[0] if chemo_candidates else None

    # Audit coverage
    coverage_records = audit_feature_coverage(df_work, cohort_name, col_map)
    cov_dict = {r['feature']: r['coverage_pct'] for r in coverage_records}

    # If enforce_panel is provided (e.g. for multi-center federated alignment), use it.
    if enforce_panel is not None:
        active_features = list(enforce_panel)
        dropped_columns_decision = []
        for feat in ['MGMT', 'KPS', 'Treatment_Radiation', 'Treatment_Chemo']:
            if feat not in active_features:
                thresh = feature_thresholds.get(feat, 30.0)
                cov = cov_dict.get(feat, 0.0)
                dropped_columns_decision.append({
                    "feature": feat,
                    "coverage_pct": cov,
                    "threshold_pct": thresh,
                    "reason": f"Dropped for multi-center alignment (Coverage {cov:.1f}%)"
                })
    else:
        active_features = ['Age', 'Gender', 'IDH1']
        dropped_columns_decision = []
        for opt_feat in ['MGMT', 'KPS', 'Treatment_Radiation', 'Treatment_Chemo']:
            thresh = feature_thresholds.get(opt_feat, 30.0)
            cov = cov_dict.get(opt_feat, 0.0)
            if cov >= thresh:
                active_features.append(opt_feat)
            else:
                dropped_columns_decision.append({
                    "feature": opt_feat,
                    "coverage_pct": cov,
                    "threshold_pct": thresh,
                    "reason": f"Coverage {cov:.1f}% below threshold {thresh:.1f}%"
                })

    # Extract standardized columns into df_clean
    df_clean = pd.DataFrame(index=df_work.index)
    if 'patient_id' in df_work.columns:
        df_clean['patient_id'] = df_work['patient_id']
    elif 'CGGA_ID' in df_work.columns:
        df_clean['patient_id'] = df_work['CGGA_ID']
    elif 'patientId' in df_work.columns:
        df_clean['patient_id'] = df_work['patientId']

    df_clean['label_12m'] = df_work['label_12m']

    # Extract continuous survival duration and event status if present
    def extract_survival_time_and_status(row):
        # 1. cBioPortal OS_MONTHS / OS_STATUS
        if 'OS_MONTHS' in row and 'OS_STATUS' in row:
            m = pd.to_numeric(row['OS_MONTHS'], errors='coerce')
            st_str = str(row['OS_STATUS']).lower()
            ev = 1.0 if ('deceased' in st_str or 'dead' in st_str or '1:' in st_str) else 0.0
            return m, ev
        # 2. CGGA OS / Censor
        censor_col = 'Censor' if 'Censor' in row else ('Censor (alive=0; dead=1)' if 'Censor (alive=0; dead=1)' in row else None)
        if 'OS' in row and censor_col is not None:
            days = pd.to_numeric(row['OS'], errors='coerce')
            m = days / 30.4375 if pd.notna(days) else np.nan
            ev = pd.to_numeric(row[censor_col], errors='coerce')
            return m, ev
        # 3. GDC days_to_death / days_to_last_follow_up
        if 'vital_status' in row:
            vs = str(row['vital_status']).lower()
            if vs == 'dead' and 'days_to_death' in row:
                d = pd.to_numeric(row['days_to_death'], errors='coerce')
                return (d / 30.4375 if pd.notna(d) else np.nan), 1.0
            elif vs == 'alive' and 'days_to_last_follow_up' in row:
                f = pd.to_numeric(row['days_to_last_follow_up'], errors='coerce')
                return (f / 30.4375 if pd.notna(f) else np.nan), 0.0
        return np.nan, np.nan

    surv_tuples = df_work.apply(extract_survival_time_and_status, axis=1)
    df_clean['OS_MONTHS'] = [t[0] for t in surv_tuples]
    df_clean['OS_STATUS'] = [t[1] for t in surv_tuples]

    # 1. Age
    if col_map['Age'] is not None:
        df_clean['Age'] = pd.to_numeric(df_work[col_map['Age']], errors='coerce')

    # 2. Gender
    def parse_gender(v):
        s = str(v).lower().strip()
        if s in ['male', 'm', '1', '1.0']: return 1.0
        if s in ['female', 'f', '0', '0.0']: return 0.0
        return np.nan
    if col_map['Gender'] is not None:
        df_clean['Gender'] = df_work[col_map['Gender']].apply(parse_gender)

    # 3. IDH1
    def parse_idh(v):
        s = str(v).lower().strip()
        if 'mut' in s or s in ['1', '1.0']: return 1.0
        if 'wild' in s or 'wt' in s or s in ['0', '0.0']: return 0.0
        return np.nan
    if col_map['IDH1'] is not None:
        df_clean['IDH1_mutation'] = df_work[col_map['IDH1']].apply(parse_idh)

    # 4. MGMT
    def parse_mgmt(v):
        s = str(v).lower().strip()
        if 'un' in s or s in ['0', '0.0']: return 0.0
        if 'methylated' in s or s in ['1', '1.0']: return 1.0
        return np.nan
    if 'MGMT' in active_features and col_map['MGMT'] is not None:
        df_clean['MGMT_methylation'] = df_work[col_map['MGMT']].apply(parse_mgmt)

    # 5. KPS
    if 'KPS' in active_features and col_map['KPS'] is not None:
        df_clean['KPS_Score'] = pd.to_numeric(df_work[col_map['KPS']], errors='coerce')

    # 6. Treatment_Radiation
    if 'Treatment_Radiation' in active_features and col_map['Treatment_Radiation'] is not None:
        df_clean['Treatment_Radiation'] = pd.to_numeric(df_work[col_map['Treatment_Radiation']], errors='coerce')

    # 7. Treatment_Chemo
    if 'Treatment_Chemo' in active_features and col_map['Treatment_Chemo'] is not None:
        df_clean['Treatment_Chemo'] = pd.to_numeric(df_work[col_map['Treatment_Chemo']], errors='coerce')

    # Filter out rows with ANY missing value in active feature panel
    feature_cols_in_df = [c for c in df_clean.columns if c not in ['patient_id', 'label_12m', 'OS_MONTHS', 'OS_STATUS']]
    valid_features_mask = df_clean[feature_cols_in_df].notna().all(axis=1)
    
    df_clean_filtered = df_clean[valid_features_mask].copy().reset_index(drop=True)
    n_final = len(df_clean_filtered)
    n_dropped = n_initial - n_final

    exclusion_log = {
        "cohort": cohort_name,
        "n_initial": n_initial,
        "n_final": n_final,
        "n_dropped_missing_features": n_dropped,
        "active_features": active_features,
        "dropped_column_decisions": dropped_columns_decision
    }

    return df_clean_filtered, feature_cols_in_df, coverage_records, exclusion_log
