"""
FED-GBM v7 Survival Labeling Module.

Handles correct 12-month survival labeling with strict censoring handling:
- Confirmed survival >= 365 days  -> 1
- Confirmed death < 365 days      -> 0
- Censored before 365 days        -> None (EXCLUDED downstream)
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple


def compute_label_12m(row: Any) -> Optional[int]:
    """
    Computes 12-month survival binary classification label with strict censoring handling.

    Returns:
        1    -> confirmed survival >= 365 days
        0    -> confirmed death < 365 days
        None -> censored before 365 days (MUST be excluded downstream)
    """
    # 1. TCGA GDC style schema (vital_status + days_to_death / days_to_last_follow_up)
    if 'vital_status' in row and ('days_to_death' in row or 'days_to_last_follow_up' in row):
        vs = str(row.get('vital_status', '')).lower()
        if vs == 'dead':
            d = row.get('days_to_death')
            if pd.isna(d):
                return None
            return 1 if float(d) >= 365.0 else 0
        if vs == 'alive':
            f = row.get('days_to_last_follow_up')
            if pd.isna(f):
                return None
            return 1 if float(f) >= 365.0 else None  # censored < 12m -> EXCLUDE

    # 2. CGGA style schema (OS + Censor)
    censor_col = None
    if 'Censor' in row:
        censor_col = 'Censor'
    elif 'Censor (alive=0; dead=1)' in row:
        censor_col = 'Censor (alive=0; dead=1)'

    if 'OS' in row and censor_col is not None:
        os_days = row.get('OS')
        censored = row.get(censor_col)
        if pd.isna(os_days) or pd.isna(censored):
            return None
        os_days = float(os_days)
        censored = float(censored)
        if censored == 1.0:  # event observed (dead)
            return 1 if os_days >= 365.0 else 0
        else:  # censored (alive)
            return 1 if os_days >= 365.0 else None  # censored < 12m -> EXCLUDE

    # 3. cBioPortal / OS_STATUS + OS_MONTHS schema
    if 'OS_STATUS' in row and 'OS_MONTHS' in row:
        status = str(row.get('OS_STATUS', '')).lower()
        os_m = row.get('OS_MONTHS')
        if pd.isna(os_m):
            return None
        os_days = float(os_m) * 30.4375
        if '1:deceased' in status or 'dead' in status or '1:dead' in status:
            return 1 if os_days >= 365.0 else 0
        elif '0:living' in status or 'alive' in status or '0:alive' in status:
            return 1 if os_days >= 365.0 else None  # censored < 12m -> EXCLUDE

    return None


def audit_label_12m(df: pd.DataFrame, cohort_name: str) -> Dict[str, Any]:
    """
    Audits label_12m computation on a raw cohort DataFrame.

    Returns dict with audit stats:
        cohort, n_before, n_after, n_dropped, drop_pct, alive_at_12m_pct, dead_at_12m_pct
    """
    n_before = len(df)
    labels = df.apply(compute_label_12m, axis=1)

    valid_mask = labels.notna()
    n_after = int(valid_mask.sum())
    n_dropped = n_before - n_after
    drop_pct = (n_dropped / n_before * 100.0) if n_before > 0 else 0.0

    valid_labels = labels[valid_mask].astype(int)
    n_alive = int((valid_labels == 1).sum())
    n_dead = int((valid_labels == 0).sum())

    alive_pct = (n_alive / n_after * 100.0) if n_after > 0 else 0.0
    dead_pct = (n_dead / n_after * 100.0) if n_after > 0 else 0.0

    return {
        "cohort": cohort_name,
        "n_before": n_before,
        "n_after": n_after,
        "n_dropped": n_dropped,
        "drop_pct": round(drop_pct, 2),
        "alive_at_12m_pct": round(alive_pct, 2),
        "dead_at_12m_pct": round(dead_pct, 2),
    }
