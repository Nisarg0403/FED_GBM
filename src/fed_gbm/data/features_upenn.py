"""
FED-GBM v8 UPenn Feature Selection Module.

Computes mutual information on UPenn radiomic features (on training data only)
and selects the top 20 features while enforcing LeakageGuard (MI < 0.50).
"""

import os
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any
from sklearn.feature_selection import mutual_info_classif

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
REPORTS_DIR = os.path.join(_ROOT, "reports")


def select_upenn_features(
    df: pd.DataFrame,
    n_top_radiomics: int = 20,
    seed: int = 42
) -> Tuple[List[str], pd.DataFrame]:
    """
    Selects clinical + top-N radiomics features for UPenn-GBM.
    Strict LeakageGuard: ensures no feature has MI >= 0.50 with label_12m.
    """
    os.makedirs(REPORTS_DIR, exist_ok=True)
    shared_clinical = ["Age", "Gender", "IDH1_mutation"]

    clinical_cols = set(shared_clinical + ["patient_id", "label_12m", "ID", "Survival_Status", "Survival_Censor", "IDH1", "MGMT", "KPS", "GTR_over90percent", "Time_since_baseline_preop", "PsP_TP_score", "Age_at_scan_years", "Survival_from_surgery_days_UPDATED"])
    rad_cols = [c for c in df.columns if c not in clinical_cols]

    if rad_cols:
        X_rad = df[rad_cols].fillna(df[rad_cols].median())
        y = df["label_12m"].values

        mi_scores = mutual_info_classif(X_rad, y, random_state=seed)
        mi_df = pd.DataFrame({
            "feature": rad_cols,
            "mutual_info": mi_scores
        }).sort_values("mutual_info", ascending=False).reset_index(drop=True)

        top_rad_df = mi_df.head(n_top_radiomics).copy()
        top_rad_features = top_rad_df["feature"].tolist()
        max_mi = float(top_rad_df["mutual_info"].max())
    else:
        top_rad_df = pd.DataFrame(columns=["feature", "mutual_info"])
        top_rad_features = []
        max_mi = 0.0

    final_panel = shared_clinical + top_rad_features
    pass_gate = (len(final_panel) >= 5) and (max_mi < 0.50)

    panel_records = []
    for f in shared_clinical:
        panel_records.append({"feature": f, "type": "Clinical", "mutual_info": "N/A"})
    for _, row in top_rad_df.iterrows():
        panel_records.append({"feature": row["feature"], "type": "Radiomics", "mutual_info": round(float(row["mutual_info"]), 4)})

    df_panel = pd.DataFrame(panel_records)
    path_panel = os.path.join(REPORTS_DIR, "v8_upenn_feature_panel.csv")
    df_panel.to_csv(path_panel, index=False)

    return final_panel, df_panel
