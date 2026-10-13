"""
FED-GBM v9 Significance Testing Script (Task 10).

Executes paired t-tests, Wilcoxon signed-rank tests, Cohen's d effect size,
and Bonferroni p-value corrections across 10 evaluation seeds.
Saves reports/v9_significance.csv.
"""

import os
import scipy.stats as stats
import numpy as np
import pandas as pd
from typing import Dict, Any, List

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
REPORTS_DIR = os.path.join(_ROOT, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)


def run_significance_tests(seed_aucs_dict: Dict[str, List[float]]):
    print("======================================================================")
    print("           TASK 10: SIGNIFICANCE TESTING (10 SEEDS)")
    print("======================================================================")

    sig_pairs = [
        ("CA-FedAvg", "FedAvg"),
        ("CA-FedAvg", "FedProx"),
        ("CA-FedAvg", "HAPA"),
        ("FedProx", "FedAvg"),
        ("CA-FedAvg", "Centralized_XGBoost")
    ]

    sig_v9_results = []
    n_tests = len(sig_pairs)

    for m1, m2 in sig_pairs:
        arr1 = np.array(seed_aucs_dict[m1])
        arr2 = np.array(seed_aucs_dict[m2])

        diffs = arr1 - arr2
        mean_diff = float(np.mean(diffs))

        if np.all(diffs == 0.0):
            t_stat, p_val_t = 0.0, 1.0
        else:
            t_stat, p_val_t = stats.ttest_rel(arr1, arr2)

        try:
            w_stat, p_val_w = stats.wilcoxon(diffs)
        except Exception:
            w_stat, p_val_w = 0.0, 1.0

        std_diff = float(np.std(diffs, ddof=1))
        cohens_d = (mean_diff / std_diff) if std_diff > 1e-8 else 0.0
        bonf_p = min(1.0, float(p_val_t) * n_tests)

        sig_v9_results.append({
            "comparison": f"{m1} vs {m2}",
            "mean_diff": round(mean_diff, 4),
            "t_stat": round(float(t_stat), 4),
            "p_value_t_test": round(float(p_val_t), 4),
            "wilcoxon_p_value": round(float(p_val_w), 4),
            "cohens_d": round(cohens_d, 4),
            "bonferroni_p": round(bonf_p, 4),
            "significant_at_05": bool(bonf_p < 0.05)
        })

    df_sig_v9 = pd.DataFrame(sig_v9_results)
    path_sig_v9 = os.path.join(REPORTS_DIR, "v9_significance.csv")
    df_sig_v9.to_csv(path_sig_v9, index=False)
    print(df_sig_v9.to_string())

    return df_sig_v9
