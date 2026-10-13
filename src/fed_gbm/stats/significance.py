"""
Statistical Significance Testing Module for FED-GBM v5.

Provides hypothesis testing, effect size, confidence interval, and p-value correction:
1. Paired t-test + Cohen's d
2. Wilcoxon signed-rank test
3. Non-parametric bootstrap confidence intervals
4. Bonferroni multiple testing correction
"""

import numpy as np
from scipy import stats


def paired_ttest(seed_results_a: list, seed_results_b: list) -> tuple:
    """
    Computes paired t-test and Cohen's d effect size between two paired seed result arrays.
    Returns: (t_stat, p_value, cohens_d)
    """
    a = np.array(seed_results_a, dtype=float)
    b = np.array(seed_results_b, dtype=float)

    diff = a - b
    if len(diff) < 2 or np.all(diff == 0):
        return 0.0, 1.0, 0.0

    res = stats.ttest_rel(a, b)
    t_stat = float(res.statistic)
    p_value = float(res.pvalue)

    std_diff = np.std(diff, ddof=1)
    cohens_d = float(np.mean(diff) / std_diff) if std_diff > 1e-12 else 0.0

    return t_stat, p_value, cohens_d


def wilcoxon_test(seed_results_a: list, seed_results_b: list) -> tuple:
    """
    Computes Wilcoxon signed-rank test for paired samples.
    Returns: (stat, p_value)
    """
    a = np.array(seed_results_a, dtype=float)
    b = np.array(seed_results_b, dtype=float)

    diff = a - b
    if np.all(diff == 0):
        return 0.0, 1.0

    try:
        res = stats.wilcoxon(a, b)
        return float(res.statistic), float(res.pvalue)
    except Exception:
        return 0.0, 1.0


def bootstrap_ci(metric_values: list, n_boot: int = 1000, ci: float = 0.95, seed: int = 42) -> tuple:
    """
    Computes non-parametric bootstrap confidence interval for metric_values.
    Returns: (lower, upper)
    """
    vals = np.array(metric_values, dtype=float)
    if len(vals) == 0:
        return 0.0, 0.0

    rng = np.random.RandomState(seed)
    boot_means = []
    n = len(vals)
    for _ in range(n_boot):
        sample = rng.choice(vals, size=n, replace=True)
        boot_means.append(np.mean(sample))

    alpha = 1.0 - ci
    lower = float(np.percentile(boot_means, 100.0 * (alpha / 2.0)))
    upper = float(np.percentile(boot_means, 100.0 * (1.0 - alpha / 2.0)))
    return lower, upper


def bonferroni_correction(p_values: list, n_comparisons: int = None) -> list:
    """
    Applies Bonferroni correction: p_adj = min(1.0, p * n_comparisons).
    Returns list of adjusted p-values.
    """
    if n_comparisons is None:
        n_comparisons = len(p_values)

    adjusted = [float(min(1.0, p * n_comparisons)) for p in p_values]
    return adjusted
