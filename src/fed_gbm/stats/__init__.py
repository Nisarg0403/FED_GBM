"""
Statistical analysis package for FED-GBM v5.
"""
from .significance import (
    paired_ttest,
    wilcoxon_test,
    bootstrap_ci,
    bonferroni_correction,
)
