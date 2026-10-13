"""
Explainability & Interpretability Modules for FED-GBM
"""
from fed_gbm.explainability.shap_explainer import (
    compute_shap_values,
    generate_shap_summary_plot,
    generate_patient_waterfall_plot,
)

__all__ = [
    "compute_shap_values",
    "generate_shap_summary_plot",
    "generate_patient_waterfall_plot",
]
