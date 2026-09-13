"""
Kaplan-Meier Survival Analysis View — FED-GBM V3 Dashboard.

Loads generated KM curve PNGs from improved_experiments and displays them
with contextual log-rank p-value summaries and plain-language interpretations.
"""

import os
import glob
import streamlit as st
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))

KM_DIRS = [
    os.path.join(_ROOT, "results", "improved_experiments", "km_curves"),
    os.path.join(_ROOT, "experiments", "results", "improved_experiments", "km_curves"),
    os.path.join(_ROOT, "results", "v3", "improved_experiments", "km_curves"),
    os.path.join(_ROOT, "results", "archive", "experiment_b", "km_curves"),
    os.path.join(_ROOT, "results", "v3", "experiment_b", "km_curves"),
]


def _find_km_images() -> list:
    images = []
    for d in KM_DIRS:
        if os.path.isdir(d):
            images.extend(glob.glob(os.path.join(d, "km_*.png")))
    return sorted(images)


def render():
    st.markdown("""
    <div style="padding:10px 0 16px;">
        <div style="font-size:24px;font-weight:700;color:#0f172a;font-family:'Inter',sans-serif;">
            📈 Kaplan-Meier Patient Survival Curves
        </div>
        <div style="font-size:14px;color:#475569;margin-top:4px;">
            Visualizing how long patients survived based on model risk predictions (High Risk vs. Low Risk).
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Easy Language Explanation Banner
    st.info(
        "💡 **How to Read Kaplan-Meier Curves (Easy Language):**\n\n"
        "- **Horizontal Axis (X-axis):** Time in days/months following diagnosis.\n"
        "- **Vertical Axis (Y-axis):** Percentage of patients in the group who are still alive.\n"
        "- **🟢 Low-Risk Curve (Green/Blue):** Patients predicted to have better survival probability. Their line stays higher longer.\n"
        "- **🔴 High-Risk Curve (Red):** Patients predicted to be high risk. Their line drops earlier.\n"
        "- **Log-Rank p-value:** Measures if the difference between the two curves is real ($p < 0.05$ indicates strong statistical proof of risk separation).",
        icon="ℹ️"
    )

    km_images = _find_km_images()

    if not km_images:
        st.warning(
            "🔄 KM curves not yet generated on disk. Run `python -m src.fed_gbm.run_improved_experiments` "
            "or `python -m src.fed_gbm.cgga_experiment_b` from the project root to generate them.",
            icon="⚠️"
        )
        return

    # Group by cohort
    st.markdown(f"### 📊 Generated Patient Survival Plots ({len(km_images)} total)")

    cohort_groups = {}
    for img_path in km_images:
        fname = os.path.basename(img_path)
        parts = fname.replace("km_", "").replace(".png", "").split("_", 2)
        cohort = parts[0] if parts else "Unknown"
        cohort_groups.setdefault(cohort, []).append(img_path)

    for cohort_name, imgs in cohort_groups.items():
        st.markdown(f"#### Cohort: {cohort_name.replace('_', ' ')}")
        cols = st.columns(min(len(imgs), 2))
        for i, img_path in enumerate(imgs[:4]):
            with cols[i % 2]:
                model_name = os.path.basename(img_path).replace("km_", "").replace(".png", "").replace("_", " ")
                st.image(img_path, caption=f"KM Curve: {model_name}", use_column_width=True)
                st.markdown(
                    f"""
                    <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:6px;padding:10px;margin-bottom:16px;font-size:12px;color:#334155;">
                        <b>💡 Plot Interpretation ({model_name}):</b><br>
                        This curve demonstrates how effectively the <b>{model_name}</b> separates patient outcomes.
                        The vertical steps indicate patient death events over time. A wider gap between High Risk and Low Risk curves shows stronger predictive power.
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        if len(imgs) > 4:
            st.caption(f"... and {len(imgs) - 4} more survival curves available in output folder")

    # Literature comparison table
    st.markdown("---")
    st.markdown("### 📚 Literature Benchmark Comparison: Survival Risk Separation")
    comp_df = pd.DataFrame([
        {"Paper / Study": "PLOS ONE 2025 (Paper 11)", "Dataset Cohort": "GBM MRI Radiomics", "Risk Separation": "p < 0.0001", "Model Type": "Radiomics Nomogram"},
        {"Paper / Study": "Sci Reports 2017 (Paper 13)", "Dataset Cohort": "TCGA MRI (N=112)", "Risk Separation": "HR = 5.13", "Model Type": "CNN + LASSO Cox"},
        {"Paper / Study": "Cancers 2024 (Paper 5)", "Dataset Cohort": "GBM MRI + Genomic", "Risk Separation": "p < 0.05", "Model Type": "Radiomics + Cox"},
        {"Paper / Study": "**Our FED-GBM v3 Model**", "Dataset Cohort": "TCGA Test Cohort (N=115)", "Risk Separation": "Stratified p-value", "Model Type": "FedAvg / FedProx SurvivalMLP"},
    ])
    st.dataframe(comp_df, use_container_width=True, hide_index=True)
