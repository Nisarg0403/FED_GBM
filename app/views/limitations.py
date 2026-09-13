"""
limitations.py — Limitations page for FED-GBM V3 dashboard (Light Theme).
"""
import streamlit as st
from app.utils.styles import section_header


LIMITATIONS = [
    {
        "category": "Data & Cohort Size",
        "color": "#2563eb",
        "items": [
            "TCGA locked test set (N=115) provides rigorous internal evaluation for 12-month survival prediction.",
            "Public cohorts (TCGA, UPenn, CGGA) simulate multi-hospital clients — this is a federated simulation framework across 1,397 total patients.",
            "Cross-cohort distribution shift (scanner protocols, patient selection, regional demographics) accounts for observed performance differences across US and Asian hospital nodes.",
            "Retrospective cohort analysis: all evaluations use historical clinical datasets.",
        ]
    },
    {
        "category": "Privacy Accounting Bounds",
        "color": "#d97706",
        "items": [
            "Epsilon values report formal differential privacy bounds under RDP accounting.",
            "Client-level DP uses replacement sensitivity across federated client datasets.",
            "Patient-level DP epsilon values scale inversely with client dataset size under Poisson subsampling.",
        ]
    },
    {
        "category": "Model Architecture & Methodology",
        "color": "#059669",
        "items": [
            "SurvivalMLP [128, 64] is a compact MLP optimized for clinical & molecular tabular data.",
            "Feature scaling statistics are computed exclusively via local hospital training splits (FederatedSufficientStatsScaler) to guarantee strict anti-leakage compliance.",
            "Prediction target is 12-month overall survival classification (survive ≥ 365 days).",
            "FedProx proximal regularization parameter μ=0.01 controls local weight drift on non-IID data distributions.",
        ]
    },
    {
        "category": "Regulatory & Clinical Scope",
        "color": "#dc2626",
        "items": [
            "This software is a research demonstration and must NOT be used for direct clinical diagnosis or patient care.",
            "No regulatory approval (FDA, CE marking, etc.) has been sought.",
            "All survival prediction models and risk calculators are intended exclusively for academic research.",
        ]
    },
]



def render():
    st.markdown(section_header(
        "💡 Future Roadmap & Next Steps",
        "Transparent academic limitations & technical development roadmap"
    ), unsafe_allow_html=True)

    st.markdown("""
    <div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:8px;
                padding:16px 20px;margin-bottom:24px;box-shadow:0 1px 3px rgba(0,0,0,0.03);">
        <div style="color:#334155;font-size:13px;line-height:1.7;">
            The following roadmap and limitations provide transparent reporting on current capabilities
            and outline future integration of full 3D MRI radiomics and MGMT/IDH molecular markers.
        </div>
    </div>
    """, unsafe_allow_html=True)

    for section in LIMITATIONS:
        st.markdown(f"""
        <div style="margin-bottom:20px;">
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;">
                <div style="width:4px;height:20px;background:{section['color']};border-radius:2px;"></div>
                <div style="font-size:15px;font-weight:700;color:{section['color']};
                            font-family:'Inter',sans-serif;">{section['category']}</div>
            </div>
        """, unsafe_allow_html=True)

        for item in section["items"]:
            st.markdown(f"""
            <div style="display:flex;align-items:flex-start;gap:10px;
                        background:#ffffff;border:1px solid #e2e8f0;border-radius:6px;
                        padding:12px 16px;margin-bottom:8px;box-shadow:0 1px 2px rgba(0,0,0,0.03);">
                <span style="color:{section['color']};font-size:16px;min-width:16px;margin-top:-1px;">•</span>
                <span style="color:#334155;font-size:13px;line-height:1.7;">{item}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # ── Peer-Review Methodological Rationales ──────────────────────────────
    st.markdown("---")
    st.markdown("### 🎓 Peer-Review Methodological Defense (IEEE / Biomedical Standards)")

    peer_rationales = [
        {
            "title": "1. External Calibration Slope & Temperature Scaling",
            "badge": "Clinical Safety",
            "color": "#0284c7",
            "content": r"""
            <b>Critique:</b> External zero-shot testing often yields overconfident probabilities with low raw calibration slopes (< 1.0).<br>
            <b>Solution & Methodology:</b> We implement post-hoc <b>Localized Temperature Scaling ($\hat{p} = \sigma(z/T)$)</b>. 
            By fitting a scalar temperature $T$ on held-out validation logits, zero-shot calibration slopes improve significantly (e.g., CGGA zero-shot slope recalibrates from raw 1.49 to <b>1.1426</b> with $T=0.765$, achieving an ECE of 0.0813).
            """
        },
        {
            "title": "2. PyTorch MLP vs. Gradient Boosted Decision Trees (GBDT)",
            "badge": "Privacy & Aggregation",
            "color": "#7c3aed",
            "content": r"""
            <b>Critique:</b> GBDTs (LightGBM/XGBoost) frequently outperform MLPs on tabular datasets.<br>
            <b>Solution & Rationale:</b> Federated parameter aggregation (FedAvg/FedProx) requires continuous, smooth gradient updates across node iterations. GBDT tree structures cannot be directly averaged across decentralized hospitals without complex private tree-splitting protocols. Furthermore, <b>PyTorch MLPs support formal DP-SGD differential privacy</b> via Opacus (Rényi Differential Privacy), enabling rigorous $(\varepsilon, \delta)$ accounting.
            """
        },
        {
            "title": "3. WHO 2021 Biomarker Classification Nuance",
            "badge": "Biomarker Provenance",
            "color": "#059669",
            "content": r"""
            <b>Critique:</b> The WHO 2021 CNS classification re-defines IDH-mutant astrocytomas separately from IDH-wildtype glioblastomas.<br>
            <b>Solution & Rationale:</b> Historical patient registries (TCGA, UPenn, CGGA spanning 2013–2020) recorded all high-grade gliomas under Grade IV GBM clinical diagnostic codes. Retaining <b><code>IDH1_mutation</code></b> as an explicit input biomarker maintains cross-cohort backward compatibility while giving the model direct access to this crucial prognostic marker.
            """
        },
        {
            "title": "4. 12-Month Classification vs. Time-to-Event Survival Analysis",
            "badge": "Target Formulation",
            "color": "#d97706",
            "content": r"""
            <b>Critique:</b> Binary survival classification at 12 months ignores right-censoring in survival data.<br>
            <b>Solution & Rationale:</b> 12-month overall survival ($S \ge 365\text{ days}$) provides a clear, standardized clinical decision threshold for 1-year mortality risk stratification. Patients censored prior to 365 days were strictly excluded during cohort curation to prevent label noise. Future iterations expand this framework to federated DeepSurv Cox proportional hazards for continuous C-index trajectory estimation.
            """
        },
        {
            "title": "5. Extent of Resection (EOR) Exclusion",
            "badge": "Data Quality",
            "color": "#dc2626",
            "content": r"""
            <b>Critique:</b> Extent of Resection (Gross Total Resection vs Partial) is a key prognostic factor in glioblastoma.<br>
            <b>Solution & Rationale:</b> EOR recording standards differed significantly across international cohorts (TCGA used surgical percentage, UPenn recorded GTR/STR/Biopsy, CGGA had ~40% missing entries). To prevent noisy missing-data imputation and preserve strict multi-center feature alignment across US and Asian hospitals, EOR was excluded from the primary 7-feature panel.
            """
        }
    ]

    for item in peer_rationales:
        with st.expander(f"{item['title']} [{item['badge']}]"):
            st.markdown(f"""
            <div style="background:#f8fafc;border-left:4px solid {item['color']};border-radius:4px;padding:12px 16px;font-size:13px;color:#334155;line-height:1.7;">
                {item['content']}
            </div>
            """, unsafe_allow_html=True)


    # ── Global disclaimers ───────────────────────────────────────────────────
    st.markdown("""
    <div style="background:#fef2f2;border:1px solid #fecaca;
                border-left:4px solid #dc2626;border-radius:8px;padding:14px 20px;margin-top:12px;">
        <div style="font-size:14px;font-weight:700;color:#dc2626;margin-bottom:6px;">
            Global Research Disclaimer
        </div>
        <div style="font-size:13px;color:#7f1d1d;line-height:1.7;">
            Research demonstration system. Not intended for clinical diagnosis, patient care, or treatment decisions.
        </div>
    </div>
    """, unsafe_allow_html=True)
