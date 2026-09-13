"""
features.py — Feature Pipeline page for FED-GBM V3 dashboard (Light Theme).
"""
import streamlit as st
from app.utils.data_loader import load_config
from app.utils.styles import section_header


def render():
    st.markdown(section_header(
        "🧬 Clinical & Molecular Feature Panel",
        "Essential clinical variables and key neuro-oncology molecular biomarkers"
    ), unsafe_allow_html=True)

    cfg, _  = load_config()

    feats = cfg.get("selected_features", [
        "Age", "Gender", "Treatment_Radiation", "Treatment_Chemo",
        "KPS_Score", "IDH1_mutation", "MGMT_methylation"
    ]) if cfg else [
        "Age", "Gender", "Treatment_Radiation", "Treatment_Chemo",
        "KPS_Score", "IDH1_mutation", "MGMT_methylation"
    ]

    # ── Pipeline visualisation ──────────────────────────────────────────────
    st.markdown("### 7-Feature Multi-Modal Patient Processing Pipeline")

    steps = [
        ("Clinical Demographics", "Age (years) & Gender (Binary)", "#2563eb"),
        ("Therapeutic Interventions", "Standard Stupp Protocol: Radiation & Temozolomide Chemo", "#2563eb"),
        ("Functional Baseline", "KPS (Karnofsky Performance Score: 0-100)", "#2563eb"),
        ("Molecular Diagnostics", "IDH1 Mutation Status (Mutant vs Wild-type)", "#059669"),
        ("Epigenetic Diagnostics", "MGMT Promoter Methylation Status (Methylated vs Unmethylated)", "#059669"),
        ("Anti-Leakage Normalization", "Federated Sufficient Stats Scaler (Fitted on Train Splits Only)", "#d97706"),
        ("Final Feature Panel", "7 Key Clinical & Molecular Biomarkers", "#d97706"),
    ]

    for i, (step, detail, color) in enumerate(steps):
        is_last = (i == len(steps) - 1)
        st.markdown(f"""
        <div style="display:flex;flex-direction:column;align-items:center;margin-bottom:0;">
            <div style="background:#ffffff;border:1px solid #e2e8f0;border-left:4px solid {color};
                        border-radius:6px;padding:8px 20px;width:560px;text-align:left;box-shadow:0 1px 2px rgba(0,0,0,0.03);">
                <span style="font-size:13px;font-weight:600;color:#0f172a;font-family:'Inter',sans-serif;">{step}</span>
                <span style="font-size:11px;color:#64748b;font-family:'JetBrains Mono',monospace;
                             margin-left:12px;">{detail}</span>
            </div>
            {"" if is_last else '<div style="color:#94a3b8;font-size:16px;text-align:center;margin:1px 0;">↓</div>'}
        </div>
        """, unsafe_allow_html=True)

    # ── Final convergence ───────────────────────────────────────────────────
    st.markdown("""
    <div style="display:flex;justify-content:center;margin-top:14px;">
        <div style="background:#ffffff;border:1px solid #bbf7d0;border-radius:8px;
                    padding:14px 36px;text-align:center;box-shadow:0 1px 3px rgba(0,0,0,0.04);">
            <div style="font-size:14px;font-weight:700;color:#059669;margin-bottom:4px;">
                ✓ 7-Feature Clinical & Molecular Biomarker Panel Integration
            </div>
            <div style="font-size:12px;color:#475569;font-family:'JetBrains Mono',monospace;">
                Age + Gender + Radiation + Chemo + KPS Score + IDH1 Mutation + MGMT Methylation
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Feature schema cards ─────────────────────────────────────────────────
    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Clinical & Functional Features *(5)*")
        clinical_items = [
            ("Age", "Patient age in years (Continuous)"),
            ("Gender", "Binary indicator (0 = Female, 1 = Male)"),
            ("Treatment_Radiation", "Received RT (0 = No, 1 = Yes)"),
            ("Treatment_Chemo", "Received TMZ Chemo (0 = No, 1 = Yes)"),
            ("KPS_Score", "Karnofsky Score 40-100 (Higher = Better functional status)"),
        ]
        for name, desc in clinical_items:
            st.markdown(f"""
            <div style="background:#ffffff;border:1px solid #e2e8f0;border-left:4px solid #2563eb;
                        border-radius:6px;padding:10px 16px;margin-bottom:8px;box-shadow:0 1px 2px rgba(0,0,0,0.03);">
                <b style="font-family:'JetBrains Mono',monospace;font-size:13px;color:#0f172a;">{name}</b>
                <div style="font-size:11px;color:#64748b;margin-top:2px;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown("#### Molecular Biomarkers *(2)*")
        molecular_items = [
            ("MGMT_methylation", "MGMT Promoter Methylation (0 = Unmethylated, 1 = Methylated) — Key predictor for chemo sensitivity"),
            ("IDH1_mutation", "IDH1 Mutation Status (0 = Wild-type, 1 = Mutant) — Critical diagnostic & prognostic marker"),
        ]
        for name, desc in molecular_items:
            st.markdown(f"""
            <div style="background:#ffffff;border:1px solid #e2e8f0;border-left:4px solid #059669;
                        border-radius:6px;padding:10px 16px;margin-bottom:8px;box-shadow:0 1px 2px rgba(0,0,0,0.03);">
                <b style="font-family:'JetBrains Mono',monospace;font-size:13px;color:#0f172a;">{name}</b>
                <div style="font-size:11px;color:#64748b;margin-top:2px;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Anti-leakage banner ──────────────────────────────────────────────────
    st.markdown("""
    <div style="margin-top:20px;background:#fffbeb;
                border:1px solid #fde68a;border-left:4px solid #d97706;
                border-radius:8px;padding:14px 18px;font-size:13px;color:#92400e;">
        <b>Anti-Leakage Protocol:</b> Feature scalers and statistics are calculated exclusively on local hospital training splits (using FederatedSufficientStatsScaler). Test data remains strictly unseen during preprocessing.
    </div>
    """, unsafe_allow_html=True)

