"""
privacy_page.py — Privacy Analysis page for FED-GBM V3 dashboard (Light Theme).
"""
import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from app.utils.data_loader import load_privacy, load_improved_aggregate, load_aggregate
from app.utils.styles import section_header, disclaimer_banner, badge


def render():
    st.markdown(section_header(
        "🔒 Patient Privacy & Data Protection",
        "Formal (ε, δ)-Differential Privacy bounds for Patient-Level DP-SGD and Client-Level DP-FedAvg"
    ), unsafe_allow_html=True)

    priv, priv_err = load_privacy()
    agg,  agg_err  = load_improved_aggregate()
    if agg is None:
        agg, agg_err = load_aggregate()

    if priv is None:
        st.error(f"⛔ Privacy results unavailable — {priv_err}")
        return

    # Easy Language Banner
    st.info(
        "💡 **What is Differential Privacy? (Easy Language):**\n\n"
        "Differential privacy adds controlled mathematical noise during training so that no single patient's sensitive medical data can ever be reconstructed or leaked from the AI model weights.",
        icon="ℹ️"
    )

    # ── Two-column header ───────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    # ── Patient-Level DP ────────────────────────────────────────────────────
    with col1:
        st.markdown("""
        <div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:8px;padding:18px 20px;box-shadow:0 1px 3px rgba(0,0,0,0.03);">
            <div style="font-size:12px;color:#d97706;font-family:'JetBrains Mono',monospace;
                        letter-spacing:0.06em;text-transform:uppercase;margin-bottom:10px;font-weight:700;">
                Patient-Level DP (DP-SGD)
            </div>
            <div style="color:#334155;font-size:13px;line-height:1.6;">
                Privacy unit: <b>individual patient record</b><br>
                Mechanism: Opacus Poisson RDP<br>
                Clipping bound: C = 1.0<br>
                δ = 1×10⁻⁵
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Client-Level DP ─────────────────────────────────────────────────────
    with col2:
        st.markdown("""
        <div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:8px;padding:18px 20px;box-shadow:0 1px 3px rgba(0,0,0,0.03);">
            <div style="font-size:12px;color:#2563eb;font-family:'JetBrains Mono',monospace;
                        letter-spacing:0.06em;text-transform:uppercase;margin-bottom:10px;font-weight:700;">
                Client-Level DP (DP-FedAvg)
            </div>
            <div style="color:#334155;font-size:13px;line-height:1.6;">
                Privacy unit: <b>entire hospital dataset</b><br>
                Mechanism: Closed-Form Gaussian RDP<br>
                ΔS (replacement sensitivity) = 1.7132<br>
                δ = 1×10⁻⁵ | Rounds T = 30
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Privacy Tradeoff Chart
    st.markdown("---")
    st.markdown("### Privacy vs. Accuracy Trade-Off (Epsilon Curve)")

    if agg is not None and priv is not None:
        cohort_col = "cohort" if "cohort" in agg.columns else "eval_set"
        available_cohorts = list(agg[cohort_col].unique())
        eval_set = st.selectbox("Select Evaluation Set", available_cohorts, key="priv_eval")
        df_p = agg[agg[cohort_col] == eval_set].copy()

        client_key = "TCGA" if "TCGA" in eval_set else ("UPenn" if "UPenn" in eval_set else "CGGA")
        patient_dp_rows = priv[(priv["level"] == "Patient-Level DP") & (priv["client"] == client_key)].copy()
        client_dp_rows = priv[priv["level"] == "Client-Level DP"].copy()

        fig = go.Figure()

        # Add Patient DP trace
        p_eps, p_auc, p_labels = [], [], []
        for _, row in patient_dp_rows.iterrows():
            sigma_str = f"sigma={row['sigma']:.2f}"
            m_row = df_p[df_p["model"].str.contains(f"Patient DP.*{sigma_str}")]
            if not m_row.empty:
                p_eps.append(row["epsilon"])
                p_auc.append(m_row.iloc[0]["roc_auc_mean"])
                p_labels.append(f"Patient DP (σ={row['sigma']})")

        if p_eps:
            fig.add_trace(go.Scatter(
                x=p_eps, y=p_auc,
                mode="markers+lines+text",
                name="Patient-Level DP (DP-SGD)",
                marker=dict(color="#d97706", size=10),
                line=dict(color="#d97706", width=2),
                text=[f"σ={s}" for s in [0.05, 0.1, 0.2][:len(p_eps)]],
                textposition="top center",
                textfont=dict(size=11, color="#d97706"),
            ))

        # Add Client DP trace
        c_eps, c_auc, c_labels = [], [], []
        for _, row in client_dp_rows.iterrows():
            sigma_str = f"sigma={row['sigma']:.2f}"
            m_row = df_p[df_p["model"].str.contains(f"Client DP.*{sigma_str}")]
            if not m_row.empty:
                c_eps.append(row["epsilon"])
                c_auc.append(m_row.iloc[0]["roc_auc_mean"])
                c_labels.append(f"Client DP (σ={row['sigma']})")

        if c_eps:
            fig.add_trace(go.Scatter(
                x=c_eps, y=c_auc,
                mode="markers+lines+text",
                name="Client-Level DP (DP-FedAvg)",
                marker=dict(color="#2563eb", size=10),
                line=dict(color="#2563eb", width=2, dash="dash"),
                text=[f"σ={s}" for s in [0.05, 0.1, 0.2][:len(c_eps)]],
                textposition="top center",
                textfont=dict(size=11, color="#2563eb"),
            ))

        fig.update_layout(
            plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
            font=dict(color="#334155", family="Inter"),
            xaxis=dict(type="log", title="ε Epsilon (log scale — Lower = Stronger Privacy)", gridcolor="#f1f5f9"),
            yaxis=dict(title="ROC-AUC Score", gridcolor="#e2e8f0", range=[0.3, 1.0]),
            legend=dict(bgcolor="#ffffff", bordercolor="#cbd5e1", borderwidth=1,
                        font=dict(color="#0f172a", size=11)),
            height=400,
            margin=dict(l=10, r=10, t=20, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        """
        <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:6px;padding:12px;margin-top:10px;font-size:13px;color:#334155;">
            <b>💡 Privacy Trade-off Interpretation (Easy Language):</b> Epsilon (ε) measures the privacy budget.
            Smaller values of ε mean stronger privacy protection, but adding more noise slightly reduces prediction accuracy.
        </div>
        """,
        unsafe_allow_html=True
    )

