"""
performance.py — Model Performance page for FED-GBM V3 dashboard.
Light theme UI with plain language interpretations below all charts.
"""
import os
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

import app.utils.data_loader as dl_mod

load_aggregate = getattr(dl_mod, "load_aggregate", None)
load_improved_aggregate = getattr(dl_mod, "load_improved_aggregate", None)

if load_improved_aggregate is None:
    def load_improved_aggregate():
        p = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "results", "v3", "improved_experiments", "aggregate_results.csv"))
        if os.path.exists(p):
            return pd.read_csv(p), ""
        return None, "File not found"

from app.utils.styles import section_header, badge

METRICS = [
    ("roc_auc",    "ROC-AUC (Overall Accuracy)", False),
    ("pr_auc",     "PR-AUC (Precision-Recall)",  False),
    ("accuracy",   "Accuracy",                   False),
    ("f1",         "F1 Score (Balanced Metric)", False),
    ("sensitivity","Sensitivity (Catch High Risk)", False),
    ("specificity","Specificity (Catch Low Risk)",  False),
    ("brier_score","Brier Score (Lower is Better)", True),
    ("ece",        "Calibration Error (Lower is Better)", True),
]

MODEL_GROUPS = {
    "Centralized Logistic Regression": "Centralized",
    "Centralized LR":                  "Centralized",
    "Centralized Random Forest":       "Centralized",
    "Centralized RF":                  "Centralized",
    "Centralized XGBoost":             "Centralized",
    "Centralized MLP":                 "Centralized",
    "FedAvg MLP":                      "Federated",
    "FedProx MLP (mu=0.01)":          "Federated",
    "Patient DP (sigma=0.05)":         "Patient DP",
    "Patient DP (sigma=0.10)":         "Patient DP",
    "Patient DP (sigma=0.20)":         "Patient DP",
    "Client DP (sigma=0.05)":          "Client DP",
    "Client DP (sigma=0.10)":          "Client DP",
    "Client DP (sigma=0.20)":          "Client DP",
}

GROUP_COLORS = {
    "Centralized": "#2563eb",
    "Federated":   "#059669",
    "Patient DP":  "#d97706",
    "Client DP":   "#dc2626",
}


def render():
    st.markdown(section_header(
        "📊 Model Accuracy & Benchmark Results",
        "Performance evaluation across 3 real clinical patient cohorts (TCGA, UPenn, CGGA) & 6 AI models."
    ), unsafe_allow_html=True)

    # Option to toggle between Real Clinical Data & Phase 7.4 Protocol Verification Data
    data_source = st.radio(
        "Data Mode",
        ["Real Patient Datasets (TCGA, UPenn, CGGA)", "Phase 7.4 Protocol Benchmark"],
        horizontal=True
    )

    if "Real Patient" in data_source:
        agg, err = load_improved_aggregate()
        if agg is None:
            st.error(f"⛔ Cannot load real clinical dataset results — {err}")
            return
        cohort_col = "cohort"
        available_cohorts = list(agg["cohort"].unique())
    else:
        agg, err = load_aggregate()
        if agg is None:
            st.error(f"⛔ Cannot load Phase 7.4 aggregate results — {err}")
            return
        cohort_col = "eval_set"
        available_cohorts = list(agg["eval_set"].unique())

    # ── Controls ────────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns([2, 2, 3])
    with c1:
        selected_cohort = st.selectbox("Select Evaluation Cohort", available_cohorts, key="perf_cohort")
    with c2:
        metric_labels = [m[1] for m in METRICS]
        sel_metric = st.selectbox("Primary Metric", metric_labels, key="perf_metric")
    with c3:
        group_filter = st.multiselect(
            "Filter by Model Group",
            ["Centralized", "Federated", "Patient DP", "Client DP"],
            default=["Centralized", "Federated", "Patient DP", "Client DP"],
            key="perf_group"
        )

    df = agg[agg[cohort_col] == selected_cohort].copy()
    df["group"] = df["model"].map(MODEL_GROUPS).fillna("Centralized")
    df = df[df["group"].isin(group_filter)]

    # Find the metric config
    metric_cfg = next(m for m in METRICS if m[1] == sel_metric)
    metric_col  = metric_cfg[0]
    lower_better = metric_cfg[2]
    mean_col = f"{metric_col}_mean"
    std_col  = f"{metric_col}_std"

    if mean_col not in df.columns:
        st.warning(f"Column {mean_col} not found in aggregate CSV.")
        return

    df_sorted = df.sort_values(mean_col, ascending=lower_better)

    # ── Tab: Chart / Table ──────────────────────────────────────────────────
    tab_chart, tab_table, tab_compare = st.tabs(["📊 Bar Chart", "📋 Ranking Table", "⚖️ Centralized vs Federated"])

    with tab_chart:
        colors = [GROUP_COLORS.get(g, "#64748b") for g in df_sorted["group"]]
        text_vals = [f"{v:.4f}" for v in df_sorted[mean_col]]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df_sorted["model"],
            y=df_sorted[mean_col],
            text=text_vals,
            textposition="outside",
            textfont=dict(size=11, color="#0f172a", family="Inter"),
            cliponaxis=False,
            error_y=dict(type="data", array=df_sorted[std_col].fillna(0).tolist() if std_col in df_sorted.columns else None, visible=True,
                         color="#64748b", thickness=1.5, width=4),
            marker_color=colors,
            hovertemplate="<b>%{x}</b><br>" + f"{sel_metric}: %{{y:.4f}}<extra></extra>",
        ))
        arrow = "↓ lower is better" if lower_better else "↑ higher is better"
        y_max = float(df_sorted[mean_col].max())
        y_padding = y_max * 0.15 if y_max > 0 else 0.1
        fig.update_layout(
            title=dict(text=f"{sel_metric} — {selected_cohort} ({arrow})", font=dict(color="#0f172a", size=14, family="Inter")),
            plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
            font=dict(color="#334155", family="Inter"),
            xaxis=dict(tickangle=-35, gridcolor="#f1f5f9", title=""),
            yaxis=dict(gridcolor="#e2e8f0", title=sel_metric,
                       range=[0, y_max + y_padding]),
            margin=dict(l=10, r=10, t=50, b=120),
            height=460,
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)
        _group_legend()

        top_model = df_sorted.iloc[0]["model"]
        top_score = df_sorted.iloc[0][mean_col]
        st.markdown(
            f"""
            <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:14px;margin-top:16px;font-size:13px;color:#334155;">
                <b>💡 Easy Language Chart Explanation:</b><br>
                This bar chart compares how accurately different AI models predict 12-month survival for <b>{selected_cohort}</b> patients.
                Taller bars indicate higher prediction accuracy.
                <b>{top_model} ({top_score:.4f})</b> achieved the top score on <b>{selected_cohort}</b> data.
            </div>
            """,
            unsafe_allow_html=True
        )

    with tab_table:
        display_cols = ["model", "group"]
        for metric_col_p, metric_lbl, _ in METRICS:
            sc = f"{metric_col_p}_str"
            if sc in df.columns:
                display_cols.append(sc)
            elif f"{metric_col_p}_mean" in df.columns:
                display_cols.append(f"{metric_col_p}_mean")

        display_df = df_sorted[display_cols].copy()
        rename = {"model": "Model", "group": "Group"}
        display_df = display_df.rename(columns=rename)
        st.dataframe(display_df, hide_index=True, use_container_width=True)

        st.markdown(
            """
            <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:12px;margin-top:10px;font-size:13px;color:#334155;">
                <b>💡 Easy Language Table Explanation:</b> Each row shows one AI model's score across 5 random test runs. Higher ROC-AUC means better prediction accuracy.
            </div>
            """,
            unsafe_allow_html=True
        )

    with tab_compare:
        _centralized_vs_federated(agg, metric_cfg, selected_cohort, cohort_col)


def _group_legend():
    html = "<div style='display:flex;gap:20px;margin-top:8px;flex-wrap:wrap;'>"
    for grp, col in GROUP_COLORS.items():
        html += f"<div style='display:flex;align-items:center;gap:6px;'>"
        html += f"<div style='width:12px;height:12px;background:{col};border-radius:2px;'></div>"
        html += f"<span style='font-size:13px;color:#334155;font-weight:600;'>{grp}</span></div>"
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def _centralized_vs_federated(agg: pd.DataFrame, metric_cfg: tuple, selected_cohort: str, cohort_col: str):
    metric_col, metric_lbl, lower_better = metric_cfg
    mean_col = f"{metric_col}_mean"

    df = agg[agg[cohort_col] == selected_cohort].copy()
    df["group"] = df["model"].map(MODEL_GROUPS).fillna("Centralized")

    fig = go.Figure()
    for grp, grp_color in GROUP_COLORS.items():
        gdf = df[df["group"] == grp]
        if gdf.empty:
            continue
        fig.add_trace(go.Scatter(
            x=gdf["model"],
            y=gdf[mean_col],
            mode="markers",
            name=grp,
            marker=dict(color=grp_color, size=12, symbol="circle"),
            hovertemplate="<b>%{x}</b><br>" + f"{metric_lbl}: %{{y:.4f}}<extra>{grp}</extra>",
        ))

    arrow = "↓ lower is better" if lower_better else "↑ higher is better"
    fig.update_layout(
        title=dict(text=f"{metric_lbl} — Group Comparison ({arrow})",
                   font=dict(color="#0f172a", size=14, family="Inter")),
        plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
        font=dict(color="#334155", family="Inter"),
        xaxis=dict(tickangle=-35, gridcolor="#f1f5f9", title=""),
        yaxis=dict(gridcolor="#e2e8f0", title=metric_lbl),
        legend=dict(bgcolor="#ffffff", bordercolor="#cbd5e1", borderwidth=1,
                    font=dict(color="#0f172a", size=12)),
        margin=dict(l=10, r=10, t=50, b=120),
        height=430,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        """
        <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:12px;margin-top:10px;font-size:13px;color:#334155;">
            <b>💡 Easy Language Group Comparison:</b> Centralized models (blue) train on all data stored together. Federated models (green) train collaboratively across hospitals without sharing private patient data.
        </div>
        """,
        unsafe_allow_html=True
    )
