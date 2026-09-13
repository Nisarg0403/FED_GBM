"""
app.py — FED-GBM V3 Research Dashboard
=======================================
Dynamic view reloading enabled for real-time Light Theme & improved experiments.
"""
import sys
import os
import importlib

# Ensure project root & app directory are in sys.path
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import streamlit as st

try:
    import app.utils.paths as paths
    import app.utils.styles as styles
    import app.utils.data_loader as data_loader
except ModuleNotFoundError:
    import utils.paths as paths
    import utils.styles as styles
    import utils.data_loader as data_loader

importlib.reload(paths)
importlib.reload(styles)
importlib.reload(data_loader)

# ─── Page configuration ────────────────────────────────────────────────────
st.set_page_config(
    page_title="FED-GBM V3 | Brain Tumor AI Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Apply global CSS ───────────────────────────────────────────────────────
st.markdown(styles.GLOBAL_CSS, unsafe_allow_html=True)

# ─── Navigation map ─────────────────────────────────────────────────────────
PAGES = {
    "🏠  Home & Project Summary":               "overview",
    "🧠  Patient Survival Predictor":           "prediction_demo",
    "📈  Kaplan-Meier Survival Curves":         "survival_analysis",
    "🔍  Feature Importance & Insights (SHAP)": "explainability",
    "📊  Accuracy & Benchmark Results":         "performance",
    "🌍  Multi-Hospital Results (CGGA/UPenn)":  "generalization",
    "🌐  Multi-Hospital Collaborative AI":      "federated",
    "🔒  Patient Privacy & Protection":         "privacy_page",
    "🗄️  Patient Datasets (TCGA, UPenn, CGGA)":  "dataset",
    "🧬  Tumor Features & Health Markers":       "features",
    "⚙️  Experiment Setup & Verification":      "reproducibility",
    "💡  Future Roadmap & Next Steps":          "limitations",
}

# ─── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:12px 0 4px;">
        <div style="font-size:20px;font-weight:700;color:#0f172a;
                    font-family:'Inter',sans-serif;letter-spacing:-0.01em;">
            🧠 FED-GBM V3
        </div>
        <div style="font-size:12px;color:#475569;font-family:'JetBrains Mono',monospace;
                    margin-top:2px;letter-spacing:0.03em;font-weight:600;">Brain Tumor AI Dashboard</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr style="border-color:#e2e8f0;margin:10px 0;"/>', unsafe_allow_html=True)

    # Navigation
    page_label = st.radio(
        "Navigation",
        list(PAGES.keys()),
        index=0,
        key="nav",
        label_visibility="collapsed",
    )

    st.markdown('<hr style="border-color:#e2e8f0;margin:10px 0;"/>', unsafe_allow_html=True)

    # System status
    p74_ok, p74_err = data_loader.phase7_final_available()

    if p74_ok:
        st.markdown("""
        <div style="display:flex;align-items:center;gap:8px;padding:4px 0;">
            <div style="width:8px;height:8px;background:#059669;border-radius:50%;"></div>
            <span style="font-size:12px;color:#059669;font-family:'JetBrains Mono',monospace;font-weight:600;">Verified Real Clinical Data</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:8px;padding:4px 0;">
            <div style="width:8px;height:8px;background:#dc2626;border-radius:50%;"></div>
            <span style="font-size:12px;color:#dc2626;font-family:'JetBrains Mono',monospace;font-weight:600;">Results Missing</span>
        </div>
        <div style="font-size:11px;color:#64748b;margin-top:4px;">{p74_err[:60]}…</div>
        """, unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div style="position:relative;margin-top:auto;padding-top:30px;">
        <div style="font-size:11px;color:#64748b;font-family:'JetBrains Mono',monospace;
                    line-height:1.6;">
            MITADT University<br>
            MIT School of Computing<br>
            <br>
            <span style="color:#dc2626;font-weight:600;">⚠️ Research System Only</span><br>
            Not for clinical diagnosis.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─── Dynamic Route to views with auto-reload ─────────────────────────────────
view_module = PAGES[page_label]

try:
    try:
        mod = importlib.import_module(f"app.views.{view_module}")
    except ModuleNotFoundError:
        mod = importlib.import_module(f"views.{view_module}")
    importlib.reload(mod)
    mod.render()
except Exception as e:
    st.error(f"⚠️ Error loading view '{page_label}': {e}")
    st.exception(e)
