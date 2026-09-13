"""
styles.py — Shared CSS injection for FED-GBM V3 dashboard.
Professional Light Theme UI with high contrast and readable styling.
"""

GLOBAL_CSS = """
<style>
/* ── Import Fonts ──────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ── Hide Streamlit default header / toolbar strip ──────────────────────── */
header[data-testid="stHeader"] {
    display: none !important;
}
[data-testid="stToolbar"] {
    display: none !important;
}
#MainMenu { display: none !important; }
footer    { display: none !important; }

/* ── Global Light Theme Background & Typography ───────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, sans-serif !important;
    color: #0f172a !important;
}
.stApp {
    background: #f8fafc !important;
    color: #0f172a !important;
}
.block-container {
    padding: 1.5rem 2.5rem !important;
    max-width: 1400px !important;
}

/* ── Sidebar ─────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #e2e8f0 !important;
}
[data-testid="stSidebar"] .stRadio > label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span { color: #334155 !important; font-size: 13px !important; font-weight: 500 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #0f172a !important; }
[data-testid="stSidebar"] hr { border-color: #e2e8f0 !important; }

/* ── Headings ─────────────────────────────────────────────────────────────── */
h1 { color: #0f172a !important; font-size: 24px !important; font-weight: 700 !important; letter-spacing: -0.02em !important; margin-bottom: 6px !important; }
h2 { color: #1e293b !important; font-size: 18px !important; font-weight: 600 !important; border-bottom: 1px solid #e2e8f0 !important; padding-bottom: 8px !important; margin-top: 24px !important; }
h3 { color: #475569 !important; font-size: 14px !important; font-weight: 600 !important; text-transform: uppercase !important; letter-spacing: 0.04em !important; }

/* ── Metric cards (st.metric) ───────────────────────────────────────────── */
div[data-testid="stMetric"] {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 8px !important;
    padding: 16px 20px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05) !important;
}
div[data-testid="stMetricLabel"] { color: #64748b !important; font-size: 11px !important; font-family: 'JetBrains Mono', monospace !important; text-transform: uppercase !important; letter-spacing: 0.05em !important; font-weight: 600 !important; }
div[data-testid="stMetricValue"] { color: #2563eb !important; font-size: 28px !important; font-weight: 700 !important; }
div[data-testid="stMetricDelta"] { font-size: 11px !important; }

/* ── DataFrames / Tables ────────────────────────────────────────────────── */
div[data-testid="stDataFrame"] { border: 1px solid #e2e8f0 !important; border-radius: 8px !important; background: #ffffff !important; }
.stDataFrame th { background: #f1f5f9 !important; color: #334155 !important; font-family: 'JetBrains Mono', monospace !important; font-size: 11px !important; text-transform: uppercase !important; letter-spacing: 0.03em !important; border-bottom: 1px solid #cbd5e1 !important; font-weight: 600 !important; }
.stDataFrame td { background: #ffffff !important; color: #0f172a !important; font-family: 'JetBrains Mono', monospace !important; font-size: 12px !important; border-bottom: 1px solid #f1f5f9 !important; }

/* ── Tabs ────────────────────────────────────────────────────────────────── */
[data-testid="stTabs"] [data-baseweb="tab-list"] { background: transparent !important; border-bottom: 2px solid #e2e8f0 !important; gap: 4px !important; }
[data-testid="stTabs"] [data-baseweb="tab"] { background: transparent !important; color: #64748b !important; font-size: 14px !important; font-weight: 600 !important; border: none !important; border-bottom: 3px solid transparent !important; padding: 10px 20px !important; }
[data-testid="stTabs"] [aria-selected="true"] { color: #2563eb !important; border-bottom: 3px solid #2563eb !important; background: transparent !important; }

/* ── Buttons ─────────────────────────────────────────────────────────────── */
.stButton > button {
    background: #2563eb !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 10px 22px !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    font-family: 'Inter', sans-serif !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.08) !important;
}
.stButton > button:hover { background: #1d4ed8 !important; }

/* ── Selectbox / Radio / Inputs ──────────────────────────────────────────── */
div[data-testid="stSelectbox"] > div { background: #ffffff !important; border: 1px solid #cbd5e1 !important; border-radius: 6px !important; color: #0f172a !important; }
div[data-testid="stRadio"] label { color: #0f172a !important; font-weight: 500 !important; }

/* ── File uploader ───────────────────────────────────────────────────────── */
[data-testid="stFileUploadDropzone"] { background: #ffffff !important; border: 2px dashed #cbd5e1 !important; border-radius: 8px !important; }

/* ── Info / Success / Warning / Error boxes ─────────────────────────────── */
[data-testid="stAlert"] { border-radius: 8px !important; border-left: 4px solid !important; color: #0f172a !important; }
div[data-testid="stAlert"][data-type="info"]    { background: #eff6ff !important; border-left-color: #2563eb !important; color: #1e3a8a !important; }
div[data-testid="stAlert"][data-type="success"] { background: #ecfdf5 !important; border-left-color: #059669 !important; color: #064e3b !important; }
div[data-testid="stAlert"][data-type="warning"] { background: #fffbeb !important; border-left-color: #d97706 !important; color: #78350f !important; }
div[data-testid="stAlert"][data-type="error"]   { background: #fef2f2 !important; border-left-color: #dc2626 !important; color: #7f1d1d !important; }

/* ── Expanders ───────────────────────────────────────────────────────────── */
[data-testid="stExpander"] { background: #ffffff !important; border: 1px solid #e2e8f0 !important; border-radius: 8px !important; }
[data-testid="stExpander"] summary { color: #0f172a !important; font-size: 14px !important; font-weight: 600 !important; }

/* ── Code blocks ─────────────────────────────────────────────────────────── */
code { background: #f1f5f9 !important; color: #0f172a !important; border: 1px solid #cbd5e1 !important; border-radius: 4px !important; font-family: 'JetBrains Mono', monospace !important; font-size: 12px !important; padding: 2px 6px !important; }

/* ── Scrollbar ───────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: #f8fafc; }
::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #94a3b8; }
</style>
"""


def kpi_card(label: str, value: str, color: str = "#2563eb", suffix: str = "") -> str:
    """Returns HTML for a single KPI card in light theme."""
    return f"""
    <div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:8px;
                padding:16px 20px;text-align:center;min-width:110px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
        <div style="font-size:28px;font-weight:700;color:{color};
                    font-family:'Inter',sans-serif;line-height:1.1;">{value}{suffix}</div>
        <div style="font-size:11px;color:#64748b;font-family:'JetBrains Mono',monospace;
                    text-transform:uppercase;letter-spacing:0.05em;margin-top:6px;font-weight:600;">{label}</div>
    </div>
    """


def badge(text: str, color: str = "#059669", bg: str = "#ecfdf5") -> str:
    return f"""<span style="background:{bg};color:{color};border:1px solid {color};
        border-radius:4px;padding:3px 10px;font-size:11px;
        font-family:'JetBrains Mono',monospace;font-weight:600;
        letter-spacing:0.05em;">{text}</span>"""


def section_header(title: str, subtitle: str = "") -> str:
    sub = f'<div style="color:#475569;font-size:14px;margin-top:4px;">{subtitle}</div>' if subtitle else ""
    return f"""
    <div style="margin-bottom:20px;">
        <div style="font-size:22px;font-weight:700;color:#0f172a;
                    font-family:'Inter',sans-serif;letter-spacing:-0.01em;">{title}</div>
        {sub}
    </div>
    """


def verification_banner(checks: dict) -> str:
    entries = list(checks.items())
    half = (len(entries) + 1) // 2
    col1_html, col2_html = "", ""
    for label, passed in entries[:half]:
        icon = "✓" if passed else "✗"
        color = "#059669" if passed else "#dc2626"
        col1_html += f'<div style="display:flex;gap:8px;margin-bottom:6px;"><span style="color:{color};font-weight:700;">{icon}</span><span style="color:#334155;font-size:13px;font-weight:500;">{label}</span></div>'
    for label, passed in entries[half:]:
        icon = "✓" if passed else "✗"
        color = "#059669" if passed else "#dc2626"
        col2_html += f'<div style="display:flex;gap:8px;margin-bottom:6px;"><span style="color:{color};font-weight:700;">{icon}</span><span style="color:#334155;font-size:13px;font-weight:500;">{label}</span></div>'

    return f"""
    <div style="background:#f0fdf4;border:1px solid #bbf7d0;
                border-left:4px solid #059669;border-radius:8px;padding:20px 24px;margin:20px 0;
                box-shadow: 0 1px 3px rgba(0,0,0,0.03);">
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:14px;">
            <span style="font-size:18px;color:#059669;font-weight:700;">✓</span>
            <span style="font-size:15px;font-weight:700;color:#064e3b;
                         font-family:'JetBrains Mono',monospace;letter-spacing:0.04em;">VERIFIED EXPERIMENTAL PROTOCOL</span>
        </div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:0 40px;">
            <div>{col1_html}</div>
            <div>{col2_html}</div>
        </div>
    </div>
    """


def pipeline_step(step: str, label: str, active: bool = True, is_last: bool = False) -> str:
    color = "#2563eb" if active else "#94a3b8"
    text_color = "#0f172a" if active else "#64748b"
    arrow = "" if is_last else '<div style="color:#94a3b8;font-size:18px;text-align:center;margin:2px 0;">↓</div>'
    return f"""
    <div style="text-align:center;">
        <div style="display:inline-block;background:#ffffff;border:2px solid {color};
                    border-radius:8px;padding:8px 18px;font-size:13px;
                    font-family:'JetBrains Mono',monospace;color:{text_color};
                    font-weight:600;min-width:200px;box-shadow:0 1px 3px rgba(0,0,0,0.04);">{step}</div>
        <div style="color:#475569;font-size:12px;margin:4px 0 2px;font-weight:500;">{label}</div>
        {arrow}
    </div>"""


def disclaimer_banner(msg: str) -> str:
    return f"""
    <div style="background:#fffbeb;border:1px solid #fde68a;
                border-left:4px solid #d97706;border-radius:8px;
                padding:12px 18px;margin:16px 0;font-size:13px;
                color:#92400e;font-family:'JetBrains Mono',monospace;font-weight:500;">
        ⚠️ {msg}
    </div>"""
