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
html, body, .stApp {
    font-family: 'Inter', -apple-system, sans-serif !important;
    background-color: #f8fafc !important;
    color: #0f172a !important;
}
.block-container {
    padding: 1.5rem 2.5rem !important;
    max-width: 1400px !important;
}

/* ── High Contrast Body & Text ────────────────────────────────────────── */
p, label, span, li, div.stMarkdown, .stMarkdown p, .stMarkdown span {
    color: #1e293b !important;
}

/* ── Sidebar ─────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background-color: #ffffff !important;
    border-right: 1px solid #e2e8f0 !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label,
[data-testid="stSidebar"] [data-testid="stRadio"] label p,
[data-testid="stSidebar"] [data-testid="stRadio"] label span,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span {
    color: #334155 !important;
    font-size: 13px !important;
    font-weight: 500 !important;
}
[data-testid="stSidebar"] [aria-checked="true"] + div p,
[data-testid="stSidebar"] [aria-checked="true"] + div span {
    color: #2563eb !important;
    font-weight: 700 !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #0f172a !important; }
[data-testid="stSidebar"] hr { border-color: #e2e8f0 !important; }

/* ── Headings ─────────────────────────────────────────────────────────────── */
h1 { color: #0f172a !important; font-size: 24px !important; font-weight: 700 !important; letter-spacing: -0.02em !important; margin-bottom: 6px !important; }
h2 { color: #1e293b !important; font-size: 18px !important; font-weight: 600 !important; border-bottom: 1px solid #e2e8f0 !important; padding-bottom: 8px !important; margin-top: 24px !important; }
h3 { color: #334155 !important; font-size: 14px !important; font-weight: 600 !important; text-transform: uppercase !important; letter-spacing: 0.04em !important; }

/* ── Metric cards (st.metric) ───────────────────────────────────────────── */
div[data-testid="stMetric"] {
    background-color: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 8px !important;
    padding: 16px 20px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05) !important;
}
div[data-testid="stMetricLabel"], div[data-testid="stMetricLabel"] * {
    color: #64748b !important;
    font-size: 11px !important;
    font-family: 'JetBrains Mono', monospace !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    font-weight: 600 !important;
}
div[data-testid="stMetricValue"], div[data-testid="stMetricValue"] * {
    color: #2563eb !important;
    font-size: 28px !important;
    font-weight: 700 !important;
}
div[data-testid="stMetricDelta"] { font-size: 11px !important; }

/* ── DataFrames / Tables ────────────────────────────────────────────────── */
div[data-testid="stDataFrame"] { border: 1px solid #e2e8f0 !important; border-radius: 8px !important; background-color: #ffffff !important; }
.stDataFrame th { background-color: #f1f5f9 !important; color: #1e293b !important; font-family: 'JetBrains Mono', monospace !important; font-size: 11px !important; text-transform: uppercase !important; letter-spacing: 0.03em !important; border-bottom: 1px solid #cbd5e1 !important; font-weight: 700 !important; }
.stDataFrame td { background-color: #ffffff !important; color: #0f172a !important; font-family: 'JetBrains Mono', monospace !important; font-size: 12px !important; border-bottom: 1px solid #f1f5f9 !important; }

/* ── Tabs ────────────────────────────────────────────────────────────────── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background-color: transparent !important;
    border-bottom: 2px solid #e2e8f0 !important;
    gap: 6px !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    background-color: #ffffff !important;
    color: #475569 !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    border: 1px solid #e2e8f0 !important;
    border-bottom: none !important;
    border-radius: 6px 6px 0 0 !important;
    padding: 10px 20px !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    color: #2563eb !important;
    border-bottom: 3px solid #2563eb !important;
    background-color: #eff6ff !important;
    font-weight: 700 !important;
}

/* ── Buttons (Explicit White Text) ────────────────────────────────────── */
.stButton > button,
.stButton > button p,
.stButton > button span,
.stDownloadButton > button,
.stDownloadButton > button p,
.stDownloadButton > button span {
    color: #ffffff !important;
    font-weight: 600 !important;
}
.stButton > button, .stDownloadButton > button {
    background-color: #2563eb !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 10px 22px !important;
    font-size: 14px !important;
    font-family: 'Inter', sans-serif !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.08) !important;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    background-color: #1d4ed8 !important;
    color: #ffffff !important;
}

/* ── Selectbox / Dropdown Popovers / Multiselect ───────────────────────── */
div[data-testid="stSelectbox"] > div,
div[data-testid="stMultiSelect"] > div,
div[data-baseweb="select"],
div[data-baseweb="select"] > div {
    background-color: #ffffff !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 6px !important;
    color: #0f172a !important;
}

div[data-baseweb="select"] div,
div[data-baseweb="select"] span,
div[data-baseweb="select"] input,
div[data-baseweb="select"] p,
div[data-baseweb="select"] div[role="combobox"] {
    background-color: transparent !important;
    color: #0f172a !important;
    font-weight: 500 !important;
}

div[data-baseweb="select"] svg {
    fill: #475569 !important;
    color: #475569 !important;
}

/* Multiselect Tags / Pills */
span[data-baseweb="tag"],
div[data-baseweb="tag"],
[data-testid="stMultiSelect"] [data-baseweb="tag"] {
    background-color: #eff6ff !important;
    border: 1px solid #bfdbfe !important;
    border-radius: 4px !important;
    color: #1d4ed8 !important;
    font-weight: 600 !important;
}
span[data-baseweb="tag"] span,
div[data-baseweb="tag"] span,
[data-testid="stMultiSelect"] [data-baseweb="tag"] * {
    color: #1d4ed8 !important;
}
span[data-baseweb="tag"] svg,
div[data-baseweb="tag"] svg {
    fill: #1d4ed8 !important;
}

/* Popover & Dropdown Options Menu */
div[data-baseweb="popover"],
div[data-baseweb="menu"],
ul[role="listbox"],
ul[data-baseweb="menu"] {
    background-color: #ffffff !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 6px !important;
    box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1), 0 4px 6px -2px rgba(0,0,0,0.05) !important;
}

li[role="option"],
div[data-baseweb="menu"] li,
div[data-baseweb="menu"] ul li {
    background-color: #ffffff !important;
    color: #0f172a !important;
    font-weight: 500 !important;
}
li[role="option"] *,
div[data-baseweb="menu"] li * {
    color: #0f172a !important;
}

li[role="option"]:hover,
li[role="option"][aria-selected="true"],
div[data-baseweb="menu"] li:hover {
    background-color: #eff6ff !important;
    color: #1d4ed8 !important;
}
li[role="option"]:hover *,
li[role="option"][aria-selected="true"] * {
    color: #1d4ed8 !important;
    font-weight: 600 !important;
}

/* ── Inputs / Sliders / Radios ────────────────────────────────────────── */
input[type="text"], input[type="number"] {
    background-color: #ffffff !important;
    color: #0f172a !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 6px !important;
}
[data-testid="stSlider"] label, [data-testid="stSlider"] label * {
    color: #0f172a !important;
    font-weight: 600 !important;
}
[data-testid="stSlider"] div[data-testid="stTickBarMin"],
[data-testid="stSlider"] div[data-testid="stTickBarMax"] {
    color: #475569 !important;
    font-weight: 600 !important;
}
[data-testid="stRadio"] label, [data-testid="stRadio"] label p, [data-testid="stRadio"] label span {
    color: #0f172a !important;
    font-weight: 500 !important;
}

/* ── File uploader ───────────────────────────────────────────────────────── */
[data-testid="stFileUploadDropzone"] { background-color: #ffffff !important; border: 2px dashed #cbd5e1 !important; border-radius: 8px !important; }

/* ── Alert boxes ──────────────────────────────────────────────────────── */
[data-testid="stAlert"] { border-radius: 8px !important; border-left: 4px solid !important; }
div[data-testid="stAlert"][data-type="info"]    { background-color: #eff6ff !important; border-left-color: #2563eb !important; }
div[data-testid="stAlert"][data-type="info"] *  { color: #1e3a8a !important; }
div[data-testid="stAlert"][data-type="success"] { background-color: #ecfdf5 !important; border-left-color: #059669 !important; }
div[data-testid="stAlert"][data-type="success"] * { color: #064e3b !important; }
div[data-testid="stAlert"][data-type="warning"] { background-color: #fffbeb !important; border-left-color: #d97706 !important; }
div[data-testid="stAlert"][data-type="warning"] * { color: #78350f !important; }
div[data-testid="stAlert"][data-type="error"]   { background-color: #fef2f2 !important; border-left-color: #dc2626 !important; }
div[data-testid="stAlert"][data-type="error"] * { color: #7f1d1d !important; }

/* ── Expanders ───────────────────────────────────────────────────────────── */
[data-testid="stExpander"] { background-color: #ffffff !important; border: 1px solid #e2e8f0 !important; border-radius: 8px !important; }
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary * { color: #0f172a !important; font-size: 14px !important; font-weight: 600 !important; }

/* ── Code blocks ─────────────────────────────────────────────────────────── */
code { background-color: #f1f5f9 !important; color: #0f172a !important; border: 1px solid #cbd5e1 !important; border-radius: 4px !important; font-family: 'JetBrains Mono', monospace !important; font-size: 12px !important; padding: 2px 6px !important; }

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


def apply_plotly_theme(fig, title: str = "", x_title: str = "", y_title: str = "", height: int = 400):
    """
    Applies high-contrast, professional Light Theme styling to any Plotly figure.
    Ensures crisp dark text (#0f172a / #1e293b) for all titles, axis labels, tick labels, hoverlabels, and legends.
    """
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>" if title else "",
            font=dict(family="Inter, sans-serif", size=15, color="#0f172a"),
            x=0.0,
            xanchor="left",
        ) if title else None,
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        height=height,
        font=dict(family="Inter, sans-serif", size=12, color="#0f172a"),
        margin=dict(l=50, r=30, t=50 if title else 30, b=110),
        xaxis=dict(
            title=dict(text=f"<b>{x_title}</b>" if x_title else "", font=dict(color="#0f172a", size=13)),
            tickfont=dict(color="#0f172a", size=12, family="Inter, sans-serif"),
            gridcolor="#e2e8f0",
            zerolinecolor="#cbd5e1",
            showline=True,
            linecolor="#cbd5e1",
        ),
        yaxis=dict(
            title=dict(text=f"<b>{y_title}</b>" if y_title else "", font=dict(color="#0f172a", size=13)),
            tickfont=dict(color="#0f172a", size=12, family="Inter, sans-serif"),
            gridcolor="#e2e8f0",
            zerolinecolor="#cbd5e1",
            showline=True,
            linecolor="#cbd5e1",
        ),
        legend=dict(
            font=dict(color="#0f172a", size=12, family="Inter, sans-serif"),
            bgcolor="#ffffff",
            bordercolor="#cbd5e1",
            borderwidth=1,
        ),
        hoverlabel=dict(
            bgcolor="#0f172a",
            font_color="#ffffff",
            font_size=13,
            font_family="Inter, sans-serif"
        )
    )
    return fig

