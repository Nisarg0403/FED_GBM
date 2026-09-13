"""
Generate the FED-GBM V2 → V3 Prototype Documentation Word Document.
Run from: FED_GBM_ANTIgravity_project/FED_GBM_v3/
"""
import os, sys, json
from datetime import datetime

try:
    from docx import Document
    from docx.shared import Pt, RGBColor, Inches, Cm
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement
except ImportError:
    print("Installing python-docx...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "python-docx"])
    from docx import Document
    from docx.shared import Pt, RGBColor, Inches, Cm
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement

# ── Colour palette (matches Syntactic Dark) ──────────────────────────────────
DARK_BG    = RGBColor(0x0d, 0x11, 0x17)
ACCENT     = RGBColor(0x1f, 0x4e, 0x79)   # dark blue for headings
BLUE       = RGBColor(0x00, 0x70, 0xC0)
GREEN      = RGBColor(0x37, 0x86, 0x10)
AMBER      = RGBColor(0xBF, 0x8F, 0x00)
RED_FG     = RGBColor(0xC0, 0x00, 0x00)
LIGHT_GREY = RGBColor(0xF2, 0xF2, 0xF2)
MID_GREY   = RGBColor(0xD9, 0xD9, 0xD9)
WHITE      = RGBColor(0xFF, 0xFF, 0xFF)
BLACK      = RGBColor(0x00, 0x00, 0x00)
TBL_HDR    = RGBColor(0x1F, 0x4E, 0x79)
TBL_EVEN   = RGBColor(0xF2, 0xF7, 0xFF)

def set_cell_bg(cell, rgb: RGBColor):
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    hex_color = str(rgb)
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), hex_color)
    tcPr.append(shd)

def add_table_borders(table):
    tbl = table._tbl
    tblPr = tbl.tblPr
    tblBorders = OxmlElement('w:tblBorders')
    for side in ('top','left','bottom','right','insideH','insideV'):
        border = OxmlElement(f'w:{side}')
        border.set(qn('w:val'), 'single')
        border.set(qn('w:sz'), '4')
        border.set(qn('w:space'), '0')
        border.set(qn('w:color'), 'BFBFBF')
        tblBorders.append(border)
    tblPr.append(tblBorders)

def heading(doc, text, level=1):
    p = doc.add_heading(text, level=level)
    run = p.runs[0]
    run.font.color.rgb = ACCENT
    run.font.bold = True
    if level == 1:
        run.font.size = Pt(16)
        p.paragraph_format.space_before = Pt(16)
        p.paragraph_format.space_after  = Pt(6)
    elif level == 2:
        run.font.color.rgb = BLUE
        run.font.size = Pt(13)
        p.paragraph_format.space_before = Pt(12)
        p.paragraph_format.space_after  = Pt(4)
    else:
        run.font.color.rgb = RGBColor(0x40, 0x40, 0x40)
        run.font.size = Pt(11)
    return p

def para(doc, text, bold=False, italic=False, size=Pt(10.5), color=None, indent=False):
    p = doc.add_paragraph()
    if indent:
        p.paragraph_format.left_indent = Inches(0.3)
    run = p.add_run(text)
    run.bold  = bold
    run.italic = italic
    run.font.size = size
    if color:
        run.font.color.rgb = color
    p.paragraph_format.space_after = Pt(3)
    return p

def bullet(doc, text, level=0):
    p = doc.add_paragraph(style='List Bullet')
    run = p.add_run(text)
    run.font.size = Pt(10.5)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.left_indent = Inches(0.3 + 0.2 * level)
    return p

def comparison_table(doc, headers, rows, col_widths=None):
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = 'Table Grid'
    add_table_borders(table)

    # Header row
    hdr = table.rows[0]
    for i, h in enumerate(headers):
        cell = hdr.cells[i]
        set_cell_bg(cell, TBL_HDR)
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        p = cell.paragraphs[0]
        run = p.add_run(h)
        run.font.bold  = True
        run.font.color.rgb = WHITE
        run.font.size  = Pt(10)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # Data rows
    for ri, row_data in enumerate(rows):
        row = table.rows[ri + 1]
        bg = TBL_EVEN if ri % 2 == 0 else WHITE
        for ci, cell_text in enumerate(row_data):
            cell = row.cells[ci]
            set_cell_bg(cell, bg)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            p = cell.paragraphs[0]
            # Colour-code V2 vs V3 in comparison tables
            text = str(cell_text)
            run = p.add_run(text)
            run.font.size = Pt(10)
            if text.startswith("✅") or text.startswith("✓"):
                run.font.color.rgb = GREEN
                run.font.bold = True
            elif text.startswith("❌") or text.startswith("✗"):
                run.font.color.rgb = RED_FG
            elif text.startswith("⚠"):
                run.font.color.rgb = AMBER
            if ci == 0:
                run.font.bold = True

    # Column widths
    if col_widths:
        for i, w in enumerate(col_widths):
            for row in table.rows:
                row.cells[i].width = Inches(w)

    doc.add_paragraph()  # spacing
    return table

def divider(doc):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after  = Pt(4)
    p._p.get_or_add_pPr()
    pBdr = OxmlElement('w:pBdr')
    bottom = OxmlElement('w:bottom')
    bottom.set(qn('w:val'), 'single')
    bottom.set(qn('w:sz'), '4')
    bottom.set(qn('w:space'), '1')
    bottom.set(qn('w:color'), 'BFBFBF')
    pBdr.append(bottom)
    p._p.pPr.append(pBdr)

# ────────────────────────────────────────────────────────────────────────────
# DOCUMENT
# ────────────────────────────────────────────────────────────────────────────
doc = Document()

# Page margins
for section in doc.sections:
    section.top_margin    = Cm(2.0)
    section.bottom_margin = Cm(2.0)
    section.left_margin   = Cm(2.5)
    section.right_margin  = Cm(2.5)

# ── COVER ──────────────────────────────────────────────────────────────────
title_p = doc.add_paragraph()
title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = title_p.add_run("FED-GBM V3 — Research Prototype Documentation")
run.font.size  = Pt(20)
run.font.bold  = True
run.font.color.rgb = ACCENT

sub_p = doc.add_paragraph()
sub_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
sub_p.add_run("Privacy-Preserving Federated Learning for\nMulti-Modal Glioblastoma Survival Prediction").font.size = Pt(13)

date_p = doc.add_paragraph()
date_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
dr = date_p.add_run(f"Generated: {datetime.now().strftime('%B %d, %Y')}   |   MITADT University — MIT School of Computing")
dr.font.size = Pt(9)
dr.font.color.rgb = RGBColor(0x60, 0x60, 0x60)

doc.add_paragraph()
divider(doc)
doc.add_paragraph()

# ════════════════════════════════════════════════════════════════════════════
# 1. PROJECT OVERVIEW
# ════════════════════════════════════════════════════════════════════════════
heading(doc, "1. Project Overview", 1)
para(doc, (
    "This document describes the design, implementation, and improvements of the FED-GBM V3 "
    "Research Dashboard — a Streamlit-based academic research demonstration platform for a "
    "privacy-preserving federated learning study on glioblastoma (GBM) 12-month survival "
    "prediction. It supersedes and completely replaces the legacy FED-GBM V2 prototype."
))
doc.add_paragraph()

heading(doc, "Study Identity", 2)
meta = [
    ("Project Title",     "Privacy-Preserving Federated Learning for Multi-Modal Glioblastoma Survival Prediction"),
    ("Version",           "V3 (Final Verified Benchmark)"),
    ("Task",              "12-Month Binary Survival Classification"),
    ("Cohorts",           "TCGA (Development, N=97) + UPenn (External Validation, N=574)"),
    ("Training Split",    "TCGA Train N=77 | UPenn Train N=460 | Combined = 537"),
    ("Test Sets",         "TCGA Locked Test N=20 | UPenn Locked Test N=114"),
    ("Feature Schema",    "15 features: 4 Clinical + 11 Radiomic (XGBoost-selected from TCGA train only)"),
    ("Models Evaluated",  "12 models across 4 paradigms × 5 random seeds × 2 evaluation sets = 120 runs"),
    ("Privacy Mechanisms","Patient-Level DP-SGD (Opacus) + Client-Level DP-FedAvg (Gaussian RDP)"),
    ("Experiment Status", "Phase 7.4 — COMPLETED & VERIFIED (0 NaN, 0 Inf, all integrity checks passed)"),
    ("Dashboard Status",  "FED-GBM V3 Streamlit Dashboard — LIVE (localhost:8501)"),
]
comparison_table(doc, ["Field", "Details"], meta, col_widths=[1.8, 4.7])

# ════════════════════════════════════════════════════════════════════════════
# 2. V2 → V3 COMPARISON
# ════════════════════════════════════════════════════════════════════════════
heading(doc, "2. V2 vs V3 — Core Comparison", 1)
para(doc, "The following tables document every dimension of change between the old V2 prototype and the new V3 dashboard.")
doc.add_paragraph()

heading(doc, "2.1 Application Architecture", 2)
arch_rows = [
    ("Code structure",       "Single monolithic file (app.py, 362 lines)",   "✅ Modular: app.py + 10 view modules + utils/"),
    ("App entry point",      "FED_GBM_v2/app.py",                            "✅ FED_GBM_v3/app.py"),
    ("Page routing",         "st.radio in-file conditionals",                 "✅ Dedicated views/ package, each page is a render() module"),
    ("Styling",              "Inline ad-hoc CSS fragments scattered in code", "✅ Centralised utils/styles.py (Stitch 'Syntactic Dark' design system)"),
    ("Path management",      "os.path.join scattered throughout",             "✅ utils/paths.py — single source of truth for all paths"),
    ("Data loading",         "No validation, no caching",                    "✅ utils/data_loader.py — @st.cache_data + NaN/row count validation"),
    ("Number of pages",      "3 (Prognosis Tool, Training Center, Explainability)", "✅ 10 (Overview + 9 research sections)"),
    ("Design system",        "None — ad-hoc GitHub dark attempt",            "✅ Stitch 'Syntactic Dark' — Geist + JetBrains Mono, 4px grid, tonal layers"),
]
comparison_table(doc, ["Dimension", "V2 (Legacy)", "V3 (Current)"], arch_rows, col_widths=[1.6, 2.5, 2.4])

heading(doc, "2.2 Scientific Methodology", 2)
sci_rows = [
    ("Prediction model",      "CoxPH (Cox Proportional Hazards) — survival regression", "✅ SurvivalMLP [32→16→1] — binary 12-month classification"),
    ("Prediction output",     "Hazard ratio + 36-month survival curve",               "✅ Probability of survival >12 months (sigmoid of logit)"),
    ("Federated strategy",    "Old FedAvg simulation",                                "✅ FedAvg + FedProx (μ=0.01) — both implemented & benchmarked"),
    ("Baselines",             "CoxPH only",                                           "✅ 4 centralized models: LR, RF, XGBoost, MLP"),
    ("Privacy mechanisms",    "None",                                                  "✅ Patient-level DP-SGD (Opacus) + Client-level DP-FedAvg"),
    ("DP noise levels",       "N/A",                                                  "✅ σ ∈ {0.05, 0.10, 0.20} for both Patient-Level and Client-Level DP"),
    ("Feature selection",     "Ad-hoc / unclear",                                    "✅ XGBoost importance on TCGA train only — anti-leakage verified"),
    ("Feature count",         "Unknown / variable",                                  "✅ 15 locked features: 4 clinical + 11 radiomic"),
    ("Seeds",                 "Not specified",                                        "✅ 5 seeds: {42, 123, 2024, 3407, 9999}"),
    ("Test set locking",      "Not locked — could vary per run",                     "✅ Locked split manifests (SHA-256 verified, identical across all seeds)"),
    ("Reporting",             "No aggregation",                                      "✅ Mean ± SD across 5 seeds — 8 metrics per model per cohort"),
]
comparison_table(doc, ["Dimension", "V2 (Legacy)", "V3 (Current)"], sci_rows, col_widths=[1.6, 2.5, 2.4])

heading(doc, "2.3 Dashboard Pages", 2)
pages_rows = [
    ("Page 1",  "Interactive Prognosis Tool (CoxPH, survival curve, hazard ratio)", "✅ Overview — KPI cards, VERIFIED banner, study abstract"),
    ("Page 2",  "Federated Training Center (re-trains model in-browser)",           "✅ MRI Prediction Demo — NIfTI upload + V3 pipeline diagram (honest: no fake predictions)"),
    ("Page 3",  "Model Explainability (CoxPH coefficients bar chart)",              "✅ Dataset & Design — cohort cards, class distributions, federated sim architecture"),
    ("Page 4",  "❌ Did not exist",                                                 "✅ Feature Pipeline — 12-step preprocessing diagram, anti-leakage protocol"),
    ("Page 5",  "❌ Did not exist",                                                 "✅ Federated Learning — FedAvg/FedProx architecture diagram + algorithm cards"),
    ("Page 6",  "❌ Did not exist",                                                 "✅ Model Performance — bar charts with value labels, 8 metrics, 12 models, 2 cohorts"),
    ("Page 7",  "❌ Did not exist",                                                 "✅ Privacy Analysis — DP tables, ε-utility trade-off scatter"),
    ("Page 8",  "❌ Did not exist",                                                 "✅ Cross-Cohort Generalization — paired bars, gap analysis"),
    ("Page 9",  "❌ Did not exist",                                                 "✅ Reproducibility — full config grid, 11-item audit checklist, raw JSON manifests"),
    ("Page 10", "❌ Did not exist",                                                 "✅ Limitations — 4-category academic limitations, global disclaimer"),
]
comparison_table(doc, ["Page", "V2 (Legacy)", "V3 (Current)"], pages_rows, col_widths=[0.6, 2.5, 3.4])

heading(doc, "2.4 Data Sources & Integrity", 2)
data_rows = [
    ("Results source",        "global_model.json (CoxPH coefficients only)",   "✅ results/v3/phase7_final/ — all 7 authoritative artifacts"),
    ("Hardcoded metrics",     "Yes — manually entered values",                  "✅ NONE — all values read dynamically from Phase 7.4 CSVs/JSONs"),
    ("Metric validation",     "No validation performed",                        "✅ Row count, NaN check, model list validated at load time"),
    ("Caching",               "No caching",                                     "✅ @st.cache_data(ttl=3600) on all loaders"),
    ("Data fallback",         "Falls back to hardcoded defaults",               "✅ Returns None + error string — never shows fabricated data"),
    ("Aggregate CSV",         "Not generated",                                  "✅ 24 rows × 27 cols — 12 models × 2 eval sets × 8 metrics"),
    ("Per-seed CSV",          "Not generated",                                  "✅ 120 rows — 12 × 5 × 2 — 0 NaN verified"),
    ("Privacy CSV",           "Not generated",                                  "✅ 9 rows — patient-level (6) + client-level (3) DP accounting"),
    ("Split verification",    "Not generated",                                  "✅ split_verification.json with SHA-256 hashes and cross-seed identity"),
    ("Feature provenance",    "Not generated",                                  "✅ feature_provenance.json documenting anti-leakage status"),
    ("Aggregation verify",    "Not generated",                                  "✅ aggregation_verification.json with mismatch counts"),
]
comparison_table(doc, ["Dimension", "V2 (Legacy)", "V3 (Current)"], data_rows, col_widths=[1.6, 2.5, 2.4])

heading(doc, "2.5 UI/UX & Visual Design", 2)
ui_rows = [
    ("Theme",                  "Dark mode attempt, inconsistent",               "✅ Unified 'Syntactic Dark' — #0d1117 bg, #161b22 cards, #58a6ff accent"),
    ("Typography",             "Browser default",                               "✅ Inter (headings/body) + JetBrains Mono (labels/code/numbers)"),
    ("Streamlit header strip", "Visible white strip across all pages",          "✅ Hidden via CSS (display:none on stHeader, stToolbar, #MainMenu, footer)"),
    ("KPI metrics",            "st.metric default styling",                     "✅ Custom HTML cards — large blue numbers, monospace uppercase labels"),
    ("Charts",                 "Matplotlib static figures",                     "✅ Plotly interactive — hover, zoom, value labels on bars"),
    ("Bar chart labels",       "❌ No labels — values only on hover",           "✅ Numeric value labels on top of every bar (textposition='outside')"),
    ("Navigation",             "3-item radio in sidebar",                       "✅ 10-item radio, live Phase 7.4 status dot, footer disclaimer"),
    ("Verification banner",    "❌ Did not exist",                              "✅ PHASE 7.4 VERIFIED banner with 8-item 2-column checklist"),
    ("Responsive layout",      "Single column",                                 "✅ st.columns() throughout — cohort cards, metric grids, algorithm pairs"),
    ("Disclaimer",             "❌ None",                                       "✅ Global amber warning banner on every applicable page"),
    ("Code blocks",            "❌ Raw code leaking from HTML comments",        "✅ All HTML uses st.columns() — no HTML comments in st.markdown()"),
]
comparison_table(doc, ["Dimension", "V2 (Legacy)", "V3 (Current)"], ui_rows, col_widths=[1.6, 2.5, 2.4])

# ════════════════════════════════════════════════════════════════════════════
# 3. EXPERIMENTAL RESULTS SUMMARY
# ════════════════════════════════════════════════════════════════════════════
heading(doc, "3. Phase 7.4 Experimental Results Summary", 1)
para(doc, (
    "All results below are from the authoritative Phase 7.4 benchmark "
    "(results/v3/phase7_final/phase7_aggregate_results.csv). "
    "Values are mean ± SD across 5 random seeds."
))
doc.add_paragraph()

heading(doc, "3.1 TCGA Internal Test (N=20, Locked)", 2)
tcga_rows = [
    ("Centralized XGBoost",        "0.7500 ± 0.0000", "0.9343 ± 0.0000", "0.8000 ± 0.0000"),
    ("Patient DP (σ=0.20)",         "0.7125 ± 0.0686", "0.9185 ± 0.0265", "0.8000 ± 0.0000"),
    ("Centralized Random Forest",  "0.6969 ± 0.0611", "0.9167 ± 0.0143", "0.8000 ± 0.0000"),
    ("Client DP (σ=0.20)",          "0.6781 ± 0.0738", "0.9040 ± 0.0267", "0.6300 ± 0.1891"),
    ("Centralized Logistic Reg.",  "0.6562 ± 0.0000", "0.8891 ± 0.0000", "0.8000 ± 0.0000"),
    ("Patient DP (σ=0.10)",         "0.5438 ± 0.1525", "0.8374 ± 0.0672", "0.7900 ± 0.0224"),
    ("FedProx MLP (μ=0.01)",       "0.5094 ± 0.1069", "0.8149 ± 0.0489", "0.8200 ± 0.0447"),
    ("Centralized MLP",            "0.5938 ± 0.0585", "0.8693 ± 0.0279", "0.7700 ± 0.0447"),
    ("Patient DP (σ=0.05)",         "0.6219 ± 0.1247", "0.8698 ± 0.0638", "0.7800 ± 0.0274"),
    ("Client DP (σ=0.10)",          "0.6219 ± 0.1463", "0.8868 ± 0.0552", "0.7900 ± 0.0224"),
    ("Client DP (σ=0.05)",          "0.6000 ± 0.0611", "0.8767 ± 0.0237", "0.8000 ± 0.0000"),
    ("FedAvg MLP",                 "0.4344 ± 0.1009", "0.7883 ± 0.0463", "0.7800 ± 0.0570"),
]
comparison_table(doc, ["Model", "ROC-AUC", "PR-AUC", "Accuracy"],
                 tcga_rows, col_widths=[2.3, 1.4, 1.4, 1.4])

heading(doc, "3.2 UPenn External Test (N=114, Locked)", 2)
upenn_rows = [
    ("Client DP (σ=0.20)",          "0.5848 ± 0.1114", "0.9239 ± 0.0291", "0.6368 ± 0.2996"),
    ("Client DP (σ=0.10)",          "0.4920 ± 0.0940", "0.8986 ± 0.0224", "0.8912 ± 0.0048"),
    ("Centralized Logistic Reg.",  "0.4730 ± 0.0000", "0.8975 ± 0.0000", "0.8860 ± 0.0000"),
    ("Patient DP (σ=0.05)",         "0.4864 ± 0.0494", "0.8989 ± 0.0176", "0.8351 ± 0.0358"),
    ("Patient DP (σ=0.20)",         "0.4658 ± 0.0674", "0.8976 ± 0.0184", "0.8930 ± 0.0073"),
    ("Patient DP (σ=0.10)",         "0.4667 ± 0.0702", "0.8988 ± 0.0190", "0.8649 ± 0.0100"),
    ("Client DP (σ=0.05)",          "0.4533 ± 0.0337", "0.8969 ± 0.0079", "0.8772 ± 0.0124"),
    ("FedAvg MLP",                 "0.4221 ± 0.0730", "0.8835 ± 0.0261", "0.8561 ± 0.0171"),
    ("FedProx MLP (μ=0.01)",       "0.3840 ± 0.0816", "0.8746 ± 0.0290", "0.8456 ± 0.0182"),
    ("Centralized XGBoost",        "0.4044 ± 0.0000", "0.8846 ± 0.0000", "0.8860 ± 0.0000"),
    ("Centralized Random Forest",  "0.3608 ± 0.0140", "0.8734 ± 0.0069", "0.8947 ± 0.0000"),
    ("Centralized MLP",            "0.3748 ± 0.0488", "0.8626 ± 0.0159", "0.8263 ± 0.0418"),
]
comparison_table(doc, ["Model", "ROC-AUC", "PR-AUC", "Accuracy"],
                 upenn_rows, col_widths=[2.3, 1.4, 1.4, 1.4])

heading(doc, "3.3 Privacy Accounting Summary", 2)
priv_rows = [
    ("Patient-Level DP", "TCGA", "σ=0.05", "q=0.2078", "T=750",  "ε = 152,149.24"),
    ("Patient-Level DP", "TCGA", "σ=0.10", "q=0.2078", "T=750",  "ε = 28,531.65"),
    ("Patient-Level DP", "TCGA", "σ=0.20", "q=0.2078", "T=750",  "ε = 3,040.33"),
    ("Patient-Level DP", "UPenn","σ=0.05", "q=0.0348", "T=4350", "ε = 796,400.96"),
    ("Patient-Level DP", "UPenn","σ=0.10", "q=0.0348", "T=4350", "ε = 85,036.44"),
    ("Patient-Level DP", "UPenn","σ=0.20", "q=0.0348", "T=4350", "ε = 2,674.03"),
    ("Client-Level DP",  "Both", "σ=0.05", "ΔS=1.7132","T=30",   "ε = 6,525.65"),
    ("Client-Level DP",  "Both", "σ=0.10", "ΔS=1.7132","T=30",   "ε = 1,762.83"),
    ("Client-Level DP",  "Both", "σ=0.20", "ΔS=1.7132","T=30",   "ε = 506.41"),
]
comparison_table(doc, ["Level", "Client", "σ", "Sampling/ΔS", "Steps", "ε (δ=1e-5)"],
                 priv_rows, col_widths=[1.4, 0.7, 0.7, 1.1, 0.7, 1.9])
para(doc, "⚠ Note: All epsilon values are weak privacy bounds. Smaller σ → larger ε → weaker formal privacy.", italic=True, color=AMBER)

# ════════════════════════════════════════════════════════════════════════════
# 4. WHAT WAS IMPLEMENTED IN V3
# ════════════════════════════════════════════════════════════════════════════
heading(doc, "4. What Was Implemented in V3", 1)

heading(doc, "4.1 Phase-by-Phase Implementation", 2)
phases = [
    ("Phase 1–2", "V3 data pipeline and cohort integration (TCGA N=97 + UPenn N=574). Feature extraction with PyRadiomics 3.0.1 from 3D MRI NIfTI scans. Clinical feature alignment (Age, Gender, KPS, Resection_Status)."),
    ("Phase 3",   "V3 preprocessing protocol. Orientation → Affine alignment → 1mm isotropic resampling → N4 bias correction → Otsu masking → Z-score normalisation → [-3,+3]σ clip. Locked train/test split manifests."),
    ("Phase 4",   "XGBoost feature selection on TCGA training data only (N=77). 733 radiomic features → 11 selected + 4 protected clinical = 15-feature locked schema. Anti-leakage verified by feature_provenance.json."),
    ("Phase 5",   "Patient-level DP-SGD implementation using Opacus v1.6.0. Poisson-compatible subsampling. RDP accounting. Formal ε calculation for σ ∈ {0.05, 0.10, 0.20} per client."),
    ("Phase 5.1", "Multi-round privacy accounting audit. Verified that Opacus receives correct sample_rate = 16/N, correct steps count, and δ=1e-5."),
    ("Phase 6",   "Client-Level DP-FedAvg implementation. Gaussian noise on aggregated model updates. Adjacency A (replacement sensitivity). ΔS = 1.7132216 (formally derived)."),
    ("Phase 6.1–6.2", "Formal audit of client-level DP sensitivity derivation and adjacency declaration. Verified consistency between docs, code, and reported epsilon values."),
    ("Phase 7",   "Initial multi-seed benchmark execution. Identified protocol violation: TCGA test set was re-generated per seed instead of using locked manifest."),
    ("Phase 7.1", "Forensic audit. Identified: (1) floating TCGA test set, (2) feature count consistency check."),
    ("Phase 7.2", "Corrective implementation. Created splits/v3/tcga_train_ids.csv and tcga_test_ids.csv. Implemented locked manifest loading. Added SHA-256 cross-seed identity verification."),
    ("Phase 7.3", "First authorised corrected benchmark run (incomplete — inconsistencies found in provenance chain)."),
    ("Phase 7.4", "FINAL AUTHORISED BENCHMARK. 120/120 runs, 0 NaN, all integrity checks passed. All artifacts written to results/v3/phase7_final/. Automated forensic audit generated."),
    ("Dashboard", "FED-GBM V3 Streamlit dashboard built from scratch with 10 modular pages, Stitch 'Syntactic Dark' design system, Plotly interactive charts, and live artifact loading with validation."),
]
for phase, desc in phases:
    p = doc.add_paragraph()
    p.paragraph_format.left_indent  = Inches(0.25)
    p.paragraph_format.space_after  = Pt(4)
    run_bold = p.add_run(f"{phase}: ")
    run_bold.bold = True
    run_bold.font.color.rgb = BLUE
    run_bold.font.size = Pt(10.5)
    run_desc = p.add_run(desc)
    run_desc.font.size = Pt(10.5)

doc.add_paragraph()

heading(doc, "4.2 Models Benchmarked", 2)
models_rows = [
    ("Centralized Logistic Regression", "Centralized",  "liblinear solver, C=1.0",               "Baseline linear model"),
    ("Centralized Random Forest",       "Centralized",  "100 trees, max_depth=4",                "Non-linear ensemble baseline"),
    ("Centralized XGBoost",             "Centralized",  "100 estimators, lr=0.05, depth=3",      "Gradient boosted baseline"),
    ("Centralized MLP",                 "Centralized",  "SurvivalMLP [32,16], Adam, BCE",        "Neural baseline (same arch as FL)"),
    ("FedAvg MLP",                      "Federated",   "T=30, E=5, batch=16, sample-wt. avg.",  "Core federated model"),
    ("FedProx MLP",                     "Federated",   "Same as FedAvg + proximal μ=0.01",      "Drift-regularised federated model"),
    ("Patient DP (σ=0.05)",             "Patient-Level DP","Opacus DP-SGD, C=1.0",              "Strong patient privacy (weak ε)"),
    ("Patient DP (σ=0.10)",             "Patient-Level DP","Opacus DP-SGD, C=1.0",              "Moderate patient privacy"),
    ("Patient DP (σ=0.20)",             "Patient-Level DP","Opacus DP-SGD, C=1.0",              "Less patient noise (better ε)"),
    ("Client DP (σ=0.05)",              "Client-Level DP","Gaussian RDP, ΔS=1.7132",            "Strong client privacy"),
    ("Client DP (σ=0.10)",              "Client-Level DP","Gaussian RDP, ΔS=1.7132",            "Moderate client privacy"),
    ("Client DP (σ=0.20)",              "Client-Level DP","Gaussian RDP, ΔS=1.7132",            "Less client noise"),
]
comparison_table(doc, ["Model Name", "Paradigm", "Key Hyperparameters", "Role"],
                 models_rows, col_widths=[1.8, 1.3, 2.1, 1.3])

# ════════════════════════════════════════════════════════════════════════════
# 5. BUGS FIXED FROM V2
# ════════════════════════════════════════════════════════════════════════════
heading(doc, "5. Bugs & Issues Fixed from V2 → V3", 1)
bugs_rows = [
    ("White header strip",      "Streamlit default header visible on all pages",
     "✅ CSS: display:none on stHeader, stToolbar, #MainMenu, footer"),
    ("Floating test set",       "TCGA test set was re-generated per random seed (Phase 7 violation)",
     "✅ Locked manifests loaded from splits/v3/tcga_test_ids.csv — SHA-256 verified"),
    ("Feature leakage risk",    "Feature selection could access test or external data",
     "✅ XGBoost selector fitted on TCGA train only (N=77) — feature_provenance.json confirms"),
    ("CoxPH methodology",       "V2 used CoxPH survival regression — not suited for this binary task",
     "✅ Replaced with SurvivalMLP binary classification + BCEWithLogitsLoss"),
    ("Hardcoded model weights", "V2 loaded global_model.json with manually exported CoxPH coefficients",
     "✅ All results from dynamic Phase 7.4 CSV/JSON artifacts — no hardcoded values"),
    ("Re-training in browser",  "V2 had 'Start Federated Loop' button that attempted to run training via subprocess",
     "✅ Training completed and locked — dashboard is read-only display of Phase 7.4 artifacts"),
    ("HTML comments in markdown","HTML comments (<!-- -->) inside st.markdown() were rendered as visible text",
     "✅ Architecture diagrams rewritten with st.columns() — no HTML comments in any markup"),
    ("No bar chart labels",     "Metric values only visible on hover — not readable at a glance",
     "✅ textposition='outside' labels on every bar in performance + generalization pages"),
    ("No privacy accounting",   "V2 had no DP implementation or epsilon reporting",
     "✅ Opacus RDP accounting (patient-level) + Closed-form Gaussian RDP (client-level)"),
    ("No external validation",  "V2 only used TCGA with no external test cohort",
     "✅ UPenn (N=574) used as independent external validation cohort"),
    ("No NaN handling",         "No validation of metrics — NaN could silently propagate",
     "✅ data_loader.py validates NaN count = 0 before returning any DataFrame"),
    ("No reproducibility info", "V2 had no seed control or reproducibility documentation",
     "✅ 5 fixed seeds, split verification, config JSON, aggregation verification all generated"),
]
comparison_table(doc, ["Issue", "V2 Problem", "V3 Fix"],
                 bugs_rows, col_widths=[1.5, 2.5, 2.5])

# ════════════════════════════════════════════════════════════════════════════
# 6. FILE STRUCTURE
# ════════════════════════════════════════════════════════════════════════════
heading(doc, "6. V3 File Structure", 1)

heading(doc, "6.1 Dashboard Application", 2)
structure_rows = [
    ("FED_GBM_v3/app.py",                   "Main entry point — page config, sidebar, router"),
    ("FED_GBM_v3/utils/paths.py",           "Centralised path resolution to all Phase 7.4 artifacts"),
    ("FED_GBM_v3/utils/data_loader.py",     "Cached, validated loaders for CSV/JSON artifacts"),
    ("FED_GBM_v3/utils/styles.py",          "Global CSS + HTML helper functions (kpi_card, badge, verification_banner, etc.)"),
    ("FED_GBM_v3/views/overview.py",        "Overview page — KPI cards, VERIFIED banner, study abstract, top-3 tables"),
    ("FED_GBM_v3/views/prediction_demo.py", "MRI Prediction Demo — NIfTI upload, 3-plane viewer, V3 pipeline diagram"),
    ("FED_GBM_v3/views/dataset.py",         "Dataset & Design — cohort cards, class distributions, federated sim params"),
    ("FED_GBM_v3/views/features.py",        "Feature Pipeline — preprocessing steps, feature schema, anti-leakage"),
    ("FED_GBM_v3/views/federated.py",       "Federated Learning — architecture diagram, FedAvg/FedProx algorithm cards"),
    ("FED_GBM_v3/views/performance.py",     "Model Performance — Plotly bar chart + labels, ranking table, scatter"),
    ("FED_GBM_v3/views/privacy_page.py",    "Privacy Analysis — DP tables, ε-utility trade-off log-scale chart"),
    ("FED_GBM_v3/views/generalization.py",  "Cross-Cohort Generalization — paired bars + labels, gap analysis"),
    ("FED_GBM_v3/views/reproducibility.py", "Reproducibility — config grid, 11-item audit checklist, raw JSON tabs"),
    ("FED_GBM_v3/views/limitations.py",     "Limitations — 4-category academic limitations, global research disclaimer"),
]
comparison_table(doc, ["File", "Purpose"], structure_rows, col_widths=[3.0, 3.5])

heading(doc, "6.2 Phase 7.4 Artifacts (Read-Only)", 2)
artifacts_rows = [
    ("results/v3/phase7_final/phase7_aggregate_results.csv",   "24 rows — mean±SD across 5 seeds for all 12 models × 2 cohorts × 8 metrics"),
    ("results/v3/phase7_final/phase7_per_seed_results.csv",    "120 rows — individual per-seed results (0 NaN verified)"),
    ("results/v3/phase7_final/phase7_privacy_results.csv",     "9 rows — formal (ε,δ)-DP accounting for all σ levels and both DP types"),
    ("results/v3/phase7_final/phase7_config.json",             "Frozen experimental configuration — seeds, models, features, DP params"),
    ("results/v3/phase7_final/split_verification.json",         "SHA-256 hashes and cross-seed identity confirmation for locked test sets"),
    ("results/v3/phase7_final/feature_provenance.json",        "Anti-leakage proof — confirms feature selector never accessed test/external data"),
    ("results/v3/phase7_final/aggregation_verification.json",  "Spot-check of aggregate statistics — 0 mismatches reported"),
    ("results/v3/phase7_final/PHASE_7.4_FINAL_AUDIT.md",       "Automated forensic audit report — PASS verdict"),
]
comparison_table(doc, ["File", "Contents"], artifacts_rows, col_widths=[3.4, 3.1])

# ════════════════════════════════════════════════════════════════════════════
# 7. LIMITATIONS
# ════════════════════════════════════════════════════════════════════════════
heading(doc, "7. Limitations & Research Disclaimers", 1)

lim_rows = [
    ("Data",         "Small TCGA locked test set (N=20) limits statistical reliability of internal metrics"),
    ("Data",         "TCGA and UPenn are used as simulated federated clients — not a real hospital deployment"),
    ("Data",         "Cross-cohort distribution shift (scanner, patient selection, labels) is not controlled for"),
    ("Privacy",      "All reported ε values are weak bounds — not strong anonymity guarantees"),
    ("Privacy",      "Client-level DP bounds one client's influence, does not imply client identity anonymity"),
    ("Privacy",      "Higher ROC-AUC at larger σ is likely a small-test-set artefact, not a real effect"),
    ("Model",        "SurvivalMLP is a compact tabular model — not a large-scale deep learning system"),
    ("Model",        "Radiomic feature IDs (rad_feat_*) are dataset-specific and may not generalise"),
    ("Model",        "Only 12-month binary survival modelled — no time-to-event or multi-timepoint analysis"),
    ("Clinical",     "NOT validated for clinical use — no FDA/CE approval — not for patient decisions"),
    ("Clinical",     "MRI preprocessing was not independently validated by radiologists"),
    ("Dashboard",    "No saved model weights — inference artifact not generated by Phase 7.4 pipeline"),
]
comparison_table(doc, ["Category", "Limitation"], lim_rows, col_widths=[1.2, 5.3])

# ════════════════════════════════════════════════════════════════════════════
# FOOTER / SIGNATURES
# ════════════════════════════════════════════════════════════════════════════
doc.add_page_break()
divider(doc)
footer_p = doc.add_paragraph()
footer_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
fr = footer_p.add_run(
    "FED-GBM V3 Research Prototype Documentation\n"
    "MITADT University — MIT School of Computing\n"
    f"Document generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    "Results source: results/v3/phase7_final/  |  Dashboard: FED_GBM_v3/app.py\n\n"
    "⚠ Research demonstration only — not for clinical diagnosis, treatment, or medical decision-making.\n"
    "Federated learning is simulated using independent public cohorts. No claim of real-world hospital deployment is made."
)
fr.font.size = Pt(9)
fr.font.color.rgb = RGBColor(0x60, 0x60, 0x60)

# ── Save ─────────────────────────────────────────────────────────────────────
out_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "FED_GBM_V3_Prototype_Documentation.docx"
)
doc.save(out_path)
print(f"Document saved: {os.path.abspath(out_path)}")
