"""
dataset.py — Dataset & Design page for FED-GBM V3 dashboard (Light Theme).
"""
import streamlit as st
from app.utils.styles import section_header, disclaimer_banner


def render():
    st.markdown(section_header(
        "🗄️ Multi-Hospital Patient Datasets (TCGA, UPenn, CGGA)",
        "Cohort structure, 3-hospital client manifests, and federated simulation architecture"
    ), unsafe_allow_html=True)

    # ── Architecture diagram (3 Hospital Client Cards) ────────────────────────
    st.markdown("### 3-Hospital Federated Learning Cohort Breakdown (N = 1,397 Total Patients)")
    col1, col2, col3 = st.columns(3)

    tcga_train, tcga_test = 459, 115
    upenn_train, upenn_test = 459, 115
    cgga_train, cgga_test = 199, 50

    with col1:
        st.markdown(f"""
        <div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:8px;padding:18px;box-shadow:0 1px 3px rgba(0,0,0,0.04);">
            <div style="font-size:12px;color:#2563eb;font-family:'JetBrains Mono',monospace;
                        letter-spacing:0.06em;text-transform:uppercase;margin-bottom:12px;font-weight:600;">
                Node 1: TCGA Client (US Cohort)
            </div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;">
                <div style="background:#f8fafc;border:1px solid #cbd5e1;border-radius:6px;padding:10px;text-align:center;">
                    <div style="font-size:22px;font-weight:700;color:#2563eb;">{tcga_train}</div>
                    <div style="font-size:10px;color:#64748b;font-family:'JetBrains Mono',monospace;margin-top:2px;font-weight:600;">TRAIN</div>
                </div>
                <div style="background:#f8fafc;border:1px solid #cbd5e1;border-radius:6px;padding:10px;text-align:center;">
                    <div style="font-size:22px;font-weight:700;color:#059669;">{tcga_test}</div>
                    <div style="font-size:10px;color:#64748b;font-family:'JetBrains Mono',monospace;margin-top:2px;font-weight:600;">TEST</div>
                </div>
            </div>
            <div style="margin-top:12px;color:#334155;font-size:12px;line-height:1.6;">
                <b>Total Cohort:</b> 574 Patients<br>
                <b>Modalities:</b> Clinical + MGMT/IDH Biomarkers<br>
                <b>Role:</b> Node 1 Federated Co-Training Client
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:8px;padding:18px;box-shadow:0 1px 3px rgba(0,0,0,0.04);">
            <div style="font-size:12px;color:#d97706;font-family:'JetBrains Mono',monospace;
                        letter-spacing:0.06em;text-transform:uppercase;margin-bottom:12px;font-weight:600;">
                Node 2: UPenn Client (US Cohort)
            </div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;">
                <div style="background:#f8fafc;border:1px solid #cbd5e1;border-radius:6px;padding:10px;text-align:center;">
                    <div style="font-size:22px;font-weight:700;color:#d97706;">{upenn_train}</div>
                    <div style="font-size:10px;color:#64748b;font-family:'JetBrains Mono',monospace;margin-top:2px;font-weight:600;">TRAIN</div>
                </div>
                <div style="background:#f8fafc;border:1px solid #cbd5e1;border-radius:6px;padding:10px;text-align:center;">
                    <div style="font-size:22px;font-weight:700;color:#059669;">{upenn_test}</div>
                    <div style="font-size:10px;color:#64748b;font-family:'JetBrains Mono',monospace;margin-top:2px;font-weight:600;">TEST</div>
                </div>
            </div>
            <div style="margin-top:12px;color:#334155;font-size:12px;line-height:1.6;">
                <b>Total Cohort:</b> 574 Patients<br>
                <b>Modalities:</b> Clinical + MGMT/IDH Biomarkers<br>
                <b>Role:</b> Node 2 Federated Co-Training Client
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:8px;padding:18px;box-shadow:0 1px 3px rgba(0,0,0,0.04);">
            <div style="font-size:12px;color:#7c3aed;font-family:'JetBrains Mono',monospace;
                        letter-spacing:0.06em;text-transform:uppercase;margin-bottom:12px;font-weight:600;">
                Node 3: CGGA Client (Asian Cohort)
            </div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;">
                <div style="background:#f8fafc;border:1px solid #cbd5e1;border-radius:6px;padding:10px;text-align:center;">
                    <div style="font-size:22px;font-weight:700;color:#7c3aed;">{cgga_train}</div>
                    <div style="font-size:10px;color:#64748b;font-family:'JetBrains Mono',monospace;margin-top:2px;font-weight:600;">TRAIN</div>
                </div>
                <div style="background:#f8fafc;border:1px solid #cbd5e1;border-radius:6px;padding:10px;text-align:center;">
                    <div style="font-size:22px;font-weight:700;color:#059669;">{cgga_test}</div>
                    <div style="font-size:10px;color:#64748b;font-family:'JetBrains Mono',monospace;margin-top:2px;font-weight:600;">TEST</div>
                </div>
            </div>
            <div style="margin-top:12px;color:#334155;font-size:12px;line-height:1.6;">
                <b>Total Cohort:</b> 249 Patients<br>
                <b>Modalities:</b> Clinical + MGMT/IDH Biomarkers<br>
                <b>Role:</b> Node 3 Federated Co-Training Client
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Federated simulation ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Federated Simulation Architecture")
    st.markdown("""
    <div style="display:flex;gap:12px;flex-wrap:wrap;margin-bottom:16px;">
        <div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:8px;
                    padding:16px 20px;flex:1;min-width:200px;box-shadow:0 1px 3px rgba(0,0,0,0.03);">
            <div style="font-size:11px;color:#64748b;font-family:'JetBrains Mono',monospace;
                        letter-spacing:0.05em;text-transform:uppercase;margin-bottom:6px;font-weight:600;">Federated Setup</div>
            <div style="color:#334155;font-size:13px;line-height:1.7;">
                K = 3 hospital clients<br>
                TCGA ↔ Client 1 (N=574)<br>
                UPenn ↔ Client 2 (N=574)<br>
                CGGA ↔ Client 3 (N=249)<br>
                No raw data transfer between nodes
            </div>
        </div>
        <div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:8px;
                    padding:16px 20px;flex:1;min-width:200px;box-shadow:0 1px 3px rgba(0,0,0,0.03);">
            <div style="font-size:11px;color:#64748b;font-family:'JetBrains Mono',monospace;
                        letter-spacing:0.05em;text-transform:uppercase;margin-bottom:6px;font-weight:600;">FL Rounds & Epochs</div>
            <div style="color:#334155;font-size:13px;line-height:1.7;">
                T = 30 communication rounds<br>
                E = 5 local epochs per round<br>
                Batch size = 32<br>
                FedAvg / FedProx (μ=0.01)
            </div>
        </div>
        <div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:8px;
                    padding:16px 20px;flex:1;min-width:200px;box-shadow:0 1px 3px rgba(0,0,0,0.03);">
            <div style="font-size:11px;color:#64748b;font-family:'JetBrains Mono',monospace;
                        letter-spacing:0.05em;text-transform:uppercase;margin-bottom:6px;font-weight:600;">Prediction Target</div>
            <div style="color:#334155;font-size:13px;line-height:1.7;">
                12-month survival classification<br>
                Binary: survive ≥ 365 days<br>
                Loss: BCEWithLogitsLoss (weighted)<br>
                Architecture: SurvivalMLP [128, 64]
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(disclaimer_banner(
        "TCGA, UPenn, and CGGA are real multi-center clinical patient cohorts simulating 3 hospital nodes. "
        "All federated learning rounds take place without exchanging raw patient data between hospitals."
    ), unsafe_allow_html=True)

