"""
federated.py — Federated Learning page for FED-GBM V3 dashboard (Light Theme).
"""
import streamlit as st
from app.utils.data_loader import load_config
from app.utils.styles import section_header


def render():
    st.markdown(section_header(
        "🌐 Multi-Hospital Collaborative AI (Federated Learning)",
        "FedAvg and FedProx multi-node decentralised training architecture across 3 hospital sites"
    ), unsafe_allow_html=True)

    cfg, _ = load_config()
    rounds = cfg.get("federated_rounds", 30) if cfg else 30
    epochs = cfg.get("local_epochs", 5) if cfg else 5
    batch  = cfg.get("batch_size", 32) if cfg else 32

    # ── Architecture diagram ─────────────────────────────────────────────────
    st.markdown("### 3-Hospital Federated Co-Training Architecture")

    col_c1, col_c2, col_srv, col_c3 = st.columns([3, 3, 3.5, 3])

    # TCGA client
    with col_c1:
        st.markdown(f"""
        <div style="display:flex;flex-direction:column;align-items:center;gap:6px;padding-top:4px;">
            <div style="background:#ffffff;border:2px solid #2563eb;border-radius:8px;
                        padding:12px 14px;text-align:center;width:100%;box-shadow:0 1px 3px rgba(0,0,0,0.04);">
                <div style="font-size:12px;color:#2563eb;font-family:'JetBrains Mono',monospace;
                            letter-spacing:0.05em;text-transform:uppercase;font-weight:700;">TCGA (Node 1)</div>
                <div style="font-size:11px;color:#64748b;margin-top:4px;">N=574 patients</div>
            </div>
            <div style="color:#94a3b8;font-size:14px;">↓</div>
            <div style="background:#f8fafc;border:1px solid #cbd5e1;border-radius:6px;
                        padding:6px 10px;text-align:center;width:100%;">
                <div style="font-size:11px;color:#0f172a;font-weight:600;">Local Training</div>
                <div style="font-size:10px;color:#64748b;">{epochs} local epochs</div>
            </div>
            <div style="color:#94a3b8;font-size:14px;">↓</div>
            <div style="background:#f8fafc;border:1px solid #cbd5e1;border-radius:6px;
                        padding:6px 10px;text-align:center;width:100%;
                        font-size:10px;color:#2563eb;font-family:'JetBrains Mono',monospace;font-weight:600;">
                Weights Update
            </div>
        </div>
        """, unsafe_allow_html=True)

    # UPenn client
    with col_c2:
        st.markdown(f"""
        <div style="display:flex;flex-direction:column;align-items:center;gap:6px;padding-top:4px;">
            <div style="background:#ffffff;border:2px solid #d97706;border-radius:8px;
                        padding:12px 14px;text-align:center;width:100%;box-shadow:0 1px 3px rgba(0,0,0,0.04);">
                <div style="font-size:12px;color:#d97706;font-family:'JetBrains Mono',monospace;
                            letter-spacing:0.05em;text-transform:uppercase;font-weight:700;">UPenn (Node 2)</div>
                <div style="font-size:11px;color:#64748b;margin-top:4px;">N=574 patients</div>
            </div>
            <div style="color:#94a3b8;font-size:14px;">↓</div>
            <div style="background:#f8fafc;border:1px solid #cbd5e1;border-radius:6px;
                        padding:6px 10px;text-align:center;width:100%;">
                <div style="font-size:11px;color:#0f172a;font-weight:600;">Local Training</div>
                <div style="font-size:10px;color:#64748b;">{epochs} local epochs</div>
            </div>
            <div style="color:#94a3b8;font-size:14px;">↓</div>
            <div style="background:#f8fafc;border:1px solid #cbd5e1;border-radius:6px;
                        padding:6px 10px;text-align:center;width:100%;
                        font-size:10px;color:#d97706;font-family:'JetBrains Mono',monospace;font-weight:600;">
                Weights Update
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Central server
    with col_srv:
        st.markdown(f"""
        <div style="display:flex;flex-direction:column;align-items:center;gap:6px;padding-top:4px;">
            <div style="background:#ffffff;border:2px solid #059669;border-radius:8px;
                        padding:12px 14px;text-align:center;width:100%;box-shadow:0 1px 3px rgba(0,0,0,0.04);">
                <div style="font-size:12px;color:#059669;font-family:'JetBrains Mono',monospace;
                            letter-spacing:0.05em;text-transform:uppercase;font-weight:700;">Central Aggregator</div>
                <div style="font-size:11px;color:#64748b;margin-top:4px;">T = {rounds} rounds</div>
            </div>
            <div style="color:#94a3b8;font-size:14px;">↓</div>
            <div style="background:#f8fafc;border:1px solid #cbd5e1;border-radius:6px;
                        padding:6px 10px;text-align:center;width:100%;">
                <div style="font-size:11px;color:#0f172a;font-weight:600;">Secure Weighted Aggregation</div>
                <div style="font-size:10px;color:#64748b;font-family:'JetBrains Mono',monospace;">
                    w_k = n_k / &sum; n_j
                </div>
            </div>
            <div style="color:#94a3b8;font-size:14px;">↓</div>
            <div style="background:#ecfdf5;border:1px solid #059669;border-radius:6px;
                        padding:6px 10px;text-align:center;width:100%;
                        font-size:11px;color:#059669;font-weight:700;">
                Global Model w*
            </div>
        </div>
        """, unsafe_allow_html=True)

    # CGGA client
    with col_c3:
        st.markdown(f"""
        <div style="display:flex;flex-direction:column;align-items:center;gap:6px;padding-top:4px;">
            <div style="background:#ffffff;border:2px solid #7c3aed;border-radius:8px;
                        padding:12px 14px;text-align:center;width:100%;box-shadow:0 1px 3px rgba(0,0,0,0.04);">
                <div style="font-size:12px;color:#7c3aed;font-family:'JetBrains Mono',monospace;
                            letter-spacing:0.05em;text-transform:uppercase;font-weight:700;">CGGA (Node 3)</div>
                <div style="font-size:11px;color:#64748b;margin-top:4px;">N=249 patients</div>
            </div>
            <div style="color:#94a3b8;font-size:14px;">↓</div>
            <div style="background:#f8fafc;border:1px solid #cbd5e1;border-radius:6px;
                        padding:6px 10px;text-align:center;width:100%;">
                <div style="font-size:11px;color:#0f172a;font-weight:600;">Local Training</div>
                <div style="font-size:10px;color:#64748b;">{epochs} local epochs</div>
            </div>
            <div style="color:#94a3b8;font-size:14px;">↓</div>
            <div style="background:#f8fafc;border:1px solid #cbd5e1;border-radius:6px;
                        padding:6px 10px;text-align:center;width:100%;
                        font-size:10px;color:#7c3aed;font-family:'JetBrains Mono',monospace;font-weight:600;">
                Weights Update
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── FedAvg vs FedProx ────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Federated Algorithms Compared")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:8px;padding:18px 20px;box-shadow:0 1px 3px rgba(0,0,0,0.03);">
            <div style="font-size:15px;font-weight:700;color:#2563eb;margin-bottom:10px;">
                FedAvg (Federated Averaging)
            </div>
            <div style="color:#334155;font-size:13px;line-height:1.8;">
                <b>Aggregation:</b> Sample-size weighted averaging of hospital weights.<br>
                <b>Local Loss:</b> BCEWithLogitsLoss<br>
                <b>Optimizer:</b> Adam<br>
                <b>Rounds:</b> {rounds} | <b>Epochs:</b> {epochs} | <b>Batch:</b> {batch}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:8px;padding:18px 20px;box-shadow:0 1px 3px rgba(0,0,0,0.03);">
            <div style="font-size:15px;font-weight:700;color:#059669;margin-bottom:10px;">
                FedProx (μ = 0.01) — Regularized FL
            </div>
            <div style="color:#334155;font-size:13px;line-height:1.8;">
                <b>Aggregation:</b> Sample-size weighted average with proximal drift penalty.<br>
                <b>Local Loss:</b> BCE + (μ/2)·‖w − w_global‖²<br>
                <b>μ = 0.01:</b> Prevents single hospital weight divergence on non-IID cross-border data.
            </div>
        </div>
        """, unsafe_allow_html=True)

