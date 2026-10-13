# 🧠 FED-GBM: Privacy-Preserving Federated Learning for Glioblastoma Prognosis

[![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Differential Privacy](https://img.shields.io/badge/Differential%20Privacy-Opacus%20RDP-green.svg)](https://opacus.ai/)
[![Tests](https://img.shields.io/badge/Unit%20Tests-102%20Passed-brightgreen.svg)](tests/unit/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**FED-GBM** is a research-grade multimodal federated learning framework designed for multi-institutional 12-month survival prognosis in Glioblastoma Multiforme (GBM). It provides formal differential privacy guarantees (patient-level DP-SGD and client-level DP-FedAvg), clinical calibration (temperature and Platt scaling), Decision Curve Analysis (DCA), and model explainability (Federated SHAP).

---

## 📑 Table of Contents
- [Architecture Overview](#-architecture-overview)
- [Repository Structure](#-repository-structure)
- [Quick Start](#-quick-start)
- [Reproducibility](#-reproducibility)
- [Federated Algorithms](#-federated-algorithms)
- [Clinical & Privacy Audits](#-clinical--privacy-audits)
- [Citation](#-citation)

---

## 🏛 Architecture Overview

FED-GBM addresses the core challenges of cross-hospital oncology AI:
1. **Multi-Cohort Heterogeneity**: Evaluated across **TCGA-GBM**, **UPenn-GBM**, and **CGGA** cohorts under Non-IID Dirichlet distribution ($\alpha \in \{0.1, 0.5, 1.0, 5.0\}$).
2. **Privacy Guarantees**: Formal $(\epsilon, \delta)$-Differential Privacy via Rényi DP (RDP) accounting for both sample-level DP-SGD and client-level model update perturbation.
3. **Clinical Calibration & Safety**: Calibration-Aware FedAvg (`CA-FedAvg`) and Heterogeneity-Aware Proximal Aggregation (`HAPA`) preventing clinical overconfidence and label drift.
4. **Anti-Leakage Safeguards**: Strict runtime guards guaranteeing 100% isolation of validation folds and zero target information leakage ($MI < 0.50$, $|r| < 0.90$).

---

## 📁 Repository Structure

```text
FED_GBM/
├── configs/                             # Experiment Configurations (YAML)
│   ├── algorithms/                      # FedAvg, FedProx, SCAFFOLD, CA-FedAvg, HAPA
│   └── privacy/                         # DP budget parameters (ε, δ)
│
├── docs/                                # Project & Research Documentation
│   ├── clinical_audits/                 # Medical evaluation, leakage audits, patient overlap
│   ├── technical/                       # System architecture, feature schema specs
│   └── milestones/                      # Project summaries, development history & checklists
│
├── experiments/                         # Scientific Experiment Drivers
│   ├── baselines/                       # Centralized models (Cox-PH, RF, XGBoost, MLP)
│   ├── benchmarks/                      # Multi-cohort federated benchmark suites (V10 IEEE)
│   ├── ablations/                       # Component ablations, feature sensitivity, non-IID sweeps
│   └── pipelines/                       # Master reproducible pipelines
│
├── paper/                               # Publication-Ready Artifacts
│   ├── figures/                         # Vector (PDF) and 300+ DPI PNG figures
│   │   └── explainability/              # Federated SHAP beeswarm & waterfall plots
│   └── tables/                          # Camera-ready LaTeX (.tex) and summary CSV tables
│
├── results/                             # Machine-Generated Outputs
│   ├── metrics/                         # Raw CSV & JSON metrics from experiment runs
│   └── archive/                         # Legacy/historical benchmark runs
│
├── scripts/                             # Utility & Operational Scripts
│   ├── data_prep/                       # Data fetching & verification (TCGA, CGGA, cBioPortal)
│   └── generation/                      # Paper figure and LaTeX table generators
│
├── src/                                 # Installable Python Library (`fed-gbm`)
│   └── fed_gbm/
│       ├── federation/                  # Federated Client, Server & Aggregators
│       │   └── aggregators/             # FedAvg, FedProx, SCAFFOLD, CA-FedAvg, HAPA, FedNova, FedPer
│       ├── models/                      # PyTorch SurvivalMLP & Cox Proportional Hazards
│       ├── privacy/                     # DP-SGD, Client-DP & Rényi DP Accountant
│       ├── data/                        # Multimodal loaders, preprocessors & cohort adapters
│       ├── evaluation/                  # Clinical metrics, Platt scaling, DCA, Kaplan-Meier
│       ├── explainability/              # Federated SHAP & waterfall attribution
│       └── stats/                       # Wilcoxon, permutation tests & bootstrap CIs
│
├── tests/                               # Stratified Test Suite
│   ├── unit/                            # 102 Fast Unit Tests (models, DP, aggregators)
│   ├── integration/                     # Simulated multi-client federation runs
│   └── audits/                          # Scientific integrity & data leakage audits
│
├── app/                                 # Streamlit Interactive Clinical Decision App
├── Makefile                             # Build & reproduction automation targets
├── REPRODUCIBILITY.md                   # 1-Click step-by-step reproduction guide
├── CITATION.cff                         # Academic citation format
└── pyproject.toml                       # Package specification & build configuration
```

---

## ⚡ Quick Start

### 1. Installation
```bash
git clone https://github.com/Nisarg0403/FED_GBM.git
cd FED_GBM

python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

pip install -r requirements.txt
pip install -e .
```

### 2. Run Test Suite
```bash
# Run all 102 unit tests
make unit-test
# Or directly via pytest
python -m pytest tests/unit/ -v
```

### 3. Launch Clinical Decision Support Dashboard
```bash
make dashboard
# Or directly:
streamlit run app/app.py
```

---

## 🔬 Reproducibility

To reproduce all tables and figures presented in the paper:

```bash
# 1. Run centralized reference baselines (Table 2)
python experiments/baselines/run_v10_centralized.py

# 2. Run multi-cohort federated benchmark matrix (Table 1, Figures 2-5)
python experiments/benchmarks/run_v10_federated.py

# 3. Run differential privacy sensitivity sweeps (Table 3, Figure 6)
python experiments/ablations/run_v10_ablations.py

# 4. Generate all camera-ready LaTeX tables and vector figures
make artifacts
```
*See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for full replication details.*

---

## 🛡 Clinical & Privacy Audits

All clinical and technical documentation has been reviewed and structured for peer review:
- **Medical Evaluation**: [`docs/clinical_audits/MEDICAL_EVALUATION_REPORT.md`](docs/clinical_audits/MEDICAL_EVALUATION_REPORT.md)
- **Data Leakage & Mutual Information Audit**: [`docs/clinical_audits/tcga_leakage_audit.md`](docs/clinical_audits/tcga_leakage_audit.md)
- **Patient Overlap Verification**: [`docs/clinical_audits/PATIENT_OVERLAP_ANALYSIS.md`](docs/clinical_audits/PATIENT_OVERLAP_ANALYSIS.md)
- **System Architecture Specification**: [`docs/technical/ARCHITECTURE_V3.md`](docs/technical/ARCHITECTURE_V3.md)

---

## 📚 Citation

If you use FED-GBM in your research, please cite:

```bibtex
@software{chaudhari2026fedgbm,
  author = {Chaudhari, Nisarg},
  title = {{FED-GBM: Privacy-Preserving Federated Learning for Glioblastoma Multiforme Prognosis Across Multi-Institutional Cohorts}},
  year = {2026},
  url = {https://github.com/Nisarg0403/FED_GBM},
  version = {3.0.0}
}
```
