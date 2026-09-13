# 🧠 FED-GBM V3: Privacy-Preserving Federated Learning for Glioblastoma Prognosis

Production-grade research framework and clinical decision-support application for 12-month Glioblastoma (GBM) survival outcome prediction, Kaplan-Meier survival analysis, SHAP feature explainability, multi-hospital cross-validation (TCGA, UPenn, CGGA), and formal Differential Privacy guarantees (Patient DP & Client DP).

---

## 📁 Repository Structure

```
FED_GBM_ANTIgravity_project/
├── app/                                # Streamlit Web Application & Dashboard
│   ├── app.py                          # Streamlit Main Router (`streamlit run app/app.py`)
│   ├── smoke_test.py                   # UI & Artifact Validation Script
│   ├── components/                     # Reusable UI Widgets & Cards
│   ├── utils/                          # Paths, Styles (Light Theme), Data Loaders
│   └── views/                          # 12 Interactive Page Modules
│
├── src/                                # Core Python Package (`fed_gbm`)
│   └── fed_gbm/
│       ├── models.py                   # PyTorch `SurvivalMLP` & Baseline Models
│       ├── data_loader.py              # Multi-Hospital Real Clinical Cohort Loaders
│       ├── preprocessing.py           # Leakage-Free Imputer, Scaler, Selector
│       ├── federated_client.py         # Local PyTorch Client & FedProx Regularization
│       ├── federated_server.py         # Parameter-Weighted FedAvg Aggregator
│       ├── client_dp.py                # Differential Privacy DP-FedAvg Engine
│       ├── dp_sgd.py                   # DP-SGD Clipping & Noise Injection
│       ├── privacy_accountant.py       # Renyi Differential Privacy (RDP) Accountant
│       ├── metrics.py                  # Evaluation Metrics (ROC-AUC, PR-AUC, Brier)
│       ├── explainability.py           # Tabular SHAP Explainability Engine
│       └── survival_curves.py          # Kaplan-Meier Estimators & Curves
│
├── experiments/                        # Master Research Benchmark Runners
│   ├── run_phase_7_experiments.py      # Authoritative Phase 7.4 Multi-Seed Benchmark
│   ├── run_improved_experiments.py     # Multi-Cohort (TCGA/UPenn/CGGA) Runner
│   ├── run_peer_review_experiments.py # Peer Review Ablation Suite
│   └── cgga_experiment_b.py            # CGGA Independent Validation Experiment
│
├── tests/                              # Unit & Integration Test Suite
│   ├── test_models.py
│   ├── test_preprocessing.py
│   ├── test_fedavg.py
│   ├── test_fedprox.py
│   ├── test_dp_sgd.py
│   ├── test_client_dp.py
│   └── test_phase_7_corrections.py
│
├── data/                               # Datasets & Frozen Manifests
│   ├── raw/                            # `clinical_unique.csv` (Real TCGA Cohort)
│   └── splits/                         # Frozen Train/Test/Val Manifest Files
│
├── results/                            # Official Benchmark Results
│   ├── phase7_final/                   # Authoritative Phase 7.4 Artifacts
│   ├── improved_experiments/           # Multi-Cohort Benchmark Artifacts
│   └── archive/                        # Legacy/Intermediate Experiment Runs
│
└── docs/                               # System Documentation & Reports
    ├── architecture/                   # Technical Specifications & Architecture Reports
    └── reports/                        # Dataset Audits, Quality Reports & Strategy Studies
```

---

## 🚀 Quick Start Guide

### 1. Environment Setup
```bash
python -m venv .venv
# On Windows PowerShell:
.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
pip install -e .
```

### 2. Launch the Streamlit Web Application
```bash
streamlit run app/app.py
```

### 3. Run the Unit Test Suite
```bash
python -m unittest discover tests
```

### 4. Execute Master Research Benchmarks
```bash
python experiments/run_phase_7_experiments.py
python experiments/run_improved_experiments.py
```

---

## 🔒 Privacy & Clinical Disclaimer
- **Differential Privacy Guarantees**: Tracked using Renyi Differential Privacy (RDP) accountants for Patient-Level and Client-Level DP-SGD / DP-FedAvg.
- **Clinical Notice**: **Research System Only.** Not certified for clinical diagnosis or treatment decision making.
