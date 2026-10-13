# Reproducibility Guide: FED-GBM

This document provides exact, deterministic commands to reproduce all benchmark tables, figures, and statistical validation tests reported in the study.

---

## 1. Environment Setup

### Prerequisites
- Python >= 3.10 (tested on Python 3.10 - 3.14)
- PyTorch >= 2.0.0
- Opacus >= 1.4.0

```bash
# Clone repository
git clone https://github.com/Nisarg0403/FED_GBM.git
cd FED_GBM

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies and editable package
pip install -r requirements.txt
pip install -e .
```

---

## 2. Verification Suite

Run the full unit test suite (102 verified unit tests across models, aggregation algorithms, differential privacy mechanics, and data preprocessing):

```bash
python -m pytest tests/unit/ -v
```

Run clinical validation and leakage audit tests:

```bash
python -m pytest tests/audits/ -v
```

---

## 3. Benchmark Reproduction

### Experiment A: Centralized Baselines (Table 2)
Trains Logistic Regression, Random Forest, XGBoost, and centralized SurvivalMLP across 10 random seeds:

```bash
python experiments/baselines/run_v10_centralized.py
```

### Experiment B: Multi-Cohort Federated Benchmarks (Table 1 & Figures 2–5)
Trains federated models (FedAvg, FedProx, CA-FedAvg, HAPA) across TCGA, UPenn-GBM, and CGGA cohorts under multiple non-IID Dirichlet partition distributions:

```bash
python experiments/benchmarks/run_v10_federated.py
```

### Experiment C: Differential Privacy Budget Sweeps (Table 3 & Figure 6)
Computes the empirical Privacy-Utility Pareto frontier across noise multipliers $\sigma \in [0.5, 3.0]$ and privacy budgets $\epsilon \in [1.0, \infty)$:

```bash
python experiments/ablations/run_v10_ablations.py
```

### Experiment D: Statistical Hypothesis Testing
Executes paired two-tailed Wilcoxon signed-rank tests and Bonferroni corrections:

```bash
python scripts/run_v10_significance.py
```

---

## 4. Paper Artifact Generation

Generate all publication-ready vector figures (PDF) and LaTeX tables (.tex):

```bash
python scripts/generation/generate_paper_artifacts_v5.py
```

All generated assets will be stored under:
- `paper/figures/` (Figures 2–6, convergence, calibration, DCA, Dirichlet non-IID degradation, privacy Pareto frontier)
- `paper/tables/` (Tables 1–5 in native LaTeX `.tex` and `.csv`)
- `paper/figures/explainability/` (SHAP beeswarm and patient-level waterfall plots)

---

## 5. Interactive Clinical Decision Support Dashboard

Launch the interactive multi-view Streamlit dashboard:

```bash
streamlit run app/app.py
```
