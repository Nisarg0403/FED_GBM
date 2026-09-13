# V3 Experimental & Research Execution Plan

**Project**: Privacy-Preserving Federated Learning for Glioblastoma Prognosis (`FED-GBM v3`)  
**Objective**: Build a publication-grade, mathematically sound, leak-free federated experimental pipeline for 12-month mortality prognosis and survival analysis in Glioblastoma Multiforme.

---

## 1. Locked Held-Out Test Set Protocol

To guarantee zero data contamination across all hyperparameter tuning, feature selection, and federated architecture iterations, the dataset will be split once into a locked 80/20 partition:

- **Total Cohort**: $N = 97$ TCGA-GBM patients.
- **Training Cohort ($80\%$)**: $N = 77$ patients.
- **Held-Out Test Cohort ($20\%$)**: $N = 20$ patients.
- **Random Seed**: Fixed `random_state = 42` with stratification on `survival_long` (12-month mortality label).
- **Storage Location**:
  - `splits/train_ids.csv` (contains the exact list of 77 patient IDs).
  - `splits/test_ids.csv` (contains the exact list of 20 patient IDs).
- **Enforcement Rule**: The 20 held-out test patients in `test_ids.csv` must **NEVER** be accessed during feature selection, median imputation fitting, scaling, cross-validation fold splitting, or hyperparameter optimization. They are accessed exclusively for final model benchmarking.

---

## 2. Leak-Free 5-Fold Cross-Validation Framework

Cross-validation will be executed strictly on the 77 training patients (`train_ids.csv`) using a 5-fold Stratified K-Fold setup ($N \approx 61$ train / $N \approx 16$ validation per fold):

```
                               TRAINING COHORT (N=77)
                                         |
                       +-----------------+-----------------+
                       | 5-Fold Stratified Cross-Validation|
                       +-----------------+-----------------+
                                         |
         +-------------------------------+-------------------------------+
         |                                                               |
         v                                                               v
  [TRAINING FOLD (N~61)]                                        [VALIDATION FOLD (N~16)]
         |                                                               |
  1. Fit Median Imputer                                                  |
  2. Fit StandardScaler                                                  |
  3. Fit Stage 1: Variance Filter (var > 0.01)                            |
  4. Fit Stage 2: Correlation Filter (|r| < 0.85)                        |
  5. Fit Stage 3: Feature Ranking (Top 15-20)                            |
         |                                                               |
  Transformed Train X -----------------------------------------> Transform Val X
         |                                                               |
  6. Train Local Node Models (5 Nodes)                                   |
  7. Execute FedAvg / FedProx Aggregation (30 Rounds)                     |
         |                                                               |
  Trained Global Model ---------------------------------------> Evaluate Val X
                                                                         |
                                                                  Log Fold Metrics
                                                                (AUC, Acc, F1, C-Index)
```

- **Feature Selection Log**: The selected feature names and feature counts for every individual fold will be exported to `results/cv_feature_selection_log.csv`.

---

## 3. Dirichlet-Based Non-IID Partitioning Protocol

To replace the legacy survival-sort partition, cross-institutional data heterogeneity across the 5 virtual hospital nodes will be simulated using a **Dirichlet distribution $\text{Dir}(\alpha)$**:

- **Dirichlet Concentration Parameters**:
  - $\alpha = 1.0$: Mild Non-IID heterogeneity (realistic clinical variation).
  - $\alpha = 0.5$: Moderate Non-IID heterogeneity (strong hospital specialization).
  - $\alpha = 0.1$: Extreme Non-IID heterogeneity (severe label skew).
- **Execution & Safety Rules**:
  - Node partitions are generated dynamically for each $\alpha$.
  - Single-class nodes (nodes with only Class 0 or Class 1) must be handled gracefully: local loss computation utilizes class-weighted Binary Cross-Entropy or laplacian smoothing rather than skipping node updates or aborting.
  - **Output Artifacts**:
    - `results/noniid_dirichlet.csv` (recording node sample sizes, local class ratios, local node performance, and global model performance per $\alpha$).
    - `results/noniid_client_distributions.png` (stacked bar chart / heatmap showing exact patient label composition per node).

---

## 4. Multi-Seed Statistical Reliability & Confidence Intervals

Due to the small cohort size ($N=97$), single-run results are susceptible to variance. All final experiments will be repeated across **10 independent random seeds** (`seed` $\in \{42, 101, 202, 303, 404, 505, 606, 707, 808, 909\}$):

- **Reported Metrics**:
  - ROC-AUC
  - Classification Accuracy
  - Precision & Recall
  - $F_1$-Score
  - Concordance Index (C-Index)
- **Statistical Summaries**: For every metric, report:
  $$\text{Mean} \pm \text{Standard Deviation} \quad [\text{95\% Confidence Interval}]$$

---

## 5. Model Baseline & Federated Benchmark Suite

### A. Centralized & Single-Node Baseline Models
- **Logistic Regression (L2 / ElasticNet)**: Linear baseline for low-sample tabular data.
- **Random Forest Classifier**: Ensemble decision tree baseline.
- **XGBoost Classifier**: Gradient boosted tree baseline.
- **PyTorch MLP (`SurvivalMLP`)**: Multi-layer perceptron neural network baseline.

### B. Federated Strategy Benchmarks
- **Centralized Baseline**: Trained on unpartitioned pooled data.
- **Local-Only Baseline**: Average performance of isolated client models without aggregation.
- **Federated IID (FedAvg)**: Standard parameter aggregation under uniform node distribution.
- **Federated Non-IID (FedAvg / FedProx)**: Parameter aggregation under Dirichlet skewed distribution ($\alpha = 0.5$).
- **Federated + DP (DP-FedAvg / DP-SGD)**: Federated aggregation with gradient clipping norm $C=1.0$ and noise multipliers $\sigma \in \{0.05, 0.10, 0.20\}$.

---

## 6. Corrected Canonical Ablation Suite

The ablation study will evaluate 8 locked configurations under identical cross-validation and testing protocols:

1. **Ablation 1 (Raw Features)**: All 733 radiomic features (no feature selection).
2. **Ablation 2 (Stage 1 Only)**: Variance Thresholding only ($\text{Var} > 0.01$).
3. **Ablation 3 (Stage 1 + Stage 2)**: Variance Thresholding + Pearson Correlation Filtering ($|r| < 0.85$).
4. **Ablation 4 (Full 3-Stage Selection)**: Variance + Correlation + Feature Importance Ranking (Top 15–20 features).
5. **Ablation 5 (No DP)**: Federated FedAvg without noise ($\sigma = 0.0$).
6. **Ablation 6 (Low DP)**: Federated DP-FedAvg with $\sigma = 0.05$.
7. **Ablation 7 (Medium DP)**: Federated DP-FedAvg with $\sigma = 0.10$.
8. **Ablation 8 (High DP)**: Federated DP-FedAvg with $\sigma = 0.20$.

- **Output File**: `results/v3/ablation_corrected.csv`.

---

## 7. Artifact Organization & Legacy Preservation

- **Legacy Results Preservation**: All previous execution results, plots, and summary text files will be preserved intact under `results/legacy/`.
- **V3 Results Storage**: All newly generated leak-free V3 results, CSV logs, figures, and model weights will be stored under `results/v3/` and `models/v3/`.
- **Result Metadata Schema**: Every generated V3 CSV file must contain standard metadata columns:
  `dataset`, `seed`, `split_type`, `model_architecture`, `federated_strategy`, `n_clients`, `dirichlet_alpha`, `dp_sigma`, `dp_epsilon`, `feature_count`, `timestamp`.
