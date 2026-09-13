# Comprehensive Research Code & Experiment Audit Report (FED-GBM v2)

**Project Title**: Privacy-Preserving Federated Learning for Glioblastoma Prognosis  
**Dataset**: TCGA-GBM Cohort ($N=97$ patients, 77 train / 20 test, 733 radiomic features)  
**Audit Scope**: Full Codebase, Pipelines, Scripts, Model Serialization, Results Artifacts, and Documentation  
**Audit Date**: September 2, 2026  

---

## Executive Summary of Audit Findings

A rigorous, line-by-line audit of the `FED-GBM v2` repository was conducted to evaluate methodological validity, data leakage, federated learning implementation correctness, differential privacy guarantees, cross-validation protocols, and UI/inference consistency.

### Key Critical Findings:
1. **Federated XGBoost Aggregation Failure**: `src/model_trainer.py` computes feature importance averages (`fedavg_aggregate`), adds noise, and **discards the aggregated output**. `models/global_model.pkl` is literally a standard centralized `XGBClassifier` fit directly on all training data (`X_train, y_train`). Actual tree parameters/structures are never federated.
2. **Cross-Validation Surrogate Mocking**: In `07_cross_validation.py`, the "Federated Model" evaluation is a direct duplicate of the Centralized Baseline (`cen_model`), fitting an unpartitioned `XGBClassifier` on the full training fold rather than executing federated aggregation across nodes.
3. **Feature Selection Leakage in CV**: In `07_cross_validation.py`, the 30-feature subset is loaded from `data/processed/selected_features.json` (pre-computed once in `01_preprocess_data_pipeline.py`) rather than being selected independently within each cross-validation training fold.
4. **Invalid Non-IID Simulation**: `05_iid_vs_noniid.py` sorts patients by `survival_time`, creating single-class nodes (e.g., Node 0 has 100% short survivors). These nodes skip local training, and the script evaluates the exact same global surrogate model for both IID and Non-IID splits, yielding an artificial `Delta AUC = 0.0000`.
5. **Output Perturbation vs. Formal Differential Privacy**: `06_dp_comparison.py` adds Gaussian noise to feature importances rather than model parameters or gradients. The script notes in line 161 that the model itself remains unperturbed, rendering formal $(\epsilon, \delta)$-DP claims invalid for model training.
6. **Streamlit MRI Inference Misrepresentation**: `app.py` does not perform real-time 3D PyRadiomics extraction on uploaded NIfTI scans. Instead, it extracts the patient ID from the uploaded filename (e.g., `TCGA-02-0006`) via regex and looks up pre-computed features in `fused_data.csv`.

---

## Section-by-Section Issue Audit (A through M)

### A. Data Leakage
- **Status**: `[REBUILD]`
- **Location**: `01_preprocess_data_pipeline.py` (lines 296–298), `07_cross_validation.py` (lines 355–361).
- **Details**: `01_preprocess_data_pipeline.py` performs a train/test split, but then re-concatenates `train_final` and `test_final` into `merged_dataset.csv` and `fused_data.csv`. Subsequent scripts (`05_iid_vs_noniid.py`, `07_cross_validation.py`, `app.py`) re-read `merged_dataset.csv` or `fused_data.csv`.
- **Correction Required**: Enforce a hard lock on `splits/train_ids.csv` and `splits/test_ids.csv`. Prevent any script from re-merging or re-splitting test set rows into training sets.

### B. Train/Test Contamination
- **Status**: `[REBUILD]`
- **Location**: `07_cross_validation.py` (lines 355–361).
- **Details**: `07_cross_validation.py` loads `merged_dataset.csv` (all 97 patients) and runs a 5-fold CV. However, the 30 features used were pre-selected on the 80/20 train/test split from `01_preprocess_data_pipeline.py`. Because all 97 patients were in `merged_dataset.csv`, fold test sets overlap with the data used to select the 30 features.
- **Correction Required**: Execute full 3-stage feature selection inside every individual CV training fold.

### C. Feature-Selection Leakage
- **Status**: `[REBUILD]`
- **Location**: `01_preprocess_data_pipeline.py` (lines 218–248), `07_cross_validation.py` (lines 50–54).
- **Details**: Stage 3 XGBoost gain selection is executed once globally to select 30 features. All downstream evaluation scripts (`05`, `06`, `07`, `08`) assume these 30 features as given.
- **Correction Required**: Feature selection must be parameterized as a pipeline transformer fitted strictly inside training folds. Save per-fold selection logs in `results/cv_feature_selection_log.csv`.

### D. Preprocessing Leakage
- **Status**: `[MODIFY]`
- **Location**: `01_preprocess_data_pipeline.py` (lines 184–213).
- **Details**: `01_preprocess_data_pipeline.py` correctly fits `StandardScaler`, median imputation, and `LabelEncoder` on `train_df`. However, `src/preprocess_v2.py` and `app.py` apply un-isolated transformations on merged CSVs.
- **Correction Required**: Standardize preprocessing pipelines across all entry points using scikit-learn `Pipeline` or `ColumnTransformer` objects fit solely on training data.

### E. Incorrect Cross-Validation
- **Status**: `[REBUILD]`
- **Location**: `07_cross_validation.py` (lines 207–217).
- **Details**: Line 208 explicitly notes: `# In federated setting, global model = proxy trained on all node data. For CV, we use the full training fold as the global surrogate.` Thus `Fed_AUC` in cross-validation is identical to Centralized Baseline `Cen_AUC`. True federated multi-node training is never performed during CV folds.
- **Correction Required**: Implement true multi-node federated training (FedAvg over $N$ rounds) inside every fold of the cross-validation loop.

### F. Invalid or Misleading Federated Aggregation
- **Status**: `[REBUILD]`
- **Location**: `src/model_trainer.py` (lines 93–106, 146–162).
- **Details**: `fedavg_aggregate()` averages local feature importances, adds noise, prints a log, and then **discards the output**. The saved `global_model.pkl` is trained via `oracle.fit(X_train, y_train)` as a centralized model. No decision tree structures, splits, or leaf weights are aggregated.
- **Correction Required**: Rebuild the federated architecture. Either implement true federated tree boosting (or histogram aggregation) or migrate to a PyTorch tabular neural network (`SurvivalMLP`) with parameter state-dict FedAvg/FedProx.

### G. Incorrect Differential Privacy Implementation
- **Status**: `[REBUILD]`
- **Location**: `06_dp_comparison.py` (lines 155–163), `src/model_trainer.py` (lines 62–66).
- **Details**: Gaussian noise is added to feature importance vectors. Line 161 of `06_dp_comparison.py` admits: `# Note: we record the noisy importance but the model itself is unchanged. In a full DP-FL system, the model *weights* would be perturbed.` The model parameters used for prediction contain no DP noise.
- **Correction Required**: Implement DP-SGD or DP-FedAvg where gradients or model weight updates are clipped to $L_2$ norm $C$ and perturbed with Gaussian noise $\mathcal{N}(0, \sigma^2 C^2 \mathbf{I})$ prior to aggregation.

### H. Invalid IID/Non-IID Experiment
- **Status**: `[REBUILD]`
- **Location**: `05_iid_vs_noniid.py` (lines 76–88, 172–181).
- **Details**: Sorting patients by `survival_time` assigns all short-survival cases to Node 0 and long-survival cases to Node 4. Nodes 0 and 4 become single-class, skipping local training. The script then evaluates an unpartitioned global surrogate model, reporting `Delta AUC = 0.0000` because the exact same centralized model was evaluated.
- **Correction Required**: Implement Dirichlet distribution partitioning ($\text{Dir}(\alpha)$ for $\alpha \in \{1.0, 0.5, 0.1\}$). Handle single-class client loss gracefully without skipping nodes or evaluating centralized fallbacks.

### I. Incorrect Ablation Experiments
- **Status**: `[REBUILD]`
- **Location**: `08_ablation_study.py` (lines 160–210).
- **Details**: The ablation study evaluates centralized surrogates rather than federated runs. Furthermore, "No Feature Selection (733 feats)" evaluates 30 features in some iterations due to pre-filtered dataset inputs.
- **Correction Required**: Create `results/v3/ablation_corrected.csv` evaluating true federated models across: (A) all 733 features, (B) variance filter only, (C) variance + correlation filter, (D) full 3-stage selection, and (E–H) DP noise levels $\sigma \in \{0, 0.05, 0.10, 0.20\}$.

### J. Mismatch Between Saved Models & Evaluation Scripts
- **Status**: `[MODIFY]`
- **Location**: `models/global_model.pkl` vs `global_model_weights.pth` vs `evaluate.py`.
- **Details**: `evaluate.py` loads `global_model.pkl` (XGBoost), while `02_federated_simulation_pytorch.py` saves `global_model_weights.pth` (PyTorch MLP). The repository contains two disjoint model families that are evaluated in isolation without unified benchmarking.
- **Correction Required**: Standardize serialization formats and ensure evaluation scripts benchmark identical architecture checkpoints.

### K. Mismatch Between Streamlit Inference & Training Pipeline
- **Status**: `[MODIFY]`
- **Location**: `app.py` (lines 122–146).
- **Details**: When a user uploads a NIfTI scan, `app.py` uses regex `(TCGA-\d{2}-\d{4})` to extract the patient ID and looks up pre-extracted features from `fused_data.csv`. If an unindexed NIfTI scan is uploaded, PyRadiomics feature extraction is not executed.
- **Correction Required**: Clarify in the UI that uploading indexed TCGA scans retrieves validated multi-modal profiles, or integrate a lightweight real-time PyRadiomics feature extractor fallback for novel NIfTI files.

### L. Result Files Generated Using Different Pipelines
- **Status**: `[REBUILD]`
- **Location**: `results/cross_validation_summary.txt`, `results/dp_comparison_summary.txt`.
- **Details**: `cross_validation_summary.txt` claims a "modest degradation" for the federated model, but `07_cross_validation.py` actually evaluated the centralized model as the federated model. `dp_comparison_summary.txt` claims a "0% accuracy drop" under DP because noisy importances were ignored during model fitting.
- **Correction Required**: Archive existing legacy results to `results/legacy/` and generate fresh, mathematically defensible result files under `results/v3/`.

### M. Unsupported Documentation Claims
- **Status**: `[MODIFY]`
- **Location**: `Project_Documentation.md`, `Viva_Preparation_FAQ.md`, `project_v.2_new.txt`.
- **Details**: Claims of formal DP-FL model training, HIPAA/GDPR/DPDP legal compliance, true time-to-event CoxPH survival curves in Streamlit, and real-time MRI feature extraction are not supported by the underlying Python code.
- **Correction Required**: Update all documentation artifacts to accurately reflect the actual implementation and empirical scope.

---

## Issue Classification Matrix

| Component / File | Issue Description | Audit Action |
| :--- | :--- | :---: |
| `01_preprocess_data_pipeline.py` | Global 80/20 train/test split & 3-stage feature selection | `[MODIFY]` |
| `data/processed/merged_dataset.csv` | Re-merging train and test sets into single dataset | `[REMOVE]` |
| `src/model_trainer.py` | Discarding DP importance sum; saving centralized XGBoost as `global_model.pkl` | `[REBUILD]` |
| `02_federated_simulation_pytorch.py` | PyTorch `SurvivalMLP` state-dict FedAvg simulation | `[KEEP]` |
| `05_iid_vs_noniid.py` | Sorting survival time causing single-class nodes & identical global evaluation | `[REBUILD]` |
| `06_dp_comparison.py` | Feature importance output perturbation mislabelled as formal DP model training | `[REBUILD]` |
| `07_cross_validation.py` | Using pre-selected 30 features & evaluating centralized model as federated | `[REBUILD]` |
| `08_ablation_study.py` | Evaluating centralized surrogates & mislabelling 30-feat runs as 733-feat runs | `[REBUILD]` |
| `app.py` | Streamlit patient ID lookup from uploaded filename regex | `[MODIFY]` |
| `Project_Documentation.md` | Claims of HIPAA/GDPR compliance, FedAvg XGBoost, CoxPH UI curves | `[MODIFY]` |
