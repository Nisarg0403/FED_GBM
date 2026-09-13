# Research Claim Audit & Verification Report

**Project**: Privacy-Preserving Federated Learning for Glioblastoma Prognosis (`FED-GBM v2`)  
**Audit Purpose**: Trace every research claim in project documentation and paper text back to the underlying Python implementation to verify scientific support.

---

## Comprehensive Claim Verification Matrix

| # | Research Claim / Statement | Evidence in Code / Implementation | Supported by Code? | Required Action / Rewrite |
| :-: | :--- | :--- | :---: | :--- |
| **1** | **"Federated XGBoost via FedAvg weight aggregation across 5 hospital nodes"** | `src/model_trainer.py` averages feature importances, prints a log, and **discards the output**. `global_model.pkl` is trained directly on `X_train` as a centralized XGBClassifier. | **NO (Unsupported)** | **REBUILD**: Implement true federated parameter aggregation (PyTorch state-dict FedAvg/FedProx or Federated Tree Boosting). Update documentation to clarify legacy vs V3 pipeline. |
| **2** | **"Formal $(\epsilon, \delta)$-Differential Privacy protection for local hospital node models"** | `06_dp_comparison.py` adds Gaussian noise to feature importances, but explicitly states in line 161: `# ...the model itself is unchanged. In a full DP-FL system, the model *weights* would be perturbed.` | **NO (Unsupported)** | **REBUILD**: Implement true DP-SGD or DP-FedAvg with gradient/update clipping and noise addition. Remove formal DP model protection claims for legacy codebase. |
| **3** | **"Zero performance degradation under severe Non-IID data distribution skew ($\Delta\text{AUC} = 0.0000$)"** | `05_iid_vs_noniid.py` sorts by survival time, creating single-class nodes that skip local training. It then evaluates the exact same unpartitioned global surrogate model for both IID and Non-IID, generating identical numbers. | **NO (Unsupported)** | **REBUILD**: Implement Dirichlet-based partitioning ($\text{Dir}(\alpha)$ for $\alpha \in \{1.0, 0.5, 0.1\}$). Train and evaluate true federated models under each Non-IID split. |
| **4** | **"Real-time 3D MRI Radiomics Extraction upon user upload in Streamlit UI"** | `app.py` uses regex `(TCGA-\d{2}-\d{4})` to match the uploaded filename against pre-computed rows in `fused_data.csv`. PyRadiomics is not invoked on uploaded NIfTI files at runtime. | **NO (Unsupported)** | **MODIFY**: Clarify in Streamlit UI that uploading indexed TCGA MRI files retrieves pre-validated radiomic profiles from disk, or implement a lightweight fallback extractor. |
| **5** | **"Cox Proportional Hazards baseline survival function predicting personalized median survival curves"** | `app.py` and `01_preprocess_data_pipeline.py` train binary `XGBClassifier` models predicting 12-month mortality (`survival_long`). No baseline hazard $S_0(t)$ or Cox regression $\beta$ coefficients are fit. | **NO (Unsupported)** | **MODIFY**: Clearly state that the primary endpoint is 12-month binary mortality classification. If CoxPH survival curves are shown, fit true Cox regression models via `lifelines` or `scikit-survival`. |
| **6** | **"Guarantees HIPAA, GDPR, and DPDP regulatory compliance for multi-institutional healthcare deployment"** | System is a local simulation script without network encryption, key management, access controls, or verified formal differential privacy bounds on released models. | **NO (Unsupported)** | **REMOVE**: Delete claims of legal compliance (HIPAA, GDPR, DPDP). Frame privacy properties in terms of mathematical DP definitions and local data retention. |
| **7** | **"Multi-modal fusion of Clinical, 3D Radiomic, and Genomic patient data"** | TCGA-GBM $N=97$ dataset contains joint Clinical + 3D Radiomics. Genomic files are stored separately for CGGA datasets (`data/cgga693/`). | **PARTIALLY SUPPORTED** | **MODIFY**: Precise framing: "Patient-level Clinical + 3D MRI Radiomic fusion on TCGA-GBM ($N=97$), with multi-cohort clinical/genomic dataset expansion on CGGA." |
| **8** | **"Leakage-free 5-fold stratified cross-validation benchmark"** | `07_cross_validation.py` loads pre-selected 30 features (selected on 80% split) and evaluates a centralized `XGBClassifier` as the federated model. | **NO (Unsupported)** | **REBUILD**: Run 3-stage feature selection and federated model training inside every individual fold of the 5-fold CV loop. |

---

## Required Documentation Rewrites

### 1. Federated Learning Method Section Rewrite:
> *Legacy Text*: "We employ Federated Averaging (FedAvg) to aggregate local XGBoost models across 5 virtual hospital nodes."  
> *Corrected V3 Text*: "We evaluate two federated learning architectures: (i) a PyTorch multi-layer perceptron trained via Federated Averaging (FedAvg) and FedProx parameter aggregation, and (ii) a centralized baseline classifier for benchmark comparison. Local client models are trained exclusively on institution-siloed data partitions."

### 2. Privacy Section Rewrite:
> *Legacy Text*: "Our system provides formal $(\epsilon=271.38, \delta=10^{-5})$-Differential Privacy guarantees and ensures full HIPAA/GDPR compliance."  
> *Corrected V3 Text*: "To evaluate privacy-utility trade-offs, we implement differential privacy mechanisms via DP-SGD/DP-FedAvg, clipping local weight updates to $L_2$ norm $C$ and adding calibrated Gaussian noise with multiplier $\sigma \in \{0.05, 0.10, 0.20\}$. Privacy loss $(\epsilon, \delta)$ is tracked using the Renyi Differential Privacy accountant."

### 3. Non-IID Evaluation Section Rewrite:
> *Legacy Text*: "Non-IID partitioning achieved identical AUC=0.573, proving complete robustness to data heterogeneity."  
> *Corrected V3 Text*: "Data heterogeneity across participating clinical sites is simulated using Dirichlet distribution partitioning ($\text{Dir}(\alpha)$ for $\alpha \in \{1.0, 0.5, 0.1\}$), controlling class imbalance and sample size variance across virtual hospital nodes."
