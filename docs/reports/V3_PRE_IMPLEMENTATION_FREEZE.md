# V3 Pre-Implementation Methodological Freeze (`V3_PRE_IMPLEMENTATION_FREEZE.md`)

**Project Title**: Privacy-Preserving Federated Learning for Glioblastoma Prognosis (`FED-GBM v3`)  
**Specification Version**: 3.0-PreImplementation-Freeze  
**Freeze Date**: September 2, 2026  

---

## 1. Section A: Final Cohort Definitions

| Client / Cohort | Assigned Role | Total Eligible ($N$) | Training Partition ($80\%$) | Locked Test Partition ($20\%$) | Linking Modalities |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Client 1: TCGA-GBM** | Primary FL Client (Development) | **$N = 97$** | $N_{\text{train},1} = 77$ | $N_{\text{test},1} = 20$ | Clinical + 3D MRI Radiomics |
| **Client 2: UPenn-GBM** | Primary FL Client (Institutional) | **$N = 574$** | $N_{\text{train},2} = 460$ | $N_{\text{test},2} = 114$ | Clinical + 3D MRI Radiomics |
| **External: CGGA-693** | Experiment B Validation Cohort | **$N = 249$** | N/A (Out-of-Sample) | $N_{\text{val}} = 249$ | Clinical + Molecular ONLY |
| **COMBINED FL TRAIN** | Federated Training System | **$N = 537$** | $N_{\text{train}} = 537$ | N/A | Multi-Center Imaging FL |
| **COMBINED HELD-OUT** | Internal Locked Test Evaluation | **$N = 134$** | N/A | $N_{\text{test}} = 134$ | Multi-Center Imaging Test |

---

## 2. Section B: Master Train / Test Identifier Manifests

To guarantee 100% reproducible data partitioning and prevent accidental test set leakage, patient IDs are locked into dedicated CSV manifests under `splits/v3/`:
- **TCGA Train Manifest**: `splits/v3/tcga_train_ids.csv` ($N = 77$).
- **TCGA Test Manifest**: `splits/v3/tcga_test_ids.csv` ($N = 20$).
- **UPenn Train Manifest**: `splits/v3/upenn_train_ids.csv` ($N = 460$).
- **UPenn Test Manifest**: `splits/v3/upenn_test_ids.csv` ($N = 114$).
- **CGGA Validation Manifest**: `splits/v3/cgga_val_ids.csv` ($N = 249$).

---

## 3. Section C & D: Decoupled Feature Schemas (Experiment A vs. Experiment B)

To maintain methodological clarity and prevent modal confusion, feature inputs for Experiment A and Experiment B are strictly decoupled:

### Experiment A (PRIMARY): Cross-Cohort Federated Clinical + 3D MRI Radiomics Model
- **Target Endpoint**: 12-Month Survival Classification (`label_12m` binary flag).
- **Clinical Features ($p_{\text{clin}} = 4$)**:
  1. `Age` (Continuous, Years, imputed via train fold median).
  2. `Gender` (Binary, $1=\text{Male}, 0=\text{Female}$).
  3. `KPS` (Karnofsky Performance Score, Continuous $0$--$100$, train fold median imputed).
  4. `Resection_Status` (Binary, $1=\text{GTR}, 0=\text{STR/Biopsy}$).
- **3D Radiomic Features ($p_{\text{rad}} \in \{6, 11, 16, 21, 26\}$)**:
  - Top $K$ PyRadiomics 3D features selected via leakage-free training-fold cross-validation ($p_{\text{total}} \in \{10, 15, 20, 25, 30\}$).
- **PROHIBITED IN EXPERIMENT A**: `MGMT_Methylation` and `IDH_Mutation` are **EXCLUDED** from Experiment A to keep the imaging radiomics pipeline clean and focused on structural MRI biomarkers.

### Experiment B (SECONDARY): Clinical + Molecular Sub-Model External Validation
- **Molecular Data Availability Gate**: Verify `Age`, `Gender`, `IDH`, and `MGMT` availability, linkage, and coding consistency across all three cohorts (TCGA, UPenn, CGGA) before starting Experiment B implementation.
- **Target Endpoint**: 12-Month Survival Classification (`label_12m` binary flag).
- **Features ($p = 4$)**:
  1. `Age` (Continuous, Years).
  2. `Gender` (Binary, $1=\text{Male}, 0=\text{Female}$).
  3. `IDH_Mutation` (Binary, $1=\text{Mutant}, 0=\text{Wildtype}$).
  4. `MGMT_Methylation` (Binary, $1=\text{Methylated}, 0=\text{Unmethylated}$).
- **External Target**: Trained on TCGA + UPenn Clinical+Molecular data $\to$ Locked $\to$ Out-of-sample evaluation on CGGA-693 ($N=249$).

---

## 4. Section E: Leak-Free Repeated 5-Fold Cross-Validation for Feature and Hyperparameter Selection

Feature selection and model hyperparameter selection on the TCGA development training set ($N=77$) must execute inside a **Leak-Free Repeated 5-Fold Cross-Validation** loop. 5-fold stratified cross-validation is repeated across the 10 experimental random seeds:

```
[TCGA TRAIN COHORT (N=77)]
            |
            v
+-----------------------------------------------------------------------+
|                    REPEATED 5-FOLD CROSS-VALIDATION                   |
|                                                                       |
|  For Fold k in {1..5}:                                                |
|   1. Split into Fold-Train (80%, N~61) and Fold-Validation (20%, N~16)|
|   2. Fit Imputation (Median) STRICTLY on Fold-Train                   |
|   3. Fit Variance Filter (Var > 0.01) STRICTLY on Fold-Train          |
|   4. Fit Pearson Correlation Filter (|r| < 0.85) STRICTLY on Fold-Train|
|   5. Fit XGBoost Gain Feature Importance Ranking on Fold-Train        |
|   6. Select top p features in {10, 15, 20, 25, 30}                    |
|   7. Apply selected feature indices to Fold-Validation                |
|   8. Evaluate PyTorch SurvivalMLP Validation Loss                     |
+-----------------------------------------------------------------------+
            |
            v
[Select Optimal Hyperparameters & Feature Count p*]
            |
            v
[Fit Final Feature Selection Pipeline on ALL N=77 TCGA Training Data]
            |
            v
[LOCK FINAL FEATURE SCHEMA]  =======> Apply Locked Schema to UPenn Train (N=460)
```

> **Strict Rule**: Preprocessing filters, imputation medians, correlation matrices, and XGBoost rankings must NEVER inspect or overlap with validation folds or locked test sets.

---

## 5. Section F: Authoritative Frozen V3 3D MRI Preprocessing Specification

All 3D MRI volumes across Client 1 (TCGA) and Client 2 (UPenn) will be processed using the exact 15-step frozen pipeline:

1. **Orientation Standardization**: Convert image headers to canonical **`RAS` (Right-Anterior-Superior)** orientation via `SimpleITK.DICOMOrient`.
2. **Registration / Alignment**: Rigid 6-DOF transformation aligned to baseline T1ce space.
3. **Resampling Grid**: Isotropic **$1.0 \times 1.0 \times 1.0\,\text{mm}^3$ voxel grid**.
4. **Resampling Interpolators**: `sitkBSpline` (3rd-order) for intensity images; `sitkNearestNeighbor` for segmentation masks.
5. **N4 Bias Field Correction**: `SimpleITK.N4BiasFieldCorrectionImageFilter` (50 max iterations) applied to raw intensity volumes **BEFORE Z-score normalization**.
6. **Brain-Mask Definition**: Otsu automated brain parenchyma mask excluding skull and background noise.
7. **Intensity Normalization**: Z-score normalization within brain mask ($\mu = 0.0, \sigma = 1.0$, scale $100.0$).
8. **Intensity Clipping**: **ENFORCED**. Z-score intensity values are clipped to the interval **$[-3.0, +3.0]$ standard deviations** prior to discretization to eliminate vascular hyperintensity outliers.
9. **Execution Order**: Resampling $\to$ N4 Bias Correction $\to$ Brain Mask Z-Score Normalization $\to$ $[-3, +3]$ Clipping $\to$ PyRadiomics Extraction.
10. **PyRadiomics Engine**: PyRadiomics v3.0.1 (IBSI compliant).
11. **Discretization**: Fixed Bin Width $= 25.0$.
12. **Image Types**: Original + Wavelet (8 sub-bands: LLH, LHL, LHH, HLL, HLH, HHL, HHH, LLL).
13. **Feature Classes**: Shape3D, FirstOrder, GLCM, GLRLM, GLSZM, NGTDM, GLDM.
14. **IBSI Compliance**: Strictly enforced.
15. **Whole Tumor Mask Construction**: Sub-region labels (ET=4, ED=2, NET=1) merged into binary Whole Tumor mask ($\text{WT} = \{1, 2, 4\}$).

---

## 6. Section G & H: Federated Normalization & Privacy Protection Rules

### Federated Sufficient Statistics:
- Each client computes local scalar totals $S_{k,j} = \sum_{i=1}^{n_k} x_{i,j}^{(k)}$ and sum of squares $SS_{k,j} = \sum_{i=1}^{n_k} (x_{i,j}^{(k)})^2$ on local training data.
- Global mean $\mu_{\text{global},j}$ and variance $\sigma^2_{\text{global},j}$ are aggregated centrally across $N_{\text{train}} = 537$ patients ($77$ TCGA train + $460$ UPenn train).

### Critical Methodological Distinction:
- **Secure Aggregation**: Protects client-level scalar summary statistics ($S_{k,j}, SS_{k,j}$) from being exposed in plaintext to the central server during communication.
- **Differential Privacy (DP)**: Requires explicit noise injection ($\mathcal{N}(0, \sigma^2 C^2 \mathbf{I})$) and formal privacy accounting (e.g. RDP Accountant). Secure aggregation alone **DOES NOT constitute Differential Privacy**.

### Strict Leakage Rules:
- All preprocessing and feature selection must occur exclusively on training folds.
- Held-out test patients (TCGA test set $N=20$, UPenn test set $N=114$) and CGGA patients ($N=249$) must **NEVER be inspected**. Test sets are standardized using previously locked training-derived statistics only.

---

## 7. Section I: Decoupled Differential Privacy Mechanisms

Patient-Level DP-SGD and Client-Level DP-FedAvg are maintained as two separate, decoupled experimental baselines:

1. **Patient-Level DP-SGD**:
   - Applies per-example local gradient clipping $\|g_i\|_2 \le C=1.0$.
   - Injects Gaussian noise $\mathcal{N}(0, \sigma^2 C^2 \mathbf{I})$ into local mini-batch updates ($\sigma \in \{0.05, 0.10, 0.20\}$).
   - Dynamically calculates final $\epsilon$ via RDP Accountant ($\delta = 10^{-5}$).
2. **Client-Level DP-FedAvg**:
   - Applies per-client weight update vector clipping $\|\Delta \theta_k\|_2 \le C_{\text{client}}$.
   - Explicitly defines client sampling and participation rate ($q$).
   - Server injects Gaussian noise into aggregated model updates with specified noise multiplier ($z$).
   - Defines the number of federated communication rounds ($T$) and aggregation weighting (e.g., sample-size weighted).
   - Provides explicit $\epsilon$ and $\delta$ ($\delta = 10^{-5}$) accounting across rounds.

---

## 8. Section J, K & L: Evaluation Protocol & Legacy Data Protection Rules

### Evaluation Metrics:
- Primary Binary Metric: Area Under the ROC Curve (ROC-AUC).
- Secondary Metrics: PR-AUC, Accuracy, F1-Score, Brier Score, Calibration.
- **Prohibited**: Concordance Index (C-Index) is strictly EXCLUDED for binary classification.
- Multi-Seed Robustness: All experiments evaluated across 10 random seeds with mean $\pm$ standard deviation reported. 5-fold stratified cross-validation is repeated across the 10 experimental random seeds.

### Versioning & Legacy Protection:
- All V3 files, logs, and CSV outputs must reside strictly under `data/v3/`, `results/v3/`, and `artifacts/v3/`.
- Legacy files (`fused_data.csv`, `Data/`, `01_` through `08_`, `app.py`) must remain 100% untouched.

---

## Final Directive: Pre-Implementation Freeze

This document represents the **final pre-implementation methodological freeze**.
- **No model training has been executed.**
- **No federated learning algorithms have been run.**
- **Code implementation for V3 awaits explicit user authorization.**
