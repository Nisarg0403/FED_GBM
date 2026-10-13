# Final Dataset Integration & Experimental Specification (`DATASET_INTEGRATION_SPEC_V3_FINAL.md`)

**Project Title**: Privacy-Preserving Federated Learning for Glioblastoma Prognosis  
**Target Research System**: Cross-Cohort Federated Learning for Multimodal Glioblastoma Survival Prognosis  
**Specification Version**: 3.0-Final  
**Date**: September 2, 2026  

---

## 1. Central Research Question & Terminology Standards

### Primary Research Question:
> *How do formal Differential Privacy constraints, cross-cohort data heterogeneity across independent clinical sites, and multimodal feature alignment impact prognostic utility in cross-cohort federated learning for Glioblastoma Multiforme (GBM)?*

### Enforced Terminology Standards:
- **Required**: *"Cross-cohort federated learning simulation"*, *"Multi-cohort federated learning"*, *"Independent institutional cohorts as federated clients"*, *"Simulated client nodes"*.
- **Prohibited**: *"100% realistic multi-institutional federated deployment"*, *"Live hospital-to-hospital FL deployment"*, *"HIPAA/GDPR compliant network deployment"*.

---

## 2. Decoupled Two-Experiment Research Architecture

To maintain methodological rigor, the project separates imaging radiomic predictions from non-imaging molecular validation into two decoupled experimental tracks:

```
+-----------------------------------------------------------------------------------+
|                     FED-GBM V3 REVISED EXPERIMENTAL TRACKS                        |
+------------------------------------+----------------------------------------------+
| EXPERIMENT A (PRIMARY EXPERIMENT)  | EXPERIMENT B (SECONDARY VALIDATION)          |
| Cross-Cohort Federated Model       | Clinical + Molecular External Validation     |
| (Clinical + V3 3D MRI Radiomics)   | (Clinical + Molecular Sub-Model)             |
|                                    |                                              |
| Client 1: TCGA-GBM Cohort          | Development: TCGA + UPenn Clinical/Molecular |
| Client 2: UPenn-GBM Cohort         | External Validation: CGGA-693 GBM Subset     |
| Aggregation: FedAvg / FedProx + DP | (Clinical + Molecular features ONLY)         |
+------------------------------------+----------------------------------------------+
```

---

## 3. Section 1: Federated Sufficient Statistics for Feature Normalization

To ensure global feature representations are mathematically aligned across Client 1 (TCGA) and Client 2 (UPenn) **without centralizing patient records**, we implement **Federated Sufficient Statistics Normalization**:

### Mathematical Formulation:
For feature $j$ across clients $k \in \{1, \dots, K\}$ with local patient counts $n_k$:
1. **Local Computations**: Each client $k$ computes local scalar summary statistics on training data:
   - Local Sum: $S_{k,j} = \sum_{i=1}^{n_k} x_{i,j}^{(k)}$
   - Local Sum of Squares: $SS_{k,j} = \sum_{i=1}^{n_k} \left(x_{i,j}^{(k)}\right)^2$
2. **Privacy-Preserving Central Aggregation**: Clients transmit scalar pairs $(S_{k,j}, SS_{k,j}, n_k)$ to the central server via Secure Aggregation or with $L_2$ bounded output noise:
   $$N_{\text{global}} = \sum_{k=1}^K n_k, \quad \mu_{\text{global},j} = \frac{1}{N_{\text{global}}} \sum_{k=1}^K S_{k,j}$$
   $$\sigma^2_{\text{global},j} = \frac{1}{N_{\text{global}}} \sum_{k=1}^K SS_{k,j} - \mu_{\text{global},j}^2$$
3. **Global Standard Scaling**: The central server broadcasts $(\mu_{\text{global},j}, \sigma_{\text{global},j})$ to all clients. Each client standardizes local feature matrices:
   $$\hat{x}_{i,j}^{(k)} = \frac{x_{i,j}^{(k)} - \mu_{\text{global},j}}{\sigma_{\text{global},j}}$$

> **Privacy Guarantee**: Transmitting scalar totals $S_{k,j}$ and $SS_{k,j}$ prevents raw patient vector reconstruction, satisfying differential privacy definitions under scalar output perturbation.

---

## 4. Section 2: Common Feature Schema & Feature Selection Strategy

### Strategy Choice: Option B (Development-Cohort Locked Feature Selection)
To guarantee identical model input dimensions ($p$ features in identical column positions) across Client 1 and Client 2, feature selection is performed on the **TCGA Training Set ($N=77$)**, locked, and applied to Client 2 (UPenn):

1. **Stage 1 (Variance Filter)**: $\text{Var}(x_j) > 0.01$ fit on TCGA Train.
2. **Stage 2 (Pearson Correlation Filter)**: Remove one feature from pairs with $|r| > 0.85$ fit on TCGA Train.
3. **Stage 3 (XGBoost Gain Importance Ranking)**: Rank remaining features on TCGA Train.
4. **Dynamic Feature Count Evaluation**: Rather than hardcoding $p$, V3 will benchmark models across feature counts:
   $$p \in \{10, 15, 20, 25, 30\}$$

### Unified Common Feature Schema Table:

| Column Position | Feature Identifier | Source Category | Input Type | Imputation Rule (Train-Only) |
| :-: | :--- | :--- | :---: | :--- |
| **1** | `Age` | Clinical | Continuous (Years) | Median |
| **2** | `Gender` | Clinical | Binary (1=Male, 0=Female) | Mode |
| **3** | `KPS` | Clinical | Continuous (0–100) | Median |
| **4** | `Resection_Status` | Clinical | Binary (1=GTR, 0=STR/Biopsy) | Mode |
| **5** | `MGMT_Methylation` | Molecular (Sub-model) | Binary (1=Methylated, 0=Unmethylated) | Mode |
| **6** | `IDH_Mutation` | Molecular (Sub-model) | Binary (1=Mutant, 0=Wildtype) | Mode |
| **7..p** | `V3_Radiomic_Feature_1..K` | PyRadiomics 3D | Continuous (Standardized) | Median |

---

## 5. Section 3: Identical V3 3D MRI + PyRadiomics Extraction Protocol

To eliminate batch effects and extraction protocol divergence, radiomics for **BOTH TCGA-GBM and UPenn-GBM will be re-extracted from raw/preprocessed MRI volumes using a unified V3 pipeline**:

```
Raw Multi-Sequence NIfTI Scans (T1, T1ce, T2, FLAIR) + Segmentation Mask
                                   |
                                   v
1. Spatial Resampling: 1.0 mm x 1.0 mm x 1.0 mm isotropic voxel grid
   - Image Interpolation: sitkBSpline
   - Mask Interpolation: sitkNearestNeighbor
                                   |
                                   v
2. Intensity Normalization: Z-score normalization within brain mask
   - Scale factor: 100.0, Shift: 0.0
                                   |
                                   v
3. N4 Bias Field Correction: SimpleITK N4BiasFieldCorrectionImageFilter (Max Iterations=50)
                                   |
                                   v
4. PyRadiomics Extraction Engine (PyRadiomics v3.0.1)
   - Discretization: Fixed Bin Width = 25
   - Image Types: Original, Wavelet (LLH, LHL, LHH, HLL, HLH, HHL, HHH, LLL)
   - Feature Classes: Shape3D, FirstOrder, GLCM, GLRLM, GLSZM, NGTDM, GLDM
   - IBSI Compliance: Enforced
                                   |
                                   v
733 Standardized 3D Radiomic Features per Patient
```

---

## 6. Section 4: Formal Differential Privacy Specification

Differential Privacy guarantees are explicitly parameterized under two distinct formal settings:

### A. Patient-Level Differential Privacy (DP-SGD)
- **Privacy Unit**: Individual patient record within a clinical cohort.
- **Gradient Clipping**: Local per-example gradient clipping applied to local loss gradients $g_i$:
  $$\bar{g}_i = \frac{g_i}{\max\left(1, \frac{\|g_i\|_2}{C}\right)}, \quad \text{Clipping Norm } C = 1.0$$
- **Noise Addition**: Gaussian noise injected into mini-batch updates:
  $$\tilde{g} = \frac{1}{B} \left( \sum_{i=1}^B \bar{g}_i + \mathcal{N}\left(0, \sigma^2 C^2 \mathbf{I}\right) \right)$$
- **Noise Multiplier**: $\sigma \in \{0.05, 0.10, 0.20\}$.
- **Privacy Accountant**: Renyi Differential Privacy (RDP) Accountant (Mironov 2017).
- **Target Target Parameters**: Fixed $\delta = 10^{-5}$, local steps $E=3$, rounds $T=30$, subsampling ratio $q = B/n_k$.
- **Final Epsilon ($\epsilon$)**: Calculated dynamically via RDP accountant from exact training logs.

### B. Client-Level Differential Privacy (DP-FedAvg)
- **Privacy Unit**: Participating institutional client node.
- **Clipping**: Per-client weight update vector clipping $\|\Delta \theta_k\|_2 \le C_{\text{client}}$.
- **Noise Addition**: Server adds noise to aggregated update $\sum w_k \Delta \theta_k + \mathcal{N}(0, \sigma_{\text{client}}^2 C_{\text{client}}^2 \mathbf{I})$.

---

## 7. Section 5: Corrected 12-Month Survival Endpoint Definition

### Primary Endpoint Name:
**"12-month survival classification"**

### Mathematical Outcome Rules:
For patient $i$ with overall survival time $T_i$ (days) and censoring flag $E_i \in \{0, 1\}$ ($1 = \text{Dead}$, $0 = \text{Alive}$):

$$\text{Class}(i) = \begin{cases} 
\mathbf{1} \text{ (Long Survival)}, & \text{if } T_i > 365\text{ days} \\ 
\mathbf{0} \text{ (Short Survival)}, & \text{if } T_i \le 365\text{ days} \text{ and } E_i = 1 \\ 
\text{EXCLUDED}, & \text{if } T_i \le 365\text{ days} \text{ and } E_i = 0 
\end{cases}$$

> **Censoring Inclusion Note**: Patients censored after 365 days ($T_i > 365$ and $E_i = 0$) **ARE INCLUDED IN CLASS 1** because their recorded follow-up duration empirically confirms survival past the 12-month milestone.

---

## 8. Section 6: CGGA Experiment B (Clinical + Molecular Model)

CGGA ($N \approx 249$ GBM subset) lacks compatible 3D MRI radiomics. It will be used exclusively for **Experiment B**:

```
[TCGA + UPenn Clinical + Molecular Data]
                  |
                  v
Fit Locked Clinical + Molecular Model (Age, Gender, IDH, MGMT)
                  |
                  v
[Lock Model Weights]
                  |
                  v
Out-of-Sample External Validation on CGGA-693 (N=249 GBM)
                  |
                  v
Log Out-of-Sample AUC, Accuracy, F1, C-Index
```

---

## 9. Section 7: Final Experimental Data Flow Diagram

```
+-----------------------------------------------------------------------------------+
|                        FED-GBM V3 FINAL DATA FLOW PIPELINE                        |
+-----------------------------------------------------------------------------------+

[RAW TCGA MRI (N=97)]                              [RAW UPenn MRI (N=630 Candidate)]
          |                                                       |
          v                                                       v
Standardized Preprocessing (Resample 1mm, N4, Z-Score)  Standardized Preprocessing (Resample 1mm, N4, Z-Score)
          |                                                       |
          v                                                       v
PyRadiomics V3 Extraction (IBSI, Bin=25)                PyRadiomics V3 Extraction (IBSI, Bin=25)
          |                                                       |
          v                                                       v
TCGA 3D Radiomics (733 Feats)                           UPenn 3D Radiomics (733 Feats)
          |                                                       |
          +---------------------------+---------------------------+
                                      |
                                      v
                 Federated Sufficient Statistics Normalization
                         (Global μ_j, σ_j Scaling)
                                      |
                                      v
                 TCGA Train Feature Selection (Top p in {10..30})
                                      |
                                      v
                   Common Feature Schema (p-dim Matrix)
                                      |
            +-------------------------+-------------------------+
            |                                                   |
            v                                                   v
CLIENT 1: TCGA-GBM ($N=77$ Train)                    CLIENT 2: UPenn-GBM (Post-QC $N_{\text{train}}$)
            |                                                   |
            +-------------------------+-------------------------+
                                      |
                                      v
                        PyTorch SurvivalMLP Training
                        (Loss = BCE + FedProx μ=0.01)
                                      |
                                      v
                   Formal DP (Gradient Clipping C + Noise σ)
                                      |
                                      v
                 Central FedAvg Weight Aggregation (Rounds T=30)
                                      |
                                      v
                      [LOCKED GLOBAL MODEL CHECKPOINT]
                                      |
            +-------------------------+-------------------------+
            |                                                   |
            v                                                   v
Internal Held-Out Test Evaluation                     Experiment B: Out-of-Sample Validation
 (TCGA Test N=20 + UPenn Test N=126)                   on CGGA-693 (Clinical + Molecular Sub-Model)
```

---

## 10. Execution Pause Directive

Per explicit user instructions:
- **NO datasets have been downloaded.**
- **NO datasets have been merged.**
- **NO models have been trained.**
- **NO code modifications or Streamlit edits have been executed.**
- **Execution halts completely upon creation of this specification.**
