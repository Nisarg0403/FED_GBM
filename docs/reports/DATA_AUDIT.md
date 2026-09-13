# Dataset Audit Report (TCGA-GBM Cohort)

**Project**: Privacy-Preserving Federated Learning for Glioblastoma Prognosis (`FED-GBM v2`)  
**Audit Date**: September 2, 2026  
**Primary Dataset**: TCGA-GBM (The Cancer Genome Atlas - Glioblastoma Multiforme)  

---

## 1. Patient Cohort Overview & Verification

- **Total Patient Count**: $N = 97$ unique patients.
- **Duplicate Patients**: 0 duplicate patient submitter IDs detected (`cases.submitter_id` / `case_id`).
- **Patient ID Format**: `TCGA-XX-XXXX` (e.g., `TCGA-02-0011`, `TCGA-06-5413`).
- **Train / Test Allocation**:
  - **Training Cohort**: $N = 77$ patients ($\approx 80\%$)
  - **Held-Out Test Cohort**: $N = 20$ patients ($\approx 20\%$)

---

## 2. Target Endpoint Analysis

The dataset contains two parallel survival representations:

### A. Binary Endpoint: 12-Month Mortality Classification (`label_12m` / `survival_long`)
- **Definition**: $\text{Label} = 1$ if overall survival $\text{survival\_time} > 365\text{ days}$; $\text{Label} = 0$ if patient died within 365 days ($\le 365\text{ days}$).
- **Class Distribution**:
  - **Long Survival ($>12$ months)**: $53\text{ patients } (54.6\%)$
  - **Short Survival ($\le 12$ months)**: $44\text{ patients } (45.4\%)$
- **Balance Ratio**: $1.20 : 1.00$ (Moderately balanced; well-suited for stratified cross-validation).

### B. Time-to-Event Endpoint (Continuous Survival Analysis)
- **`survival_time`**: Continuous integer duration in days (Range: $22\text{ days}$ to $1731\text{ days}$, Median: $376.0\text{ days}$, Mean: $421.4\text{ days}$).
- **`event` / Vital Status**: Binary censoring flag ($\text{Dead} = 1$, $\text{Alive} = 0$).
  - **Observed Events (Deaths)**: $81\text{ patients } (83.5\%)$
  - **Censored (Alive at last follow-up)**: $16\text{ patients } (16.5\%)$

---

## 3. Modality Availability Audit per Patient

| Modality Layer | Data Source / Format | Availability per Patient ($N=97$) | Notes |
| :--- | :--- | :---: | :--- |
| **Clinical Attributes** | Demographics, Vital Status, Days to Death/Follow-up | **97 / 97 (100%)** | Age, Gender, Vital Status, Follow-up days |
| **3D MRI Radiomics** | BraTS TCGA-GBM NIfTI Scans (T1, T1ce, T2, FLAIR) | **97 / 97 (100%)** | 733 Shape, Intensity & Texture features extracted |
| **Genomic Markers** | TCGA / CGGA Genomic Files | **Separate Cohorts** | Genomic profiles exist in CGGA-693 / TCGA GDC separate files |

> **Critical Audit Note on Multimodality**: For the core $N=97$ TCGA-GBM dataset, every patient has **joint Clinical + 3D Radiomics** profiles. Genomic features are present in separate CGGA cohorts (`data/cgga693/`). Calling the TCGA-GBM 97-patient model "patient-level clinical + radiomic fusion" is scientifically valid, but claiming simultaneous 3-way patient-level genomic fusion on the 97 TCGA cohort requires explicitly noting cohort boundaries.

---

## 4. Feature Space & Distribution Statistics

- **Raw Feature Count**: 733 PyRadiomics features + 4 Clinical variables = **737 candidate variables**.
- **Selected Feature Subset**: Top **30 features** after 3-stage consensus filtering:
  - *Stage 1 (Variance Filter)*: Removes variables with $\text{Var} < 0.01$ (filters low-signal background radiomics).
  - *Stage 2 (Pearson Correlation Filter)*: Drops collinear pairs with $|r| > 0.90$ (eliminates redundant GLCM/GLRLM texture matrix variants).
  - *Stage 3 (XGBoost Gain Importance)*: Selects top 30 features by feature contribution.

### Sample-to-Feature Ratio Audit:
- **Training Cohort**: $N = 77\text{ samples}, p = 30\text{ features}$.
- **Ratio**: $77 / 30 = 2.57 : 1$.
- **Recommendation**: To maintain statistical power and prevent overfitting on small cohorts ($N=77$), the feature count $p$ should ideally be restricted to $15$--$20$ features ($\ge 4 : 1$ ratio) or evaluated via penalized regression (LASSO / ElasticNet).

---

## 5. Missing Value & Outlier Audit

- **Clinical Variables**: Age has $< 2\%$ missing values (imputed using training set median). Gender has $0\%$ missing values.
- **Radiomic Features**: 0 missing values across the 733 PyRadiomics columns for the 97 NIfTI scans.
- **Outlier Handling**: High-skew texture matrices (e.g., `TEXTURE_NGTDM_ED_T1Gd_Busyness`, `HISTO_NET_T1Gd_Bin8`) exhibit kurtosis $> 5.0$. Standard scaling (`StandardScaler`) fit solely on training data handles range scaling; robust scaling (`RobustScaler`) is recommended for V3.
