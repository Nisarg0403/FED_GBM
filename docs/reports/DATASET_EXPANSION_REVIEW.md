# Public Glioblastoma (GBM) Dataset Expansion Review

**Project Title**: Privacy-Preserving Federated Learning for Glioblastoma Prognosis  
**Scope**: Systematic evaluation of publicly available Glioblastoma Multiforme (GBM) cohorts for project expansion.  
**Review Date**: September 2, 2026  

---

## Executive Overview of Candidate Datasets

To transition from the current $N=97$ development cohort to a high-impact, publication-grade study, 6 publicly available Glioblastoma datasets were rigorously evaluated across 15 technical criteria (A through O):

```
+-----------------------------------------------------------------------------------+
|                        CANDIDATE PUBLIC GBM DATASETS                              |
+------------------------------------+----------------------------------------------+
| 1. UPenn-GBM (TCIA)               | N = 630 GBM (Clinical + 3D MRI + MGMT/IDH)  |
| 2. BraTS 2021 Adult Glioma         | N = 1,251 Scans (N ~ 571 with Survival)      |
| 3. CGGA-693 / CGGA-325             | N = 1,018 Glioma (N ~ 387 GBM, Genomics)     |
| 4. CPTAC-3 GBM                     | N ~ 100 GBM (Clinical + MRI + Proteomics)    |
| 5. Ivy GAP                         | N = 41 GBM (Clinical + MRI + RNA-seq)        |
| 6. TCGA-GBM (Full TCIA Release)    | N = 262 Patients (N = 97 in current repo)    |
+------------------------------------+----------------------------------------------+
```

---

## Detailed Evaluation of Candidate Cohorts

### 1. UPenn-GBM Cohort (The Cancer Imaging Archive - TCIA) $\star$ TOP RECOMMENDATION $\star$
*Published by Penn Medicine / University of Pennsylvania (Bakas et al. 2022, Scientific Data)*

- **A. Patient Count**: $N = 630$ total patients.
- **B. GBM-Specific Size**: **630 Glioblastoma patients** ($100\%$ histologically confirmed de novo GBM).
- **C. Survival Time**: Overall survival duration in days for all patients ($0$ to $1,800+$ days).
- **D. Censoring Status**: Complete event/censoring indicator ($1 = \text{Dead}$, $0 = \text{Censored}$).
- **E. Clinical Data**: Age, Gender, Karnofsky Performance Score (KPS), Surgical Resection Status (GTR/STR).
- **F. MRI Availability**: **100% 3D NIfTI MRI scans** available for all 630 patients.
- **G. MRI Sequences**: Multi-parametric 3T MRI: T1-weighted, T1-contrast enhanced (T1ce/T1Gd), T2-weighted, and T2-FLAIR.
- **H. Radiomic Extraction Feasibility**: **100% Feasible**. Standardized skull-stripped, co-registered NIfTI volumes with pre-computed sub-region tumor segmentations (Edema, Enhancing Tumor, Non-enhancing Core).
- **I. Genomic/Molecular Data**: **MGMT promoter methylation status**, **IDH1/IDH2 mutation status**, and EGFR amplification status.
- **J. Patient-Level Linkage**: **100% Verified Patient-Level Linkage** (Every patient row connects Clinical + 3D MRI + Survival + MGMT/IDH).
- **K. Overlap with TCGA**: **0% Overlap** (Independent single-institution cohort from Penn Medicine).
- **L. Endpoint Compatibility**: Direct $1:1$ match with TCGA-GBM 12-month mortality binary classification and Cox continuous survival time.
- **M. Preprocessing Compatibility**: Fully compatible with PyRadiomics and standard StandardScaler pipelines.
- **N. Licensing / Access**: Open-access under Creative Commons CC-BY 4.0 on TCIA (No credentialing barriers).
- **O. Paper Feasibility**: **Highest**. Recognized as the gold-standard single-institution GBM imaging dataset.

---

### 2. BraTS 2021 Benchmark Dataset (MICCAI / RSNA)
*MICCAI Brain Tumor Segmentation & Survival Prediction Challenge*

- **A. Patient Count**: $N = 1,251$ total training cases.
- **B. GBM-Specific Size**: Includes $N \approx 571$ patients with clinical survival labels.
- **C. Survival Time**: Survival duration in days for the training subset ($N \approx 571$).
- **D. Censoring Status**: Event status included for survival cases.
- **E. Clinical Data**: Age and Resection Status (GTR/STR). Limited demographics.
- **F. MRI Availability**: **100% 3D NIfTI MRI scans** for all cases.
- **G. MRI Sequences**: Co-registered T1, T1ce, T2, and FLAIR scans ($1\,\text{mm}^3$ isotropic).
- **H. Radiomic Extraction Feasibility**: **100% Feasible** (Includes expert manual segmentations).
- **I. Genomic/Molecular Data**: Minimal (MGMT status available for subset in BraTS-MGMT extension).
- **J. Patient-Level Linkage**: Clinical + MRI linked for $N \approx 571$ patients.
- **K. Overlap with TCGA**: **High Overlap** (The $N=97$ TCGA-GBM cohort is a strict sub-component of BraTS 2021).
- **L. Endpoint Compatibility**: High match for survival prediction.
- **M. Preprocessing Compatibility**: Identical image spacing and preprocessing.
- **N. Licensing / Access**: Open academic benchmark license.
- **O. Paper Feasibility**: Excellent for imaging, but requires filtering out TCGA cases to prevent overlap.

---

### 3. CGGA-693 & CGGA-325 Cohorts (Chinese Glioma Genome Atlas)
*Published by Beijing Tiantan Hospital / Capital Medical University*

- **A. Patient Count**: $N = 1,018$ total glioma patients ($N=693$ in CGGA-693, $N=325$ in CGGA-325).
- **B. GBM-Specific Size**: **$N \approx 249$ GBM patients** in CGGA-693; **$N \approx 138$ GBM patients** in CGGA-325.
- **C. Survival Time**: Overall survival duration in days for all patients.
- **D. Censoring Status**: Event/vital status available ($1 = \text{Dead}$, $0 = \text{Alive}$).
- **E. Clinical Data**: Age, Gender, Primary/Recurrent status, Chemotherapy/Radiotherapy history.
- **F. MRI Availability**: **NO 3D MRI SCANS** available in public CGGA repositories.
- **G. MRI Sequences**: N/A.
- **H. Radiomic Extraction Feasibility**: Impossible (No imaging files).
- **I. Genomic/Molecular Data**: **Extensive**. IDH mutation, MGMT methylation, 1p19q codeletion, and whole-transcriptome mRNA-seq gene expression.
- **J. Patient-Level Linkage**: Clinical + Genomic + Survival linked for CGGA patients. **Unlinked to TCGA MRI**.
- **K. Overlap with TCGA**: **0% Overlap** (Chinese population cohort).
- **L. Endpoint Compatibility**: Compatible survival time and 12-month mortality labels.
- **M. Preprocessing Compatibility**: Compatible for tabular clinical/genomic feature selection.
- **N. Licensing / Access**: Open access via CGGA data portal (`http://www.cgga.org.cn/`).
- **O. Paper Feasibility**: Ideal as a **separate federated client** for clinical/molecular survival modeling or external genomic validation.

---

### 4. CPTAC-3 GBM Cohort (Clinical Proteomic Tumor Analysis Consortium)
*NCI CPTAC Program*

- **A. Patient Count**: $N \approx 100$ patients.
- **B. GBM-Specific Size**: $N \approx 100$ Glioblastoma cases.
- **C. Survival Time**: Overall survival available via GDC portal.
- **D. Censoring Status**: Event status available.
- **E. Clinical Data**: Rich clinical and treatment metadata.
- **F. MRI Availability**: 3D MRI scans available on TCIA for a subset ($N \approx 60$).
- **G. MRI Sequences**: T1, T1ce, T2, FLAIR.
- **H. Radiomic Extraction Feasibility**: Requires custom segmentation pipeline.
- **I. Genomic/Molecular Data**: **Multi-omics**: Mass-spectrometry proteomics, phosphoproteomics, and RNA-seq.
- **J. Patient-Level Linkage**: Clinical + Proteomics + Genomics linked.
- **K. Overlap with TCGA**: **0% Overlap**.
- **L. Endpoint Compatibility**: Compatible.
- **M. Preprocessing Compatibility**: High.
- **N. Licensing / Access**: Open access on TCIA / GDC.
- **O. Paper Feasibility**: Strong multi-omics candidate, but small sample size.

---

### 5. Ivy Glioblastoma Atlas Project (Ivy GAP)
*Allen Institute for Brain Science*

- **A. Patient Count**: $N = 41$ patients.
- **B. GBM-Specific Size**: 41 Glioblastoma cases.
- **C. Survival Time**: Overall survival available.
- **D. Censoring Status**: Available.
- **E. Clinical Data**: Age, Gender, KPS.
- **F. MRI Availability**: Pre-operative MRI scans available on TCIA.
- **G. MRI Sequences**: T1, T1ce, T2, FLAIR.
- **H. Radiomic Extraction Feasibility**: Feasible.
- **I. Genomic/Molecular Data**: Anatomic structure-guided RNA-seq.
- **J. Patient-Level Linkage**: Linked across 41 patients.
- **K. Overlap with TCGA**: **0% Overlap**.
- **L. Endpoint Compatibility**: Compatible.
- **M. Preprocessing Compatibility**: Compatible.
- **N. Licensing / Access**: Open access.
- **O. Paper Feasibility**: Very small sample size ($N=41$), best suited for exploratory sub-analyses.

---

## Comparative Candidate Matrix

| Dataset Candidate | Cohort $N$ (GBM) | 3D MRI Available? | Clinical Data? | Survival Outcome? | Genomic / Molecular? | TCGA Overlap? | Primary Research Role |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **TCGA-GBM (Current Repo)** | **97** | **Yes** | **Yes** | **Yes** | Molecular (MGMT/IDH) | Baseline | **Development & Debugging Cohort** |
| **UPenn-GBM (TCIA)** | **630** | **Yes** | **Yes** | **Yes** | Molecular (MGMT/IDH) | **0%** | **Primary Multi-Modal Federated Cohort** |
| **BraTS 2021 (Survival)** | **571** | **Yes** | Partial | **Yes** | Minimal | **High (97)** | Benchmark Imaging Reference |
| **CGGA-693** | **249** | **No** | **Yes** | **Yes** | **mRNA-seq / IDH / MGMT** | **0%** | **Separate Molecular FL Client / Validation** |
| **CPTAC-3 GBM** | **100** | Subset | **Yes** | **Yes** | Proteomics / Genomics | **0%** | Multi-Omics Secondary Validation |
| **Ivy GAP** | **41** | **Yes** | **Yes** | **Yes** | Anatomic RNA-seq | **0%** | Exploratory Anatomic Reference |
