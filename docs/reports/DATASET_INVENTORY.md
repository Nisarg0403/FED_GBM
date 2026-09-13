# Dataset Inventory & Repository Data Audit (`FED-GBM v2`)

**Project Title**: Privacy-Preserving Federated Learning for Glioblastoma Prognosis  
**Audit Purpose**: Complete audit of all local datasets, files, and cohorts currently stored within the workspace directory.  
**Audit Date**: September 2, 2026  

---

## 1. Local Workspace File & Cohort Breakdown

### Dataset 1: TCGA-GBM Fused Clinical + Radiomics Cohort
- **File Location**: `Data/clinical_unique.csv`, `Data/TCGA-images/.../TCGA_GBM_radiomicFeatures.csv`, `fused_data.csv`, `data/tcga/fused_data.csv`
- **Patient Count**: $N = 97$ unique Glioblastoma patients.
- **Patient Identifier Format**: `TCGA-XX-XXXX` (e.g., `TCGA-02-0011`, `TCGA-06-5413`).
- **Clinical Attributes Available**: Age (`demographic.age_at_index`), Gender (`demographic.gender`), Vital Status (`demographic.vital_status`).
- **Survival Data Available**:
  - `survival_time`: Continuous overall survival duration in days (Range: $22$ to $1731$ days).
  - `event`: Censoring indicator ($1 = \text{Dead}$, $0 = \text{Alive}$ at last follow-up).
  - `label_12m` / `survival_long`: Binary 12-month mortality outcome ($1 = \text{survival} > 365\text{ days}$, $0 = \text{survival} \le 365\text{ days}$).
- **MRI Availability**: 3D NIfTI brain MRI volumes (T1, T1ce, T2, FLAIR) and expert tumor segmentation masks stored in `Data/TCGA-images/PKG - BraTS-TCGA-GBM/BraTS-TCGA-GBM/Pre-operative_TCGA_GBM_NIfTI_and_Segmentations/` ($102$ total folders, $97$ matching clinical records).
- **Radiomics Availability**: 733 pre-extracted 3D PyRadiomics features per patient (Shape, Intensity, GLCM, GLRLM, GLSZM, NGTDM, GLDM).
- **Genomic Availability**: MGMT methylation and IDH mutation status available in extended GDC clinical TSV (`Data/tcga/clinical.tsv`). High-throughput RNA-seq gene expression is stored in separate GDC genomic manifests, not linked inside `fused_data.csv`.
- **Missingness**: $<2\%$ missing clinical values (imputed using median); $0\%$ missing radiomics values across the 97 patients.
- **File Formats**: `.csv`, `.tsv`, `.nii`, `.nii.gz`
- **Preprocessing Status**: Fused dataset ready; 3-stage feature selection performed.
- **Patient-Level Linkage**: **100% Patient-Level Linkage** (Every patient row connects Clinical + 3D MRI Radiomics + Survival Outcome under the same TCGA ID).

---

### Dataset 2: TCGA GDC Clinical Manifest (`clinical.tsv`)
- **File Location**: `Data/tcga/clinical.tsv`
- **Patient Count**: $N = 617$ raw patient records (includes recurrent cases, duplicates, and non-MRI TCGA-GBM patients).
- **Patient Identifier Format**: `TCGA-XX-XXXX`
- **Clinical Attributes**: Age, Gender, Ethnicity, Race, Primary Diagnosis, Tumor Stage, Vital Status, Days to Death, Days to Last Follow-up.
- **Survival Data**: Raw `days_to_death` and `days_to_last_follow_up`.
- **MRI Availability**: Raw clinical TSV only (linking manifest for GDC TCIA image download).
- **Patient-Level Linkage**: Serves as the master clinical reference table for TCGA-GBM.

---

### Dataset 3: CGGA-693 Cohort Placeholder Directory (`data/cgga693/`)
- **File Location**: `Data/cgga693/`
- **Patient Count**: $N = 0$ local files currently stored (Empty placeholder directory created in `src/preprocess_v2.py`).
- **Target External Cohort**: Chinese Glioma Genome Atlas (CGGA-693 dataset containing $N=693$ glioma patients, of which $N \approx 249$ are Glioblastoma).
- **Attributes in Literature/Schema**: Clinical variables, survival time, event status, IDH mutation, MGMT methylation, 1p19q codeletion, and mRNA-seq gene expression.
- **MRI Availability**: **NO 3D MRI SCANS** available in public CGGA repository.
- **Patient-Level Linkage**: Clinical + Genomic linkage available for CGGA patients, but **NO linkage to TCGA MRI images**.

---

### Dataset 4: BraTS Directory (`Data/brats/`)
- **File Location**: `Data/brats/`
- **Patient Count**: $N = 0$ files stored directly in `Data/brats/` (BraTS-TCGA-GBM subset is stored under `Data/TCGA-images/`).
- **Target Dataset**: MICCAI BraTS Benchmark Dataset (Brain Tumor Segmentation).
- **MRI Availability**: Multi-sequence NIfTI MRI scans and segmentations.

---

## 2. Summary Inventory Table

| Cohort Name | File Path | Patient Count ($N$) | Clinical Data | 3D MRI Scans | 3D Radiomics | Genomic / Molecular | Patient-Level Linkage Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **TCGA-GBM Fused** | `fused_data.csv` | **97** | Yes | Yes (102 NIfTI) | Yes (733 feats) | Molecular (GDC) | **Verified 100% Patient-Level Linkage** |
| **TCGA Clinical TSV** | `Data/tcga/clinical.tsv` | **617** | Yes | GDC Manifest | No | Biospecimen metadata | Master Clinical Identifier Reference |
| **CGGA-693** | `Data/cgga693/` | **0 (Empty)** | Schema defined | **No MRI** | No | mRNA-seq / IDH / MGMT | Unlinked to TCGA (Separate Chinese Cohort) |
| **BraTS Local** | `Data/brats/` | **0 (Empty)** | Schema defined | BraTS 2021 schema | No | No | NIfTI segmentations repository |

---

## 3. Key Findings of Current Data Audit

1. **Current Development Cohort**: The existing $N=97$ TCGA-GBM cohort is the **only active multi-modal dataset** currently present in the repository with verified patient-level Clinical + 3D Radiomics + Survival linkage.
2. **Missing Local Files**: `Data/cgga693/` and `Data/brats/` exist as directory stubs without downloaded raw files.
3. **Modal Boundary Verification**: TCGA-GBM ($N=97$) provides complete Clinical + 3D MRI Radiomics fusion. CGGA represents a completely separate external cohort without MRI scans.
