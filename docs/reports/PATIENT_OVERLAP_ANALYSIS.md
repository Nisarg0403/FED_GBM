# Patient Overlap & Cohort Disambiguation Analysis

**Project Title**: Privacy-Preserving Federated Learning for Glioblastoma Prognosis  
**Purpose**: Cross-reference identifiers and clinical metadata across candidate datasets to prevent patient double-counting.  
**Analysis Date**: September 2, 2026  

---

## Executive Summary of Patient Overlap Audit

A strict requirement for multi-cohort federated learning and external validation is verifying that no individual patient is represented twice across client nodes or train/test splits.

### Key Audit Findings:
1. **TCGA-GBM vs. BraTS 2021**: **HIGH OVERLAP (100% of current repo cohort)**. The $N=97$ local TCGA-GBM patients in `fused_data.csv` represent the exact `BraTS-TCGA-GBM` pre-operative imaging subset included in the MICCAI BraTS 2021 benchmark dataset.
2. **TCGA-GBM vs. UPenn-GBM**: **0% OVERLAP (Verified Independent)**. UPenn-GBM ($N=630$) is a distinct single-institution cohort collected exclusively at the Hospital of the University of Pennsylvania (Penn Medicine).
3. **TCGA-GBM vs. CGGA-693 / CGGA-325**: **0% OVERLAP (Verified Independent)**. CGGA is an Asian population cohort collected at Beijing Tiantan Hospital and affiliated Chinese neuro-oncology centers.
4. **CPTAC-3 GBM vs. TCGA-GBM**: **0% OVERLAP (Verified Independent)**. CPTAC-3 is a distinct NCI-funded prospective proteogenomic cohort.

---

## Detailed Pairwise Overlap Matrix

| Dataset Pair | Cohort 1 | Cohort 2 | Overlap Status | Overlap Count ($N$) | Action / Handling Rule |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **Pair 1** | TCGA-GBM ($N=97$) | BraTS 2021 ($N=571$ survival) | **KNOWN OVERLAP** | **97 Patients** | Exclude TCGA IDs from BraTS 2021 if pooling to prevent duplicate rows. |
| **Pair 2** | TCGA-GBM ($N=97$) | UPenn-GBM ($N=630$) | **0% OVERLAP** | **0 Patients** | Safe for pooling or separate institutional federated client assignment. |
| **Pair 3** | TCGA-GBM ($N=97$) | CGGA-693 ($N=249$ GBM) | **0% OVERLAP** | **0 Patients** | Safe for separate cohort-level federated client assignment. |
| **Pair 4** | UPenn-GBM ($N=630$) | CGGA-693 ($N=249$ GBM) | **0% OVERLAP** | **0 Patients** | Safe for separate cohort-level federated client assignment. |
| **Pair 5** | TCGA-GBM ($N=97$) | CPTAC-3 GBM ($N=100$) | **0% OVERLAP** | **0 Patients** | Safe for external validation. |
| **Pair 6** | UPenn-GBM ($N=630$) | CPTAC-3 GBM ($N=100$) | **0% OVERLAP** | **0 Patients** | Safe for external validation. |

---

## Disambiguation & Verification Rules for V3

1. **Unique Patient Identifier Standard**:
   - TCGA Cohort: `TCGA-XX-XXXX`
   - UPenn Cohort: `UPenn-GBM-XXXXX`
   - CGGA Cohort: `CGGA_XXXX`
   - CPTAC Cohort: `CPTAC-GBM-XXXXX`
2. **Explicit Verification of Non-Overlap**: When constructing multi-site federated simulations or pooled training sets, every patient ID must be verified against master lists (`splits/train_ids.csv` and `splits/test_ids.csv`).
3. **Handling Unknown Overlaps**: If a third-party institutional dataset lacks explicit anonymized patient mappings back to TCIA/GDC master registries, its overlap status must be marked as **`UNKNOWN`** and quarantined from train/test split definitions.
