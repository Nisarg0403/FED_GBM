# Data Acquisition & Quality Control Final Report (`DATA_ACQUISITION_REPORT.md`)

**Project Title**: Privacy-Preserving Federated Learning for Glioblastoma Prognosis (`FED-GBM v3`)  
**Scope**: Final summary report on UPenn-GBM data acquisition, 10-step Quality Control pipeline execution, modality linkage verification, and cross-cohort feature compatibility.  
**Report Date**: September 2, 2026  

---

## 1. Summary of Data Acquisition & QC Results (Points 1–15)

1. **Candidate UPenn Sample Size**: **$N_{\text{candidate}} = 630$ raw patients** (`UPenn-GBM-00001` through `UPenn-GBM-00630`).
2. **Final Eligible UPenn Sample Size**: **$N_{\text{eligible}} = 574$ patients** ($460$ training patients $\approx 80\%$, $114$ held-out test patients $\approx 20\%$).
3. **Number Excluded**: **$N_{\text{excluded}} = 56$ patients** ($8.89\%$ attrition rate).
4. **Detailed Reasons for Exclusion**:
   - Stage 2 (Survival Availability): $18$ patients lacking recorded follow-up duration.
   - Stage 3 (Valid Event Flag): $1$ patient with ambiguous censoring status.
   - Stage 4 (3D MRI Availability): $18$ patients missing baseline pre-op scan series (T1ce or FLAIR).
   - Stage 5 (Valid Segmentation Mask): $11$ patients with corrupted header, missing segmentation, or 0-voxel mask.
   - Stage 6 (Artifact & Motion QC): $3$ patients with severe field distortion or truncated field of view (FOV).
   - Stage 8 (PyRadiomics Extraction): $5$ patients with non-contiguous slice mask topological mesh failures.
5. **MRI Availability**: $593$ candidate pre-operative multi-sequence 3D MRI scans.
6. **Segmentation Availability**: $582$ valid expert tumor segmentation masks.
7. **Survival & Event Availability**: $611$ valid survival time and vital status pairs.
8. **Clinical Completeness**: $100\%$ complete post training-fold median/mode imputation (Age, Gender, KPS, Resection status).
9. **Radiomics Extraction Success Rate**: **$99.13\%$** ($574 / 579$ artifact-free volumes successfully extracted without NaN/Inf values).
10. **Final Feature Count**: 733 candidate PyRadiomics 3D features pared to $p \in \{10, 15, 20, 25, 30\}$ common schema dimensions.
11. **TCGA / UPenn Feature Compatibility**: **100% Verified Feature Name & Column Position Alignment**.
12. **Patient Overlap Verification**: **0% Overlap Confirmed**. All TCGA IDs (`TCGA-XX-XXXX`) and UPenn IDs (`UPenn-GBM-XXXXX`) cross-referenced against TCIA master registries; no duplicate patients detected.
13. **Storage Requirements**: Raw NIfTI volumes $\approx 45\,\text{GB}$; extracted V3 radiomics CSVs $\approx 18\,\text{MB}$.
14. **Unexpected Problems & Handling**: Multi-label sub-region segmentations (ET, ED, NET) required explicit merging into binary Whole Tumor ($\text{WT} = \{1, 2, 4\}$) masks prior to PyRadiomics extraction.
15. **Final Readiness Recommendation**: **UPenn-GBM has passed dataset acquisition, QC, MRI-pipeline validation, and feature-schema compatibility checks and is ready for controlled V3 implementation, subject to final leakage and statistical validation.**

---

## 2. Final Dataset Allocation Table for V3 Execution

```
+-----------------------------------------------------------------------------------+
|                        FED-GBM V3 FINAL DATASET ALLOCATION                        |
+------------------------------------+----------------------------------------------+
| CLIENT 1: TCGA-GBM COHORT          | CLIENT 2: UPenn-GBM COHORT                   |
| Total Eligible: N = 97             | Total Eligible: N = 574                      |
| - Training Set (80%): N = 77       | - Training Set (80%): N = 460                |
| - Locked Test Set (20%): N = 20    | - Locked Test Set (20%): N = 114             |
+------------------------------------+----------------------------------------------+
| COMBINED FEDERATED TRAIN COHORT    | N = 537 Patients (77 TCGA + 460 UPenn)       |
| COMBINED INTERNAL HELD-OUT TEST    | N = 134 Patients (20 TCGA + 114 UPenn)       |
+------------------------------------+----------------------------------------------+
| EXPERIMENT B EXTERNAL VALIDATION   | CGGA-693 GBM Subset (N = 249 Patients)       |
+------------------------------------+----------------------------------------------+
```

---

## 3. Strict Compliance Directive

Per explicit user instructions:
- **NO models have been trained.**
- **NO federated learning code (FedAvg, FedProx, DP) has been executed.**
- **NO legacy files or existing `fused_data.csv` datasets have been overwritten.**
- **Streamlit interface remains untouched.**
- **Execution halts completely upon creation of these final data acquisition and QC reports.**
