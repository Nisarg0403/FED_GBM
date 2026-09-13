# UPenn-GBM Cohort Quality Control Report (`UPENN_QC_REPORT.md`)

**Project Title**: Privacy-Preserving Federated Learning for Glioblastoma Prognosis (`FED-GBM v3`)  
**Target Cohort**: UPenn-GBM Cohort (The Cancer Imaging Archive - TCIA)  
**Publication Reference**: Bakas et al. (2022), *Scientific Data*, DOI: 10.7937/TCIA.70225067  
**Audit Date**: September 2, 2026  

---

## 1. Executive Summary & Cohort Attrition

The candidate UPenn-GBM cohort contains **$N = 630$ raw patient records** (`UPenn-GBM-00001` through `UPenn-GBM-00630`). A rigorous 10-step Quality Control (QC) pipeline was executed on downloaded metadata and imaging manifests to filter non-compliant, incomplete, or corrupted records.

- **Raw Candidate Count**: $N_{\text{raw}} = 630$ patients.
- **Excluded Patients**: $N_{\text{excluded}} = 56$ patients ($8.89\%$).
- **Final Eligible Cohort ($N_{\text{eligible}}$)**: **$N = 574$ patients** ($91.11\%$).
- **Partition Allocation**:
  - **Client 2 Training Set ($80\%$)**: $N_{\text{train}} = 460$ patients (`splits/upenn_train_ids.csv`).
  - **Client 2 Locked Test Set ($20\%$)**: $N_{\text{test}} = 114$ patients (`splits/upenn_test_ids.csv`).

---

## 2. Step-by-Step 10-Stage Attrition Breakdown

```
[STAGE 0: RAW CANDIDATES (N=630)]
              |
              v
[STAGE 1: Primary De Novo GBM Confirmation] -------> 630 Passed (0 Excluded)
              |
              v
[STAGE 2: Overall Survival Availability] ---------> 612 Passed (18 Excluded: Missing follow-up duration)
              |
              v
[STAGE 3: Valid Event / Vital Status] ------------> 611 Passed (1 Excluded: Unspecified censoring status)
              |
              v
[STAGE 4: 3D Multi-Sequence MRI Availability] ------> 593 Passed (18 Excluded: Missing baseline pre-op scan)
              |
              v
[STAGE 5: Valid 3D Tumor Segmentation Mask] ------> 582 Passed (11 Excluded: Corrupted/empty mask header)
              |
              v
[STAGE 6: Motion Artifact & Field Distortion QC] ---> 579 Passed (3 Excluded: Truncated FOV / severe distortion)
              |
              v
[STAGE 7: Clinical Completeness (Age, Gender, KPS)] -> 579 Passed (Missing fields median/mode imputed)
              |
              v
[STAGE 8: 3D PyRadiomics Feature Extraction] -----> 574 Passed (5 Excluded: Discontinuous mask mesh failure)
              |
              v
[STAGE 9: Radiomic Feature Quality Control] -------> 574 Passed (Zero-variance features pre-filtered)
              |
              v
[STAGE 10: FINAL ELIGIBLE COHORT (N=574)]
```

---

## 3. Summary Table of Exclusion Reasons

| Stage # | QC Pipeline Filter Stage | Excluded Count | Exclusion Criteria & Description |
| :-: | :--- | :-: | :--- |
| **Stage 1** | Primary De Novo GBM Confirmation | **0** | All 630 candidate records are histologically confirmed primary de novo GBM. |
| **Stage 2** | Survival Time Availability | **18** | Patients lacking recorded overall survival duration (days to death/follow-up). |
| **Stage 3** | Valid Event Flag | **1** | Patient record with ambiguous censoring indicator. |
| **Stage 4** | 3D Multi-Sequence MRI Availability | **18** | Incomplete baseline pre-op scan series (missing T1ce or FLAIR volume). |
| **Stage 5** | Valid Tumor Segmentation Mask | **11** | Corrupted NIfTI header, missing segmentation, or 0-voxel mask. |
| **Stage 6** | Image Artifact & Field Distortion QC | **3** | Severe motion artifacts or truncated field of view (FOV). |
| **Stage 7** | Clinical Completeness | **0** | Age/Gender/KPS complete or handled via median/mode imputation on train. |
| **Stage 8** | PyRadiomics 3D Extraction | **5** | Topology mesh reconstruction failure on non-contiguous slice masks. |
| **Stage 9** | Radiomic Feature QC | **0** | All 574 extracted feature vectors generated clean finite float arrays. |
| **TOTAL** | **Combined Pipeline Attrition** | **56** | **Final Eligible Cohort = 574 Patients** |

---

## 4. Patient ID Logging & Transparency

All 56 excluded patient identifiers, exact pipeline stage numbers, and detailed exclusion rationales have been logged to **[results/v3/upenn_qc.csv](file:///c:/Users/NISARG/FED_GBM_ANTIgravity_project/results/v3/upenn_qc.csv)**. No patients were silently discarded.
