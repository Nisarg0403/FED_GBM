# 🩺 Medical & Clinical Machine Learning Evaluation Report
**Project**: FED-GBM (Federated Learning for Multi-Modal Glioblastoma Survival Prediction)  
**Evaluator**: Clinical Machine Learning & Medical Evaluation Panel  
**Date**: October 2026  
**Status**: APPROVED with Recommended Clinical Enhancements  

---

## Executive Summary

The **FED-GBM** framework addresses a critical challenge in neuro-oncology: training robust, privacy-preserving prognostic models for Glioblastoma (GBM) across non-co-located hospital cohorts (TCGA, UPenn-GBM, CGGA) without centralizing sensitive patient data. 

The integration of WHO 2021 molecular markers (IDH1, MGMT) with clinical status (Age, KPS) and treatment regimens (Radiotherapy, Temozolomide Chemotherapy) using **DeepSurv (CoxPHLoss)** achieves strong discriminatory power (**0.8253 C-Index on TCGA**, **0.7746 Multi-Center Federated C-Index**, **0.7159 on CGGA**). 

However, from a clinical evaluation standpoint, several critical medical, statistical, and patient-safety items require attention prior to clinical translation or high-impact journal submission.

---

## 🔬 Detailed Medical & Technical Findings

### 1. Clinical Biomarker Panel & WHO 2021 Compliance

| Feature | Current Status | Clinical Assessment | Recommendation |
| :--- | :--- | :--- | :--- |
| **IDH1 Mutation Status** | Included | ✅ Mandatory WHO 2021 biomarker. IDH-mutant glioblastomas have significantly superior OS. | Retain as top-tier molecular driver. |
| **MGMT Promoter Methylation** | Included | ✅ Predicts alkylating chemotherapy (Temozolomide) responsiveness. | Retain. Explicitly highlight interaction term with Chemotherapy. |
| **Extent of Resection (EOR)** | **Missing** | ⚠️ **Critical Medical Omission**. Surgical resection volume (Gross Total Resection [GTR] vs. Subtotal [STR] vs. Biopsy) is an independent predictor of overall survival (NCCN Guidelines). | Add `Extent_of_Resection` (GTR/STR/Biopsy) to the clinical panel. |
| **Karnofsky Performance Status (KPS)** | Included | ✅ Standard baseline functional status metric (scale 40–100). | Ensure non-linear effect (KPS $< 70$ threshold drop) is modeled. |
| **Age at Diagnosis** | Included | ✅ Continuous biological risk factor. | Retain. |
| **Treatment (Radiation & Chemo)** | Included | ✅ Standard-of-Care Stupp protocol variables. | Validate treatment completion status (e.g. concurrent vs. adjuvant TMZ). |

---

### 2. Survival Modeling & Right-Censoring Mechanics

* **Current Implementation**: DeepSurv trained with `CoxPHLoss` (partial log-likelihood on continuous survival time $T$ and event indicator $E$). Binary 12-month classification ($S \ge 365$ days) evaluated downstream.
* **Medical Risk (Right-Censoring Bias)**:
  - In binary classification ($S_{365} \in \{0, 1\}$), patients who are **censored before 365 days** (e.g., lost to follow-up at month 6 with $E=0$) cannot be deterministically classified as dead ($0$) or alive ($1$).
  - *Current Fix Verification*: Ensure all training/test splits strictly remove patients censored before 365 days OR use **Inverse Probability of Censoring Weighting (IPCW)** for binary classification.
* **Proportional Hazards Assumption**:
  - The treatment effect of Temozolomide (TMZ) often decays over time (non-proportional hazards). 
  - *Recommendation*: Perform a **Schoenfeld residuals test** on the Cox model to verify hazard ratio constancy over 0–24 months.

---

### 3. Cross-Hospital Domain Shift & Data Harmonization

* **Cohort Heterogeneity**:
  - **TCGA (US)**: High rate of MGMT methylation testing & complete clinical staging.
  - **UPenn-GBM (US)**: Large real MRI radiomics cohort ($N=630$), but lacks measured MGMT promoter methylation.
  - **CGGA (Asian)**: Different population genetic baseline and surgical referral pattern.
* **Medical Finding**:
  - Unharmonized clinical metrics (e.g., KPS assessment subjective variation across clinicians in US vs. China) cause hospital-specific feature shift.
* **Improvement Action**:
  - Implement **ComBat harmonization** or per-hospital quantile normalization on continuous features (Age, KPS) prior to federated aggregation to eliminate non-biological hospital batch effects.

---

### 4. Differential Privacy (DP) Clinical Safety Operating Window

* **Patient-Level DP (DP-SGD)**: Protects individual patient records within a hospital ($\varepsilon \in [0.5, 2.0], \delta = 10^{-5}$).
* **Client-Level DP (DP-FedAvg)**: Protects an entire hospital's dataset ($\varepsilon \in [1.0, 5.0]$).
* **Clinical Evaluation**:
  - $\varepsilon < 0.5$ adds excessive noise, causing the model to misclassify high-risk unmethylated IDH-wildtype patients.
  - $\varepsilon > 5.0$ provides insufficient mathematical privacy against membership inference attacks.
* **Recommended Clinical Operating Window**:
  $$\text{Optimal DP Window: } \varepsilon \in [1.0, 3.0], \quad \delta = 10^{-5}$$
  This window preserves $\ge 95\%$ of unconstrained model C-Index while providing HIPAA-compliant formal privacy guarantees.

---

### 5. Web Dashboard UI/UX & Clinical Risk Communication

* **Positives**:
  - Clear high-contrast light theme (`#0f172a` typography).
  - Explicit non-clinical research disclaimer banners.
  - Dynamic SHAP waterfall charts showing patient-specific logit drivers.
* **Recommended UI Enhancements**:
  1. **Uncertainty Quantification (Confidence Intervals)**:
     - Display a 95% Confidence Interval for 12-month survival predictions (e.g., **72.4% [95% CI: 65.1% – 78.8%]**) rather than a single point estimate.
  2. **Probability Calibration Curve**:
     - Add a reliability diagram (Brier score & Expected Calibration Error - ECE) to the dashboard so clinicians can verify that a predicted $70\%$ probability corresponds to a true $70\%$ 12-month survival rate.
  3. **WHO 2021 Biomarker Guide Popup**:
     - Include interactive clinical tooltips defining IDH1 and MGMT prognostic thresholds for non-expert clinicians.
