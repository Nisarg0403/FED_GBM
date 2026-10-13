# Common Feature Schema & Compatibility Report (`V3_FEATURE_SCHEMA_REPORT.md`)

**Project Title**: Privacy-Preserving Federated Learning for Glioblastoma Prognosis (`FED-GBM v3`)  
**Scope**: Verification of feature name compatibility, column alignment, common feature schema definitions, and leakage-free federated normalization rules across Client 1 (TCGA) and Client 2 (UPenn).  
**Report Date**: September 2, 2026  

---

## 1. Feature Name Alignment & Compatibility Analysis

A pilot extraction test comparing raw/preprocessed TCGA sample volumes and UPenn sample volumes under the V3 IBSI-compliant extraction protocol verified **100% 1:1 feature name alignment** across 733 radiomic candidate features.

- **Feature Naming Standard**: IBSI standard feature taxonomy (e.g., `original_shape_Sphericity`, `original_firstorder_Mean`, `original_glcm_Correlation`, `original_glrlm_RunLengthNonUniformity`).
- **Feature Matching Rate**: $733 / 733 \text{ features } (100.0\%)$ exact match across Client 1 and Client 2.
- **Column Position Rule**: For any input matrix fed into the PyTorch `SurvivalMLP` model, feature $j$ occupies the **exact same column index** across Client 1 and Client 2.

---

## 2. Pre-Defined Common Feature Schema

The unified feature input matrix is structured into three distinct feature blocks ($p \in \{10, 15, 20, 25, 30\}$ features total):

| Column Index | Feature Identifier | Data Category | Data Type | Imputation Rule (Fit on Train Only) |
| :-: | :--- | :--- | :---: | :--- |
| **Col 0** | `Age` | Clinical | Float (Years) | Training Median |
| **Col 1** | `Gender` | Clinical | Binary ($1=\text{Male}, 0=\text{Female}$) | Training Mode |
| **Col 2** | `KPS` | Clinical | Continuous ($0$--$100$) | Training Median |
| **Col 3** | `Resection_Status` | Clinical | Binary ($1=\text{GTR}, 0=\text{STR/Biopsy}$) | Training Mode |
| **Col 4** | `MGMT_Methylation` | Molecular (Sub-Model) | Binary ($1=\text{Methylated}, 0=\text{Unmethylated}$) | Training Mode |
| **Col 5** | `IDH_Mutation` | Molecular (Sub-Model) | Binary ($1=\text{Mutant}, 0=\text{Wildtype}$) | Training Mode |
| **Col 6..p-1** | `V3_Radiomic_Feature_1..K` | PyRadiomics 3D | Float (Standardized) | Training Median |

---

## 3. Federated Normalization & Leakage-Free Scaling Rules

To ensure global feature representations are normalized across Client 1 and Client 2 without centralizing patient records or causing test set contamination, the following protocol is strictly enforced:

```
[CLIENT 1: TCGA TRAIN (N=77)]                   [CLIENT 2: UPenn TRAIN (N=460)]
              |                                               |
   Compute Local Sum S_1,j                         Compute Local Sum S_2,j
   Compute Sum Squares SS_1,j                      Compute Sum Squares SS_2,j
              |                                               |
              +-----------------------+-----------------------+
                                      |
                                      v
                 Central Server Privacy-Preserving Summation
                   N_global = N_1,train + N_2,train = 537
                   μ_global,j  = (S_1,j + S_2,j) / N_global
                   σ_global,j  = sqrt( (SS_1,j + SS_2,j)/N_global - μ_global,j^2 )
                                      |
                                      v
                 Broadcast (μ_global,j, σ_global,j) to Clients
                                      |
            +-------------------------+-------------------------+
            |                                                   |
            v                                                   v
Standardize TCGA Train & Test Rows                Standardize UPenn Train & Test Rows
(Using μ_global, σ_global from TRAIN)             (Using μ_global, σ_global from TRAIN)
```

### Strict Normalization Leakage Rules:
1. **Train-Only Statistics**: Global mean $\mu_{\text{global},j}$ and variance $\sigma^2_{\text{global},j}$ are computed **EXCLUSIVELY using training patients** ($N_{\text{train},1} = 77$, $N_{\text{train},2} = 460$, total $N_{\text{train}} = 537$).
2. **Locked Test Isolation**: Held-out test patients ($N_{\text{test},1} = 20$, $N_{\text{test},2} = 114$, total $N_{\text{test}} = 134$) and CGGA external validation patients ($N = 249$) are **NEVER included** in the calculation of $\mu_{\text{global},j}$ or $\sigma^2_{\text{global},j}$. Test vectors are transformed using the previously locked training statistics.
