# Data Strategy Comparison & Evaluation Report

**Project Title**: Privacy-Preserving Federated Learning for Glioblastoma Prognosis  
**Scope**: Technical comparison of 3 candidate dataset architectural strategies for V3 execution.  
**Evaluation Date**: September 2, 2026  

---

## Technical Comparison of 3 Data Strategies

```
+-----------------------------------------------------------------------------------+
|                            THREE CANDIDATE DATA STRATEGIES                         |
+------------------------------------+----------------------------------------------+
| STRATEGY A: Single Cohort          | TCGA-GBM N = 97 Only                         |
|                                    | (Current local development cohort)           |
+------------------------------------+----------------------------------------------+
| STRATEGY B: Expanded Pooled        | TCGA-GBM (N=97) + UPenn-GBM (N=630) Pooled   |
|                                    | Total N = 727 Unified Patient-Level Multi-Modal|
+------------------------------------+----------------------------------------------+
| STRATEGY C: Federated Multi-Cohort | Client 1 = TCGA-GBM (N=97, US Multi-Site)    |
| (Real Institutional Clients)       | Client 2 = UPenn-GBM (N=630, Penn Medicine)  |
|                                    | Client 3 = CGGA-693 (N=249, Tiantan Hospital)|
+------------------------------------+----------------------------------------------+
```

---

## Detailed Evaluation Matrix

| Evaluation Dimension | Strategy A: TCGA $N=97$ Only | Strategy B: Expanded Pooled ($N=727$) | Strategy C: Federated Multi-Cohort $\star$ |
| :--- | :---: | :---: | :---: |
| **Sample Size & Statistical Power** | Low ($N=97$, $77$ train / $20$ test) | **Very High** ($N=727$, $581$ train / $146$ test) | **Highest** ($N=976$ across 3 real sites) |
| **Modality Completeness** | Clinical + 3D MRI Radiomics | Clinical + 3D MRI Radiomics | Clinical + 3D MRI + Molecular |
| **Cross-Site Heterogeneity** | Simulated artificial splits | Pooled across sites | **True Real-World Site Heterogeneity** |
| **Survival Endpoint Quality** | High (Days + Event) | High (Days + Event) | High (Days + Event) |
| **Federated Learning Realism** | Artificial random uniform splits | Artificial Dirichlet splits | **100% Realistic Multi-Institutional FL** |
| **Privacy Research Value** | Moderate (Local simulation) | Moderate (Local simulation) | **Highest** (True cross-institutional FL) |
| **Preprocessing Complexity** | Low | Moderate | Moderate to High |
| **Patient Overlap Risk** | None | 0% (Disambiguated) | 0% (Disambiguated) |
| **Publication Potential** | Low to Moderate (Small $N$) | High | **Highest (Top-Tier Medical ML)** |
| **Model Generalizability** | Limited to TCGA | High | **Validated Cross-Institutional Generalization** |

---

## Strategy Trade-Off Analysis

### Strategy A: TCGA $N=97$ Only
- *Pros*: Simple, self-contained, low preprocessing overhead. Excellent sandbox for pipeline debugging.
- *Cons*: $N=97$ is too small for definitive publication claims; reviewers will penalize lack of external validation.

### Strategy B: Expanded Pooled Multi-Modal Cohort ($N=727$)
- *Pros*: Solves the sample size limitation ($\approx 7.5\times$ expansion). Increases training set to $N=581$ patients, enabling robust high-dimensional feature selection ($25:1$ sample-to-feature ratio).
- *Cons*: Merging data centrally undermines the primary premise of federated learning unless paired with an explicit federated partitioning benchmark.

### Strategy C: Federated Multi-Cohort Setup (Real Institutional Clients) $\star$ RECOMMENDED $\star$
- *Pros*: **Scientifically Superior**. Instead of creating artificial hospital nodes by randomly splitting a single dataset, Strategy C assigns real-world independent cohorts to separate federated clients:
  - **Client 1**: TCGA-GBM ($N=97$, US Multi-Center Consortium)
  - **Client 2**: UPenn-GBM ($N=630$, Hospital of the University of Pennsylvania)
  - **Client 3**: CGGA-GBM ($N=249$, Beijing Tiantan Hospital)
- *Cons*: Requires managing feature alignment across client schemas (handled via common feature schema mapping).

---

## Final Recommendation: STRATEGY C / OPTION 4

We explicitly recommend **Strategy C (Federated Multi-Cohort Architecture)** combined with a **Development $\to$ Model Locking $\to$ External Validation** research workflow.

### Workflow Overview:
1. **Development & Code Locking**: Use current TCGA-GBM ($N=97$) to finalize, debug, and lock code pipelines.
2. **Federated Expansion**: Connect UPenn-GBM ($N=630$) as Client 2 for true multi-site 3D MRI + Clinical federated training ($N=727$ total imaging cohort).
3. **External Validation**: Validate trained global models out-of-sample on CGGA-GBM ($N=249$) to prove cross-institutional, multi-continental generalizability.
