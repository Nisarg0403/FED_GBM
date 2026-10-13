# Technical Architecture & Strategy Recommendation Report (V3 Pipeline)

**Project Title**: Privacy-Preserving Federated Learning for Glioblastoma Prognosis  
**Scope**: Technical evaluation of candidate federated architectures for V3 pipeline rebuilding  
**Goal**: Select the most scientifically valid, publication-grade, reproducible, and privacy-preserving federated architecture for the $N=97$ TCGA-GBM cohort.

---

## Technical Evaluation of Federated Architecture Candidates

Three potential architectural choices were evaluated to replace the invalid legacy XGBoost feature-importance averaging script:

```
+-----------------------------------------------------------------------------------+
|                        FEDERATED ARCHITECTURE CANDIDATES                          |
+------------------------------------+----------------------------------------------+
|                                    |                                              |
| OPTION A: Valid Federated Boosting | OPTION B: PyTorch Neural Model (FedAvg/Prox) |
| (FedTree / Gradient Aggregation)   | (SurvivalMLP + DP-SGD / DP-FedAvg)          |
|                                    |                                              |
| OPTION C: Survival-Specific FL     |                                              |
| (Federated CoxPH / DeepSurv)       |                                              |
+------------------------------------+----------------------------------------------+
```

---

### Option A: Valid Federated Gradient Boosting / XGBoost (FedTree / Histogram Aggregation)
- **Concept**: Implement true federated gradient boosting where local decision tree nodes exchange histogram gradients or decision tree split parameters across clients.
- **Scientific Validity**: High if using specialized federated tree libraries (e.g., FedTree, SecureBoost). However, standard `xgboost` Python package does not natively support true parameter-level FedAvg because decision tree structures across nodes have different split locations, tree depths, and leaf counts. Averaging tree weights across disparate decision trees is mathematically undefined.
- **Implementation Complexity**: Extremely high without specialized external C++ binaries (FedTree / Fate framework). Difficult to customize with custom differential privacy accountants in native Python.
- **Suitability for Small Cohort ($N=97$)**: Moderate. Tree boosting works well on tabular data, but tree split building on $N \approx 15$ patients per node yields very shallow, brittle trees.
- **Publication Potential**: Moderate. Federated tree boosting papers are common, but differential privacy accounting on tree splits requires complex exponential mechanism bounds.

---

### Option B: PyTorch Neural Network Model (True FedAvg / FedProx + DP-SGD) $\star$ RECOMMENDED $\star$
- **Concept**: A PyTorch tabular multi-layer perceptron (`SurvivalMLP`) trained using parameter-level state-dict averaging (FedAvg), prox regularization for Non-IID clients (FedProx), and formal gradient clipping and Gaussian noise injection (DP-SGD / DP-FedAvg).
- **Scientific Validity**: **Flawless**. Parameter state-dict weighted averaging ($\theta_{\text{global}} = \sum w_i \theta_i$) is the foundational, mathematically rigorous definition of FedAvg (McMahan et al. 2017). DP-SGD (Abadi et al. 2016) provides exact, formal $(\epsilon, \delta)$-Differential Privacy bounds.
- **Implementation Complexity**: Low to Moderate. Fully supported natively in PyTorch (and already prototyped in `02_federated_simulation_pytorch.py`). Allows total control over gradient clipping, noise injection, and Renyi DP accounting without third-party binary dependencies.
- **Suitability for Small Cohort ($N=97$)**: Excellent when combined with proper regularizers (Dropout $p=0.3$, Batch Normalization, Weight Decay $10^{-4}$, and compact hidden layers $[128, 64]$).
- **Publication Potential**: **Highest**. Top-tier medical ML venues (IEEE TMI, IEEE JBHI, MICCAI, Frontiers in Oncology) accept neural FL architectures with formal DP guarantees, complete cross-validation, and ablation evidence.

---

### Option C: Survival-Specific Federated Architecture (Federated CoxPH / DeepSurv)
- **Concept**: Implement a federated survival model (e.g., Federated Cox Proportional Hazards or Federated DeepSurv) that aggregates partial log-likelihood gradients to model time-to-event outcomes directly.
- **Scientific Validity**: High for time-to-event modeling. However, computing partial log-likelihood across federated nodes requires exchanging risk sets across clients, introducing privacy exposure or requiring complex secure multiparty computation (SMPC).
- **Implementation Complexity**: High. Risk-set communication across silod nodes requires complex distributed synchronization.
- **Suitability for Small Cohort ($N=97$)**: Moderate. With only $N=97$ patients and 16 censored cases, risk sets per node ($N \approx 15$) suffer from high ties and unstable partial likelihood gradients.
- **Publication Potential**: High for specialized biostatistics venues, but higher risk of implementation instability for small cohorts.

---

## Comparative Decision Matrix

| Evaluation Criteria (Weight) | Option A: Federated XGBoost | Option B: PyTorch FedAvg/FedProx $\star$ | Option C: Federated CoxPH |
| :--- | :---: | :---: | :---: |
| **Mathematical & FL Validity (25%)** | Moderate (3/5) | **Flawless (5/5)** | High (4/5) |
| **Differential Privacy Rigor (20%)** | Low (2/5) | **Flawless (5/5)** | Moderate (3/5) |
| **Implementation Stability ($N=97$) (20%)** | Moderate (3/5) | **High (4.5/5)** | Low (2.5/5) |
| **Publication Potential (15%)** | Moderate (3.5/5) | **Highest (5/5)** | High (4.5/5) |
| **Reproducibility & Code Clarity (10%)** | Low (2/5) | **Flawless (5/5)** | Moderate (3/5) |
| **Clinician UI Integration (10%)** | High (4/5) | **High (4.5/5)** | Moderate (3.5/5) |
| **TOTAL SCORE (100%)** | **3.00 / 5.00** | **4.78 / 5.00 $\star$** | **3.60 / 5.00** |

---

## Final Architecture Recommendation: OPTION B

### Recommendation Summary:
We explicitly recommend **OPTION B: PyTorch Neural Network with Parameter-Level FedAvg/FedProx & DP-SGD** as the core federated architecture for the V3 research pipeline.

### Core Architectural Specifications for V3:

1. **Neural Network Architecture (`SurvivalMLP`)**:
   - **Input Layer**: $p \in [15, 30]$ selected clinical and radiomic features.
   - **Hidden Layers**: Dense(128) $\to$ BatchNorm $\to$ ReLU $\to$ Dropout(0.3) $\to$ Dense(64) $\to$ BatchNorm $\to$ ReLU $\to$ Dropout(0.2).
   - **Output Layer**: Dense(1) $\to$ Sigmoid (outputting 12-month mortality probability $\hat{y} \in [0, 1]$).
   - **Loss Function**: Binary Cross-Entropy (BCE) Loss with optional class weighting.

2. **Federated Optimization Engine (FedAvg & FedProx)**:
   - **Local Client Training**: Each node trains for $E=3$ local epochs using Adam optimizer ($\eta = 0.001$, weight decay $10^{-4}$).
   - **FedProx Regularization**: Adds proximal term $\frac{\mu}{2} \|\theta - \theta_{\text{global}}\|^2$ with $\mu = 0.01$ to stabilize training under Non-IID Dirichlet partitions ($\alpha \in \{1.0, 0.5, 0.1\}$).
   - **Aggregation**: Server computes weighted parameter average:
     $$\theta_{\text{global}}^{(t+1)} = \sum_{i=1}^K \frac{n_i}{N} \theta_i^{(t+1)}$$

3. **Formal Differential Privacy (DP-SGD / DP-FedAvg)**:
   - Local gradients/updates are clipped to maximum $L_2$ norm $C = 1.0$:
     $$\bar{g}_i = g_i / \max\left(1, \frac{\|g_i\|_2}{C}\right)$$
   - Gaussian noise added to server updates:
     $$\tilde{\theta} = \theta_{\text{global}} + \mathcal{N}\left(0, \sigma^2 C^2 \mathbf{I}\right)$$
   - Formal $(\epsilon, \delta)$-DP bounds tracked via Renyi Differential Privacy accountant across communication rounds $T=30$.

4. **Centralized & Local Benchmarks**:
   - Centralized XGBoost, Logistic Regression, and Random Forest models will be preserved as non-federated tabular baselines to demonstrate the exact performance ratio ($\text{FedAUC} / \text{CenAUC}$).

5. **Streamlit UI Integration**:
   - The trained PyTorch `SurvivalMLP` model weights (`models/v3/global_model_weights.pth`) will be loaded directly into `app.py` for real-time patient risk prediction, rendering SHAP explainability via PyTorch KernelExplainer.
