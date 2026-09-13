# Phase 6 & Phase 6.2 Specification — Formal Client-Level DP Mechanism & Adjacency Consistency Audit

## 1. Objective & Overview
Phase 6 implements **Client-Level Differential Privacy** for the federated learning pipeline (Experiment A: TCGA + UPenn). 

Unlike Phase 5 (Patient-Level DP-SGD), where the privacy unit is an individual patient's record within local datasets, Phase 6 defines the **privacy unit as an entire federated client dataset** (e.g., Client 1 TCGA or Client 2 UPenn).

---

## 2. Declared Privacy Adjacency & Scope (Adjacency A)

### A. Formal Adjacency Definition
- **Adjacency Model**: **Fixed Participating Clients / Replacement (Dataset-Modification) Adjacency**.
- **Public Metadata**: The set of $K=2$ participating institutional clients $\{ \text{TCGA}, \text{UPenn} \}$ and their sample sizes $\{n_1 = 77, n_2 = 460\}$ are fixed, public metadata of the 2-hospital federation.
- **Neighboring Datasets**: Two federated dataset configurations $D = \{D_1, D_2\}$ and $D' = \{D_1', D_2'\}$ are adjacent ($D \sim D'$) if they differ in the complete dataset/model update of **exactly one client $j$** ($D_j \neq D_j'$, while $D_k = D_k'$ for all $k \neq j$).

### B. Explicit Scope & Claim Limitations
- **No Client Participation Privacy Claim**: Because participating client identities $\{ \text{TCGA}, \text{UPenn} \}$ are fixed and public, this mechanism **does NOT hide client participation or presence** ($q_{\text{client}} = 1.0$).
- **Formal Privacy Scope**: The formal privacy claim is **strictly limited to protecting the client's dataset contribution / model update vector $\Delta_j$ under dataset modification** (replacement of $D_j$ with $D_j'$).

---

## 3. Global Replacement Sensitivity Derivations

Under **Replacement Adjacency (Adjacency A)**, each client update is clipped to maximum $L_2$ norm $\|\Delta_k\|_2 \le S$ (where $S = 1.0$). The worst-case difference between two clipped updates $\Delta_j$ and $\Delta_j'$ is:

$$\|\Delta_j - \Delta_j'\|_2 \le \|\Delta_j\|_2 + \|\Delta_j'\|_2 \le 2 S$$

The $L_2$ global replacement sensitivity of the aggregated update vector $\bar{\Delta}$ is derived as:

### A. Sample-Size Weighted Aggregation ($w_1 = \frac{77}{537} \approx 0.1434, w_2 = \frac{460}{537} \approx 0.8566$)
$$\Delta S_{\text{replacement}} = 2 \cdot w_{\text{max}} \cdot S = 2 \times \left(\frac{460}{537}\right) \times 1.0 \approx 1.7132216 \cdot S$$

### B. Equal-Weight Aggregation ($w_1 = w_2 = 0.5$)
$$\Delta S_{\text{replacement}} = 2 \cdot \left(\frac{1}{K}\right) \cdot S = 2 \times 0.5 \times 1.0 = 1.0000 \cdot S$$

---

## 4. Server Noise & Analytical RDP Accounting

### A. Server Gaussian Noise Injection
Calibrated Gaussian noise is added at the server directly proportional to the replacement sensitivity:

$$\text{noise\_std} = \sigma \cdot \Delta S_{\text{replacement}}$$

$$\Delta_{\text{private}} = \bar{\Delta} + \mathcal{N}\left(0, (\sigma \cdot \Delta S_{\text{replacement}})^2 \mathbf{I}\right)$$

### B. Closed-Form RDP Minimization (30 Rounds)
For deterministic client participation ($q_{\text{client}} = 1.0$) across $T=30$ rounds, noise scaling $\sigma_{\text{noise}} = \sigma \cdot \Delta S$ yields per-round RDP:

$$\epsilon_{\text{RDP}}(\alpha) = \frac{\alpha (\Delta S)^2}{2 (\sigma \cdot \Delta S)^2} = \frac{\alpha}{2 \sigma^2}$$

The exact optimal order $\alpha_{\text{opt}}$ minimizing $(\epsilon, \delta)$-DP at target $\delta = 10^{-5}$ is given in closed form:

$$\alpha_{\text{opt}} = 1 + \sqrt{\frac{2 \sigma^2 \ln(1/\delta)}{30}}$$

$$\epsilon_{\text{exact}}(\delta) = \frac{30}{2 \sigma^2} + \frac{\sqrt{60 \ln(1/\delta)}}{\sigma}$$

---

## 5. Corrected 30-Round Epsilon Results Table ($S = 1.0, T = 30, \delta = 10^{-5}$)

| Aggregation Scheme | Sensitivity Mode | Replacement Sensitivity ($\Delta S$) | Noise Multiplier ($\sigma$) | Server Noise Std ($\sigma_{\text{noise}}$) | Optimal Order ($\alpha_{\text{opt}}$) | Exact Closed-Form 30-Round Epsilon ($\epsilon$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Sample-Size Weighted** | Replacement | **1.7132216** | $0.05$ | **0.085661** | $1.0438$ | **6,525.65** |
| **Sample-Size Weighted** | Replacement | **1.7132216** | $0.10$ | **0.171322** | $1.0876$ | **1,762.83** |
| **Sample-Size Weighted** | Replacement | **1.7132216** | $0.20$ | **0.342644** | $1.1752$ | **506.41** |
| **Equal Weighted** | Replacement | **1.0000000** | $0.05$ | **0.050000** | $1.0438$ | **6,525.65** |
| **Equal Weighted** | Replacement | **1.0000000** | $0.10$ | **0.100000** | $1.0876$ | **1,762.83** |
| **Equal Weighted** | Replacement | **1.0000000** | $0.20$ | **0.200000** | $1.1752$ | **506.41** |

---

## 6. Methodological & Population Limitations ($K=2$ Clients)
In Experiment A, only $K=2$ institutional cohorts exist (TCGA $n_1=77$, UPenn $n_2=460$). 

Because $K=2$ and participation is **deterministic** ($q_{\text{client}} = 1.0$ every round), client-level DP cannot benefit from subsampling privacy amplification. This implementation serves as a **methodological privacy-mechanism baseline** and does NOT provide strong client anonymity (which requires large client populations $K \gg 100$).
