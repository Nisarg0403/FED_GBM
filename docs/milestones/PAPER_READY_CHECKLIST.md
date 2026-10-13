# IEEE Journal Submission Readiness Checklist (FED-GBM v4)

## Manuscript Components
- [x] **Title & Abstract Written:** Formulated abstract highlighting multi-cohort FL, CA-FedAvg, Platt calibration, and Youden J decision thresholding.
- [x] **Introduction & Background:** Positioned relative to Oncul et al., Kickingereder et al., Abdul Salam et al., and Sheller et al.
- [x] **Methodology Detailed:** Formal mathematical definitions for CA-FedAvg, HAPA, Differential Privacy DP-SGD, Platt scaling, and Youden's J statistic.
- [x] **Results & Experiments:** Full 5-seed benchmark results across TCGA, UPenn, and CGGA cohorts (N=1,398).
- [x] **Discussion & Limitations:** Comprehensive analysis of non-IID Dirichlet limits ($\alpha=0.1$) and privacy-utility Pareto frontiers.

## Figures & Visualizations (300 DPI PDF + PNG)
- [x] **Figure 1:** Federated Learning System Architecture (`paper/figures/fig1_architecture.pdf`)
- [x] **Figure 2:** Convergence Trajectories Across FL Aggregators (`paper/figures/fig2_convergence.pdf`)
- [x] **Figure 3:** Calibration Reliability Diagrams Before vs After Platt Scaling (`paper/figures/fig3_calibration_reliability.pdf`)
- [x] **Figure 4:** Decision Curve Analysis (Net Benefit) (`paper/figures/fig4_dca_net_benefit.pdf`)
- [x] **Figure 5:** Model Robustness Across Non-IID Dirichlet Ratios (`paper/figures/fig5_dirichlet_degradation.pdf`)
- [x] **Figure 6:** Privacy-Utility Tradeoff Curve (`paper/figures/fig6_privacy_utility_pareto.pdf`)
- [x] **Figure 7:** SHAP Feature Importance (`paper/figures/fig7_shap_importance.pdf`)
- [x] **Figure 8:** Kaplan-Meier Risk Stratification (`paper/figures/fig8_kaplan_meier.pdf`)

## Tables (LaTeX .tex + CSV)
- [x] **Table I:** Multi-Cohort Benchmark Performance (`paper/tables/table1_multicohort_results.tex`)
- [x] **Table II:** Comparison with Published GBM Survival Studies (`paper/tables/table2_benchmark_comparison.tex`)
- [x] **Table III:** Differential Privacy Accounting & Utility Tradeoff (`paper/tables/table3_dp_accounting.tex`)
- [x] **Table IV:** Systematic Ablation Matrix Summary (`paper/tables/table4_ablation_summary.tex`)
- [x] **Table V:** Communication Overhead and Payload Size (`paper/tables/table5_communication_overhead.tex`)

## Reproducibility & Integrity Verification
- [x] **Unit Tests:** 110/110 passing cleanly.
- [x] **Zero Data Leakage:** Preprocessing fitted strictly on local training split per institution/fold (`LeakageFreeImputer`, `FederatedSufficientStatsScaler`).
- [x] **Multi-Seed Validation:** All reported metrics computed over 5 random seeds (42, 101, 2024, 777, 999).
- [x] **No Fabricated Data:** Empirical logs saved in `reports/` and `paper/`.
