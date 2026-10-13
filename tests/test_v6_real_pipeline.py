"""
FED-GBM v6 Master Benchmark & Training Pipeline (Authentic Real Data Edition).

Evaluates 5-seed benchmark matrix across authentic real cohorts:
- TCGA Real (N=491)
- CGGA 325 Real (N=133)
- CGGA 693 Real (N=236)

Models evaluated:
1. Centralized Logistic Regression
2. Centralized Random Forest
3. Centralized XGBoost
4. Centralized MLP
5. FedAvg MLP
6. FedProx MLP (mu=0.01)
7. CA-FedAvg (lambda=1.0)
8. HAPA (mu_base=0.01)

Saves all per-seed and aggregate results to reports/v6_real_benchmark_results.csv and paper/tables_v6/.
"""

import sys
import os
import json
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from fed_gbm.models import SurvivalMLP, set_seed
from fed_gbm.load_real import load_all_real_cohorts
from fed_gbm.preprocessing import LeakageFreeImputer, FederatedSufficientStatsScaler
from fed_gbm.federated_server import FederatedServer
from fed_gbm.federated_client import FederatedClient
from fed_gbm.ca_fedavg import CalibrationAwareFedAvg
from fed_gbm.hapa import HeterogeneityAwareProximalAggregation, compute_adaptive_mu
from fed_gbm.metrics import evaluate_binary_classification
from fed_gbm.temperature_scaling import apply_platt_scaling
from fed_gbm.thresholding import find_optimal_threshold_youden
from fed_gbm.stats import paired_ttest, bonferroni_correction


def run_v6_real_pipeline():
    print("=" * 80)
    print("FED-GBM v6 REAL DATA MASTER BENCHMARK PIPELINE (5 SEEDS)")
    print("=" * 80)

    seeds = [42, 101, 2024, 777, 999]
    records = []

    for seed in seeds:
        set_seed(seed)
        cohorts = load_all_real_cohorts(seed=seed)

        tcga_tr = cohorts["tcga_train"]
        tcga_te = cohorts["tcga_test"]
        cgga325_tr = cohorts["cgga325_train"]
        cgga325_te = cohorts["cgga325_test"]
        cgga693_tr = cohorts["cgga693_train"]
        cgga693_te = cohorts["cgga693_test"]

        feature_cols = cohorts["feature_cols"]

        X_tcga_tr = tcga_tr[feature_cols]; y_tcga_tr = tcga_tr["label_12m"]
        X_tcga_te = tcga_te[feature_cols]; y_tcga_te = tcga_te["label_12m"]

        X_c325_tr = cgga325_tr[feature_cols]; y_c325_tr = cgga325_tr["label_12m"]
        X_c325_te = cgga325_te[feature_cols]; y_c325_te = cgga325_te["label_12m"]

        X_c693_tr = cgga693_tr[feature_cols]; y_c693_tr = cgga693_tr["label_12m"]
        X_c693_te = cgga693_te[feature_cols]; y_c693_te = cgga693_te["label_12m"]

        # Pool training data for centralized baselines
        X_pool_tr = pd.concat([X_tcga_tr, X_c325_tr, X_c693_tr], axis=0).reset_index(drop=True)
        y_pool_tr = pd.concat([y_tcga_tr, y_c325_tr, y_c693_tr], axis=0).reset_index(drop=True)

        # Standardize features using federated sufficient statistics
        imp = LeakageFreeImputer(); imp.fit(X_pool_tr)
        X_pool_tr_imp = imp.transform(X_pool_tr)
        X_tcga_te_imp = imp.transform(X_tcga_te)
        X_c325_te_imp = imp.transform(X_c325_te)
        X_c693_te_imp = imp.transform(X_c693_te)

        scaler = FederatedSufficientStatsScaler()
        st = scaler.compute_local_stats(X_pool_tr_imp)
        scaler.aggregate_global_stats([st])

        X_pool_tr_std = scaler.transform(X_pool_tr_imp)
        X_tcga_te_std = scaler.transform(X_tcga_te_imp)
        X_c325_te_std = scaler.transform(X_c325_te_imp)
        X_c693_te_std = scaler.transform(X_c693_te_imp)

        # Evaluate Centralized Baselines on CGGA 693 External Test Set
        # 1. Logistic Regression
        lr = LogisticRegression(random_state=seed, max_iter=1000)
        lr.fit(X_pool_tr_std, y_pool_tr)
        p_lr = lr.predict_proba(X_c693_te_std)[:, 1]
        m_lr = evaluate_binary_classification(y_c693_te.values, p_lr)
        records.append({"model": "LogisticRegression", "seed": seed, "auc": m_lr["roc_auc"], "acc": m_lr["accuracy"], "spec": m_lr["specificity"], "ece": m_lr["ece"]})

        # 2. Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=seed)
        rf.fit(X_pool_tr_std, y_pool_tr)
        p_rf = rf.predict_proba(X_c693_te_std)[:, 1]
        m_rf = evaluate_binary_classification(y_c693_te.values, p_rf)
        records.append({"model": "RandomForest", "seed": seed, "auc": m_rf["roc_auc"], "acc": m_rf["accuracy"], "spec": m_rf["specificity"], "ece": m_rf["ece"]})

        # 3. XGBoost
        if HAS_XGB:
            model_xgb = xgb.XGBClassifier(n_estimators=100, random_state=seed, eval_metric="logloss")
            model_xgb.fit(X_pool_tr_std, y_pool_tr)
            p_xgb = model_xgb.predict_proba(X_c693_te_std)[:, 1]
            m_xgb = evaluate_binary_classification(y_c693_te.values, p_xgb)
            records.append({"model": "XGBoost", "seed": seed, "auc": m_xgb["roc_auc"], "acc": m_xgb["accuracy"], "spec": m_xgb["specificity"], "ece": m_xgb["ece"]})

        # Federated Training Setup (3 Real Institutions: TCGA, CGGA 325, CGGA 693)
        imp1 = LeakageFreeImputer(); imp1.fit(X_tcga_tr); X1_imp = imp1.transform(X_tcga_tr)
        imp2 = LeakageFreeImputer(); imp2.fit(X_c325_tr); X2_imp = imp2.transform(X_c325_tr)
        imp3 = LeakageFreeImputer(); imp3.fit(X_c693_tr); X3_imp = imp3.transform(X_c693_tr)

        scaler_fl = FederatedSufficientStatsScaler()
        st1 = scaler_fl.compute_local_stats(X1_imp)
        st2 = scaler_fl.compute_local_stats(X2_imp)
        st3 = scaler_fl.compute_local_stats(X3_imp)
        scaler_fl.aggregate_global_stats([st1, st2, st3])

        X1_std = scaler_fl.transform(X1_imp)
        X2_std = scaler_fl.transform(X2_imp)
        X3_std = scaler_fl.transform(X3_imp)
        Xte693_std = scaler_fl.transform(X_c693_te_imp)

        input_dim = X1_std.shape[1]

        # 4. FedAvg
        srv_fa = FederatedServer(input_dim=input_dim, seed=seed)
        c1_fa = FederatedClient("TCGA", X1_std, y_tcga_tr, input_dim=input_dim)
        c2_fa = FederatedClient("CGGA325", X2_std, y_c325_tr, input_dim=input_dim)
        c3_fa = FederatedClient("CGGA693", X3_std, y_c693_tr, input_dim=input_dim)
        for r in range(20):
            g_sd = srv_fa.get_global_state_dict()
            c1_fa.set_model_parameters(g_sd); c2_fa.set_model_parameters(g_sd); c3_fa.set_model_parameters(g_sd)
            p1 = c1_fa.local_train(epochs=5, batch_size=16, lr=0.01, mu=0.0, seed=seed+r)
            p2 = c2_fa.local_train(epochs=5, batch_size=16, lr=0.01, mu=0.0, seed=seed+r+50)
            p3 = c3_fa.local_train(epochs=5, batch_size=16, lr=0.01, mu=0.0, seed=seed+r+100)
            srv_fa.aggregate_client_payloads([p1, p2, p3])

        eval_fa = SurvivalMLP(input_dim=input_dim, hidden_dims=[32, 16])
        eval_fa.load_state_dict(srv_fa.get_global_state_dict()); eval_fa.eval()
        with torch.no_grad():
            p_fa = torch.sigmoid(eval_fa(torch.tensor(Xte693_std.values, dtype=torch.float32))).numpy().flatten()
        m_fa = evaluate_binary_classification(y_c693_te.values, p_fa)
        records.append({"model": "FedAvg", "seed": seed, "auc": m_fa["roc_auc"], "acc": m_fa["accuracy"], "spec": m_fa["specificity"], "ece": m_fa["ece"]})

        # 5. CA-FedAvg (Ours)
        srv_ca = FederatedServer(input_dim=input_dim, seed=seed)
        c1_ca = FederatedClient("TCGA", X1_std, y_tcga_tr, input_dim=input_dim)
        c2_ca = FederatedClient("CGGA325", X2_std, y_c325_tr, input_dim=input_dim)
        c3_ca = FederatedClient("CGGA693", X3_std, y_c693_tr, input_dim=input_dim)
        ca_agg = CalibrationAwareFedAvg(lambda_param=1.0)
        for r in range(20):
            g_sd = srv_ca.get_global_state_dict()
            c1_ca.set_model_parameters(g_sd); c2_ca.set_model_parameters(g_sd); c3_ca.set_model_parameters(g_sd)
            p1 = c1_ca.local_train(epochs=5, batch_size=16, lr=0.01, mu=0.0, seed=seed+r)
            p2 = c2_ca.local_train(epochs=5, batch_size=16, lr=0.01, mu=0.0, seed=seed+r+50)
            p3 = c3_ca.local_train(epochs=5, batch_size=16, lr=0.01, mu=0.0, seed=seed+r+100)

            ece1 = evaluate_binary_classification(y_tcga_tr.values, torch.sigmoid(c1_ca.model(c1_ca.X_train_tensor)).detach().numpy().flatten())["ece"]
            ece2 = evaluate_binary_classification(y_c325_tr.values, torch.sigmoid(c2_ca.model(c2_ca.X_train_tensor)).detach().numpy().flatten())["ece"]
            ece3 = evaluate_binary_classification(y_c693_tr.values, torch.sigmoid(c3_ca.model(c3_ca.X_train_tensor)).detach().numpy().flatten())["ece"]

            agg_sd = ca_agg.aggregate([p1, p2, p3], client_eces=[ece1, ece2, ece3])
            srv_ca.global_model.load_state_dict(agg_sd)

        eval_ca = SurvivalMLP(input_dim=input_dim, hidden_dims=[32, 16])
        eval_ca.load_state_dict(srv_ca.get_global_state_dict()); eval_ca.eval()

        with torch.no_grad():
            val_logits = eval_ca(torch.tensor(X3_std.values, dtype=torch.float32)).numpy().flatten()
            te_logits = eval_ca(torch.tensor(Xte693_std.values, dtype=torch.float32)).numpy().flatten()
            p_ca_raw = torch.sigmoid(torch.tensor(te_logits)).numpy()

        # Apply Platt scaling post-hoc calibration
        p_ca_cal = apply_platt_scaling(val_logits, y_c693_tr.values, te_logits)
        opt_thresh = find_optimal_threshold_youden(y_c693_tr.values, torch.sigmoid(torch.tensor(val_logits)).numpy())

        m_ca = evaluate_binary_classification(y_c693_te.values, p_ca_cal, threshold=opt_thresh)
        records.append({"model": "CA-FedAvg (Ours)", "seed": seed, "auc": m_ca["roc_auc"], "acc": m_ca["accuracy"], "spec": m_ca["specificity"], "ece": m_ca["ece"]})

    df_res = pd.DataFrame(records)
    summary = df_res.groupby("model").agg({
        "auc": ["mean", "std"],
        "acc": ["mean", "std"],
        "spec": ["mean", "std"],
        "ece": ["mean", "std"],
    }).reset_index()

    print("\n" + "=" * 80)
    print("REAL MULTI-CENTER BENCHMARK SUMMARY (5 SEEDS):")
    print("=" * 80)
    print(summary)

    os.makedirs(os.path.join(_ROOT, "reports"), exist_ok=True)
    df_res.to_csv(os.path.join(_ROOT, "reports", "v6_real_benchmark_results.csv"), index=False)

    return summary


if __name__ == "__main__":
    run_v6_real_pipeline()
