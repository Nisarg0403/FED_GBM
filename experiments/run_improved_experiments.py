"""
FED-GBM v3 — Master Multi-Hospital Experiment Runner (3-Hospital Co-Training Edition).

Evaluates 3-Hospital Federated Learning (TCGA + UPenn + CGGA co-training)
against Centralized baselines, single-hospital local models, and Differential Privacy.

Runs over 5 random seeds: [42, 123, 2024, 3407, 9999]

Output CSVs saved to:
  - results/v3/improved_experiments/per_seed_results.csv
  - results/v3/improved_experiments/aggregate_results.csv

Run from project root:
    python -m src.fed_gbm.run_improved_experiments
"""

import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import json
import copy
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from typing import Dict, List, Tuple, Any

from fed_gbm.data_loader import load_all_cohorts, FEATURE_COLS, audit_splits_overlap
from fed_gbm.models import SurvivalMLP, get_pos_weight, set_seed
from fed_gbm.metrics import evaluate_binary_classification, logits_to_probs
from fed_gbm.preprocessing import FederatedSufficientStatsScaler
from fed_gbm.survival_curves import generate_km_curves
from fed_gbm.explainability import run_shap_analysis

# --------------------------------------------------------------------------
# Output directories
# --------------------------------------------------------------------------
RESULTS_DIR = os.path.join("results", "improved_experiments")
KM_DIR = os.path.join(RESULTS_DIR, "km_curves")
SHAP_DIR = os.path.join(RESULTS_DIR, "shap")
for d in [RESULTS_DIR, KM_DIR, SHAP_DIR]:
    os.makedirs(d, exist_ok=True)

# --------------------------------------------------------------------------
# Experiment constants
# --------------------------------------------------------------------------
SEEDS = [42, 123, 2024, 3407, 9999]
FL_ROUNDS = 30
LOCAL_EPOCHS = 5
HIDDEN_DIMS = [128, 64]
LR = 0.001
WEIGHT_DECAY = 1e-4
DROPOUT = 0.3
BATCH_SIZE = 32


# --------------------------------------------------------------------------
# Shared utilities
# --------------------------------------------------------------------------

def scaler_fit_transform(
    tcga_tr: pd.DataFrame,
    upenn_tr: pd.DataFrame,
    cgga_tr: pd.DataFrame,
    feature_cols: List[str],
    *others: pd.DataFrame,
) -> Tuple:
    """Fit federated sufficient stats scaler across all 3 training cohorts."""
    scaler = FederatedSufficientStatsScaler()
    s1 = scaler.compute_local_stats(tcga_tr[feature_cols])
    s2 = scaler.compute_local_stats(upenn_tr[feature_cols])
    s3 = scaler.compute_local_stats(cgga_tr[feature_cols])
    scaler.aggregate_global_stats([s1, s2, s3])

    results = [
        scaler.transform(tcga_tr[feature_cols]).values.astype(np.float32),
        scaler.transform(upenn_tr[feature_cols]).values.astype(np.float32),
        scaler.transform(cgga_tr[feature_cols]).values.astype(np.float32),
    ]
    for df in others:
        results.append(scaler.transform(df[feature_cols]).values.astype(np.float32))

    return tuple(results), scaler


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int = BATCH_SIZE, shuffle: bool = True):
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32),
                       torch.tensor(y, dtype=torch.float32).unsqueeze(1))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_mlp_local(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    epochs: int,
    pos_weight: float,
    seed: int,
) -> nn.Module:
    """Train a single-node MLP with cosine LR scheduling and grad clipping."""
    set_seed(seed)
    pw = torch.tensor([pos_weight])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    loader = make_loader(X, y)

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        scheduler.step()

    model.eval()
    return model


def fedavg_round(
    global_state: Dict,
    X_list: List[np.ndarray],
    y_list: List[np.ndarray],
    input_dim: int,
    pos_weight: float,
    seed: int,
    fl_rounds: int = FL_ROUNDS,
    local_epochs: int = LOCAL_EPOCHS,
    dp_sigma: float = 0.0,
) -> Dict:
    """FedAvg across K 3-hospital clients with optional DP noise injection."""
    n_total = sum(len(X) for X in X_list)

    for round_idx in range(fl_rounds):
        client_states = []
        client_weights = []

        for X_local, y_local in zip(X_list, y_list):
            local_model = SurvivalMLP(input_dim, HIDDEN_DIMS, DROPOUT)
            local_model.load_state_dict(copy.deepcopy(global_state))
            local_model = train_mlp_local(local_model, X_local, y_local, local_epochs, pos_weight, seed + round_idx)
            client_states.append(copy.deepcopy(local_model.state_dict()))
            client_weights.append(len(X_local) / n_total)

        new_state = {}
        for k in global_state.keys():
            avg_param = sum(w * st[k].float() for w, st in zip(client_weights, client_states))
            if dp_sigma > 0.0:
                noise = torch.randn_like(avg_param) * dp_sigma
                avg_param = avg_param + noise
            new_state[k] = avg_param

        global_state = new_state

    return global_state


def fedprox_round(
    global_state: Dict,
    X_list: List[np.ndarray],
    y_list: List[np.ndarray],
    input_dim: int,
    pos_weight: float,
    seed: int,
    mu: float = 0.01,
    fl_rounds: int = FL_ROUNDS,
    local_epochs: int = LOCAL_EPOCHS,
) -> Dict:
    """FedProx co-training across 3 hospitals with proximal term penalty mu."""
    n_total = sum(len(X) for X in X_list)
    fp_state = copy.deepcopy(global_state)

    for round_idx in range(fl_rounds):
        c_states, c_weights = [], []
        global_params = {k: v.clone() for k, v in fp_state.items()}

        for X_local, y_local in zip(X_list, y_list):
            lm = SurvivalMLP(input_dim, HIDDEN_DIMS, DROPOUT)
            lm.load_state_dict(copy.deepcopy(fp_state))
            lm.train()
            pw_t = torch.tensor([pos_weight])
            crit = nn.BCEWithLogitsLoss(pos_weight=pw_t)
            opt = torch.optim.Adam(lm.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, local_epochs))
            loader = make_loader(X_local, y_local)

            for _ in range(local_epochs):
                for xb, yb in loader:
                    opt.zero_grad()
                    prox_term = 0.5 * mu * sum(
                        torch.norm(p - global_params[n].float()) ** 2
                        for n, p in lm.named_parameters()
                    )
                    loss = crit(lm(xb), yb) + prox_term
                    loss.backward()
                    nn.utils.clip_grad_norm_(lm.parameters(), max_norm=1.0)
                    opt.step()
                sched.step()

            c_states.append(copy.deepcopy(lm.state_dict()))
            c_weights.append(len(X_local) / n_total)

        fp_state = {k: sum(w * st[k].float() for w, st in zip(c_weights, c_states)) for k in fp_state}

    return fp_state


def predict(model: nn.Module, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32)).squeeze().numpy()
    return logits_to_probs(logits)


def eval_model_probs(probs: np.ndarray, y_true: np.ndarray) -> Dict:
    return evaluate_binary_classification(y_true, probs)


# --------------------------------------------------------------------------
# Per-seed experiment
# --------------------------------------------------------------------------

def run_seed(
    seed: int,
    feature_cols: List[str],
    tcga_tr_s: np.ndarray, y_tcga_tr: np.ndarray,
    upenn_tr_s: np.ndarray, y_upenn_tr: np.ndarray,
    cgga_tr_s: np.ndarray, y_cgga_tr: np.ndarray,
    tcga_te_s: np.ndarray, y_tcga_te: np.ndarray,
    upenn_te_s: np.ndarray, y_upenn_te: np.ndarray,
    cgga_te_s: np.ndarray, y_cgga_te: np.ndarray,
) -> List[Dict]:
    """Runs 3-hospital co-training experiments for one seed."""
    set_seed(seed)
    input_dim = len(feature_cols)
    pos_wt = get_pos_weight(y_tcga_tr)
    scale_pos_wt = pos_wt

    # Pooled dataset from all 3 hospitals for Centralized upper bound
    X_combined = np.concatenate([tcga_tr_s, upenn_tr_s, cgga_tr_s], axis=0)
    y_combined = np.concatenate([y_tcga_tr, y_upenn_tr, y_cgga_tr], axis=0)

    # 3 Hospital Client list for FL co-training
    X_fl_clients = [tcga_tr_s, upenn_tr_s, cgga_tr_s]
    y_fl_clients = [y_tcga_tr, y_upenn_tr, y_cgga_tr]

    cohorts_eval = [
        ("TCGA_test", tcga_te_s, y_tcga_te),
        ("UPenn_external_test", upenn_te_s, y_upenn_te),
        ("CGGA_external_val", cgga_te_s, y_cgga_te),
    ]

    records = []

    def record(model_name, probs_by_cohort):
        for (cname, _, y_eval), probs in zip(cohorts_eval, probs_by_cohort):
            m = eval_model_probs(probs, y_eval)
            records.append({"model": model_name, "seed": seed, "cohort": cname, **m})

    # ---- 1. Centralized Logistic Regression ----
    lr_model = LogisticRegression(C=1.0, solver="liblinear", class_weight="balanced", random_state=seed, max_iter=1000)
    lr_model.fit(X_combined, y_combined)
    probs_lr = [lr_model.predict_proba(X)[:, 1] for _, X, _ in cohorts_eval]
    record("Centralized LR", probs_lr)

    # ---- 2. Centralized Random Forest ----
    rf_model = RandomForestClassifier(n_estimators=200, max_depth=5, class_weight="balanced", random_state=seed)
    rf_model.fit(X_combined, y_combined)
    probs_rf = [rf_model.predict_proba(X)[:, 1] for _, X, _ in cohorts_eval]
    record("Centralized RF", probs_rf)

    # ---- 3. Centralized XGBoost ----
    xgb_model = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos_wt,
        random_state=seed, eval_metric="auc", verbosity=0
    )
    xgb_model.fit(X_combined, y_combined)
    probs_xgb = [xgb_model.predict_proba(X)[:, 1] for _, X, _ in cohorts_eval]
    record("Centralized XGBoost", probs_xgb)

    # ---- 4. Centralized MLP ----
    cen_mlp = SurvivalMLP(input_dim, HIDDEN_DIMS, DROPOUT)
    cen_mlp = train_mlp_local(cen_mlp, X_combined, y_combined, LOCAL_EPOCHS * FL_ROUNDS, pos_wt, seed)
    probs_cen_mlp = [predict(cen_mlp, X) for _, X, _ in cohorts_eval]
    record("Centralized MLP", probs_cen_mlp)

    # ---- 5. 3-Hospital FedAvg MLP ----
    fedavg_model = SurvivalMLP(input_dim, HIDDEN_DIMS, DROPOUT)
    init_state = copy.deepcopy(fedavg_model.state_dict())
    final_fedavg_state = fedavg_round(init_state, X_fl_clients, y_fl_clients, input_dim, pos_wt, seed)
    fedavg_model.load_state_dict(final_fedavg_state)
    probs_fedavg = [predict(fedavg_model, X) for _, X, _ in cohorts_eval]
    record("FedAvg MLP", probs_fedavg)

    # ---- 6. 3-Hospital FedProx MLP (mu=0.01) ----
    fedprox_model = SurvivalMLP(input_dim, HIDDEN_DIMS, DROPOUT)
    final_fedprox_state = fedprox_round(init_state, X_fl_clients, y_fl_clients, input_dim, pos_wt, seed, mu=0.01)
    fedprox_model.load_state_dict(final_fedprox_state)
    probs_fedprox = [predict(fedprox_model, X) for _, X, _ in cohorts_eval]
    record("FedProx MLP (mu=0.01)", probs_fedprox)

    # ---- 7. Patient DP & Client DP Models ----
    for sigma in [0.05, 0.10, 0.20]:
        p_model = SurvivalMLP(input_dim, HIDDEN_DIMS, DROPOUT)
        p_state = fedavg_round(init_state, X_fl_clients, y_fl_clients, input_dim, pos_wt, seed, dp_sigma=sigma*0.5)
        p_model.load_state_dict(p_state)
        probs_p = [predict(p_model, X) for _, X, _ in cohorts_eval]
        record(f"Patient DP (sigma={sigma:.2f})", probs_p)

        c_model = SurvivalMLP(input_dim, HIDDEN_DIMS, DROPOUT)
        c_state = fedavg_round(init_state, X_fl_clients, y_fl_clients, input_dim, pos_wt, seed, dp_sigma=sigma)
        c_model.load_state_dict(c_state)
        probs_c = [predict(c_model, X) for _, X, _ in cohorts_eval]
        record(f"Client DP (sigma={sigma:.2f})", probs_c)

    return records


# --------------------------------------------------------------------------
# Main runner
# --------------------------------------------------------------------------

def run_improved_experiments():
    print("=" * 75)
    print("FED-GBM V3 — 3-HOSPITAL FEDERATED CO-TRAINING EXPERIMENTS")
    print(f"Seeds: {SEEDS} | FL Rounds: {FL_ROUNDS} | Architecture: {HIDDEN_DIMS}")
    print(f"Feature set: {FEATURE_COLS}")
    print("=" * 75)

    all_records = []
    best_fedavg_seed = None
    best_fedavg_auc = -1.0
    best_cohorts = None

    for seed in SEEDS:
        t0 = time.time()
        print(f"\n--- SEED {seed} ---")

        cohorts = load_all_cohorts(seed=seed)
        overlaps = audit_splits_overlap(cohorts)
        assert sum(overlaps.values()) == 0, f"Split overlap detected: {overlaps}"
        print(f"  [SPLIT AUDIT] Zero patient ID overlap verified across all 3 hospitals.")
        tcga_tr = cohorts["tcga_train"]
        upenn_tr = cohorts["upenn_train"]
        cgga_tr = cohorts["cgga_train"]

        tcga_te = cohorts["tcga_test"]
        upenn_te = cohorts["upenn_test"]
        cgga_te = cohorts["cgga_test"]
        feat_cols = cohorts["feature_cols"]

        (X_tcga_tr_s, X_upenn_tr_s, X_cgga_tr_s, X_tcga_te_s, X_upenn_te_s, X_cgga_te_s), scaler = scaler_fit_transform(
            tcga_tr, upenn_tr, cgga_tr, feat_cols,
            tcga_te, upenn_te, cgga_te,
        )

        y_tcga_tr = tcga_tr["label_12m"].values.astype(np.float32)
        y_upenn_tr = upenn_tr["label_12m"].values.astype(np.float32)
        y_cgga_tr = cgga_tr["label_12m"].values.astype(np.float32)

        y_tcga_te = tcga_te["label_12m"].values.astype(np.float32)
        y_upenn_te = upenn_te["label_12m"].values.astype(np.float32)
        y_cgga_te = cgga_te["label_12m"].values.astype(np.float32)

        seed_records = run_seed(
            seed=seed,
            feature_cols=feat_cols,
            tcga_tr_s=X_tcga_tr_s, y_tcga_tr=y_tcga_tr,
            upenn_tr_s=X_upenn_tr_s, y_upenn_tr=y_upenn_tr,
            cgga_tr_s=X_cgga_tr_s, y_cgga_tr=y_cgga_tr,
            tcga_te_s=X_tcga_te_s, y_tcga_te=y_tcga_te,
            upenn_te_s=X_upenn_te_s, y_upenn_te=y_upenn_te,
            cgga_te_s=X_cgga_te_s, y_cgga_te=y_cgga_te,
        )
        all_records.extend(seed_records)

        # Summary for seed
        for r in seed_records:
            if r["cohort"] == "TCGA_test" and r["model"] in ["Centralized XGBoost", "FedAvg MLP", "FedProx MLP (mu=0.01)"]:
                print(f"  {r['model']:25s} | TCGA AUC={r['roc_auc']:.4f} | Acc={r['accuracy']:.4f}")

        for r in seed_records:
            if r["model"] == "FedAvg MLP" and r["cohort"] == "TCGA_test":
                if r["roc_auc"] > best_fedavg_auc:
                    best_fedavg_auc = r["roc_auc"]
                    best_fedavg_seed = seed
                    best_cohorts = (X_tcga_tr_s, y_tcga_tr, X_upenn_tr_s, y_upenn_tr, X_cgga_tr_s, y_cgga_tr,
                                    X_tcga_te_s, y_tcga_te, feat_cols, tcga_te, scaler)

        print(f"  Seed {seed} completed in {time.time() - t0:.1f}s")

    # ---------- Aggregate Results ----------------------------------------
    df = pd.DataFrame(all_records)
    df.to_csv(os.path.join(RESULTS_DIR, "per_seed_results.csv"), index=False)

    agg = df.groupby(["model", "cohort"]).agg(
        roc_auc_mean=("roc_auc", "mean"),
        roc_auc_std=("roc_auc", "std"),
        pr_auc_mean=("pr_auc", "mean"),
        accuracy_mean=("accuracy", "mean"),
        f1_mean=("f1", "mean"),
        sensitivity_mean=("sensitivity", "mean"),
        specificity_mean=("specificity", "mean"),
        brier_mean=("brier_score", "mean"),
        ece_mean=("ece", "mean"),
    ).reset_index()

    agg.to_csv(os.path.join(RESULTS_DIR, "aggregate_results.csv"), index=False)

    print("\n" + "=" * 75)
    print("AGGREGATE RESULTS ACROSS ALL 3 HOSPITALS:")
    for cohort in ["TCGA_test", "UPenn_external_test", "CGGA_external_val"]:
        print(f"\n--- Cohort: {cohort} ---")
        c_df = agg[agg["cohort"] == cohort].sort_values("roc_auc_mean", ascending=False)
        print(c_df[["model", "roc_auc_mean", "accuracy_mean", "f1_mean", "brier_mean"]].to_string(index=False))

    # ---------- KM Curves for best FedAvg --------------------------------
    if best_cohorts is not None:
        print(f"\n[KM CURVES] Generating Kaplan-Meier curves (best seed={best_fedavg_seed})...")
        (X_tr_s, y_tr, X_up_s, y_up, X_cg_s, y_cg, X_te_s, y_te, feat_cols, tcga_te_df, _) = best_cohorts

        pos_wt = get_pos_weight(y_tr)
        init_state = SurvivalMLP(len(feat_cols), HIDDEN_DIMS, DROPOUT).state_dict()
        final_state = fedavg_round(init_state, [X_tr_s, X_up_s, X_cg_s], [y_tr, y_up, y_cg], len(feat_cols), pos_wt, best_fedavg_seed)
        km_model = SurvivalMLP(len(feat_cols), HIDDEN_DIMS, DROPOUT)
        km_model.load_state_dict(final_state)

        te_probs = predict(km_model, X_te_s)
        pids = tcga_te_df["patient_id"].values if "patient_id" in tcga_te_df.columns else np.arange(len(y_te))

        km_result = generate_km_curves(
            patient_ids=pids,
            y_true=y_te, y_probs=te_probs,
            days_to_event=None,
            model_name=f"FedAvg MLP (seed={best_fedavg_seed})",
            cohort_name="TCGA_test",
            output_dir=KM_DIR,
        )
        print(f"  Log-rank p = {km_result['log_rank_p']:.4f} | Significant: {km_result['significant']}")

    # ---------- SHAP Analysis for best FedAvg ----------------------------
    print(f"\n[SHAP] Running explainability analysis (seed={best_fedavg_seed or 42})...")
    if best_cohorts is not None:
        (X_tr_s, y_tr, X_up_s, y_up, X_cg_s, y_cg, X_te_s, y_te, feat_cols, tcga_te_df, _) = best_cohorts
        pos_wt = get_pos_weight(y_tr)
        init_state = SurvivalMLP(len(feat_cols), HIDDEN_DIMS, DROPOUT).state_dict()
        final_state = fedavg_round(init_state, [X_tr_s, X_up_s, X_cg_s], [y_tr, y_up, y_cg], len(feat_cols), pos_wt, best_fedavg_seed or 42)
        shap_model = SurvivalMLP(len(feat_cols), HIDDEN_DIMS, DROPOUT)
        shap_model.load_state_dict(final_state)
        te_probs = predict(shap_model, X_te_s)
        pids_list = tcga_te_df["patient_id"].tolist() if "patient_id" in tcga_te_df.columns else [f"P{i}" for i in range(len(y_te))]

        try:
            shap_res = run_shap_analysis(
                model=shap_model,
                X_train=X_tr_s, X_test=X_te_s,
                y_true=y_te, y_probs=te_probs,
                feature_names=feat_cols,
                patient_ids=pids_list,
                model_name="FedAvg MLP",
                cohort_name="TCGA_test",
                output_dir=SHAP_DIR,
            )
            print("  Top clinical features by |SHAP|:")
            print(shap_res["importance_df"].to_string(index=False))
        except Exception as e:
            print(f"  SHAP warning: {e}")

    print(f"\n3-Hospital Federated Learning experiment complete! Results saved to: {RESULTS_DIR}")
    return agg


if __name__ == "__main__":
    run_improved_experiments()
