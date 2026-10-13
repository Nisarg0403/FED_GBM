"""
FED-GBM v3 Experiment B — CGGA External Validation.

Trains a locked model on combined TCGA+UPenn data (N=537 total)
and evaluates on the CGGA cohort (N=249) without any re-training.

This is the critical external validation experiment required for publication-grade
federated learning papers (matches what Nature Comms 2022 and BMC 2024 do).

Run from project root:
    python -m src.fed_gbm.cgga_experiment_b
"""

import copy
import json
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from fed_gbm.data_loader import load_all_cohorts, FEATURE_COLS
from fed_gbm.models import SurvivalMLP, set_seed
from fed_gbm.metrics import evaluate_binary_classification, logits_to_probs
from fed_gbm.preprocessing import FederatedSufficientStatsScaler
from fed_gbm.federated_client import FederatedClient
from fed_gbm.federated_server import FederatedServer
from fed_gbm.survival_curves import generate_km_curves
from fed_gbm.explainability import run_shap_analysis

RESULTS_DIR = os.path.join("results", "archive", "experiment_b")
SHAP_DIR = os.path.join(RESULTS_DIR, "shap")
KM_DIR = os.path.join(RESULTS_DIR, "km_curves")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(SHAP_DIR, exist_ok=True)
os.makedirs(KM_DIR, exist_ok=True)

SEEDS = [42, 123, 2024, 3407, 9999]
FL_ROUNDS = 30
LOCAL_EPOCHS = 5
HIDDEN_DIMS = [128, 64]
LR = 0.001
WEIGHT_DECAY = 1e-4
DROPOUT = 0.3


def df_to_tensor(df: pd.DataFrame, feature_cols: list) -> torch.Tensor:
    return torch.tensor(df[feature_cols].values, dtype=torch.float32)


def labels_to_tensor(series) -> torch.Tensor:
    return torch.tensor(series.values, dtype=torch.float32).unsqueeze(1)


def train_fedavg_model(
    X_tcga_tr: np.ndarray,
    y_tcga_tr: np.ndarray,
    X_upenn_tr: np.ndarray,
    y_upenn_tr: np.ndarray,
    input_dim: int,
    seed: int,
    pos_weight: float = 1.0,
) -> SurvivalMLP:
    """Trains FedAvg MLP across TCGA + UPenn for FL_ROUNDS."""
    set_seed(seed)

    pw = torch.tensor([pos_weight])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

    global_model = SurvivalMLP(input_dim=input_dim, hidden_dims=HIDDEN_DIMS, dropout_rate=DROPOUT)
    global_state = copy.deepcopy(global_model.state_dict())

    n_total = len(X_tcga_tr) + len(X_upenn_tr)

    for round_idx in range(FL_ROUNDS):
        client_states = []
        client_weights = []

        for X_local, y_local in [(X_tcga_tr, y_tcga_tr), (X_upenn_tr, y_upenn_tr)]:
            local_model = SurvivalMLP(input_dim=input_dim, hidden_dims=HIDDEN_DIMS, dropout_rate=DROPOUT)
            local_model.load_state_dict(copy.deepcopy(global_state))
            local_model.train()

            X_t = torch.tensor(X_local, dtype=torch.float32)
            y_t = torch.tensor(y_local, dtype=torch.float32).unsqueeze(1)
            dataset = TensorDataset(X_t, y_t)
            loader = DataLoader(dataset, batch_size=16, shuffle=True)

            optimizer = torch.optim.Adam(local_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=LOCAL_EPOCHS)

            for _ in range(LOCAL_EPOCHS):
                for xb, yb in loader:
                    optimizer.zero_grad()
                    loss = criterion(local_model(xb), yb)
                    loss.backward()
                    nn.utils.clip_grad_norm_(local_model.parameters(), max_norm=1.0)
                    optimizer.step()
                scheduler.step()

            client_states.append(copy.deepcopy(local_model.state_dict()))
            client_weights.append(len(X_local) / n_total)

        # Weighted FedAvg aggregation
        new_state = {}
        for key in global_state.keys():
            new_state[key] = sum(
                w * st[key].float() for w, st in zip(client_weights, client_states)
            )
        global_state = new_state

    global_model.load_state_dict(global_state)
    global_model.eval()
    return global_model


def run_experiment_b():
    """Executes CGGA external validation across all seeds."""
    print("=" * 70)
    print("FED-GBM V3 — EXPERIMENT B: CGGA EXTERNAL VALIDATION")
    print(f"Seeds: {SEEDS} | FL Rounds: {FL_ROUNDS} | Hidden: {HIDDEN_DIMS}")
    print("=" * 70)

    all_records = []

    for seed in SEEDS:
        print(f"\n[SEED {seed}]")
        set_seed(seed)

        # Load all cohorts
        cohorts = load_all_cohorts(seed=seed)
        tcga_train = cohorts["tcga_train"]
        tcga_test = cohorts["tcga_test"]
        upenn_train = cohorts["upenn_train"]
        upenn_test = cohorts["upenn_test"]
        cgga_val = cohorts["cgga_val"]
        feature_cols = cohorts["feature_cols"]

        X_tcga_tr = tcga_train[feature_cols].values.astype(np.float32)
        y_tcga_tr = tcga_train["label_12m"].values.astype(np.float32)
        X_upenn_tr = upenn_train[feature_cols].values.astype(np.float32)
        y_upenn_tr = upenn_train["label_12m"].values.astype(np.float32)

        X_tcga_te = tcga_test[feature_cols].values.astype(np.float32)
        y_tcga_te = tcga_test["label_12m"].values.astype(np.float32)
        X_upenn_te = upenn_test[feature_cols].values.astype(np.float32)
        y_upenn_te = upenn_test["label_12m"].values.astype(np.float32)
        X_cgga = cgga_val[feature_cols].values.astype(np.float32)
        y_cgga = cgga_val["label_12m"].values.astype(np.float32)

        # Federated StandardScaler (fit on train, apply to all)
        scaler = FederatedSufficientStatsScaler()
        stats1 = scaler.compute_local_stats(pd.DataFrame(X_tcga_tr, columns=feature_cols))
        stats2 = scaler.compute_local_stats(pd.DataFrame(X_upenn_tr, columns=feature_cols))
        scaler.aggregate_global_stats([stats1, stats2])

        X_tcga_tr_s = scaler.transform(pd.DataFrame(X_tcga_tr, columns=feature_cols))
        X_upenn_tr_s = scaler.transform(pd.DataFrame(X_upenn_tr, columns=feature_cols))
        X_tcga_te_s = scaler.transform(pd.DataFrame(X_tcga_te, columns=feature_cols))
        X_upenn_te_s = scaler.transform(pd.DataFrame(X_upenn_te, columns=feature_cols))
        X_cgga_s = scaler.transform(pd.DataFrame(X_cgga, columns=feature_cols))

        # Compute class imbalance weight
        pos_rate = y_tcga_tr.mean()
        pos_weight = (1.0 - pos_rate) / max(pos_rate, 1e-6)

        input_dim = len(feature_cols)

        # Train FedAvg model
        model = train_fedavg_model(
            X_tcga_tr_s.values, y_tcga_tr,
            X_upenn_tr_s.values, y_upenn_tr,
            input_dim=input_dim,
            seed=seed,
            pos_weight=pos_weight,
        )

        # Evaluate on all three cohorts
        for (X_eval, y_eval, cname, df_ref) in [
            (X_tcga_te_s.values, y_tcga_te, "TCGA_test", tcga_test),
            (X_upenn_te_s.values, y_upenn_te, "UPenn_external_test", upenn_test),
            (X_cgga_s.values, y_cgga, "CGGA_external_val", cgga_val),
        ]:
            with torch.no_grad():
                logits = model(torch.tensor(X_eval, dtype=torch.float32)).squeeze().numpy()
            probs = logits_to_probs(logits)
            metrics = evaluate_binary_classification(y_eval, probs)

            record = {
                "seed": seed,
                "cohort": cname,
                "roc_auc": metrics["roc_auc"],
                "pr_auc": metrics["pr_auc"],
                "accuracy": metrics["accuracy"],
                "f1": metrics["f1"],
                "sensitivity": metrics["sensitivity"],
                "specificity": metrics["specificity"],
                "brier_score": metrics["brier_score"],
                "ece": metrics["ece"],
            }
            all_records.append(record)
            print(f"  [{cname}] AUC={metrics['roc_auc']:.4f} | F1={metrics['f1']:.4f} | Brier={metrics['brier_score']:.4f}")

            # KM curve
            generate_km_curves(
                patient_ids=df_ref["patient_id"].values if "patient_id" in df_ref.columns else np.arange(len(y_eval)),
                y_true=y_eval,
                y_probs=probs,
                days_to_event=None,
                model_name=f"FedAvg_ExptB_seed{seed}",
                cohort_name=cname,
                output_dir=KM_DIR,
            )

    # Aggregate across seeds
    df_records = pd.DataFrame(all_records)
    df_records.to_csv(os.path.join(RESULTS_DIR, "experiment_b_per_seed.csv"), index=False)

    agg = df_records.groupby("cohort").agg(
        roc_auc_mean=("roc_auc", "mean"),
        roc_auc_std=("roc_auc", "std"),
        pr_auc_mean=("pr_auc", "mean"),
        accuracy_mean=("accuracy", "mean"),
        f1_mean=("f1", "mean"),
        brier_mean=("brier_score", "mean"),
    ).reset_index()

    agg.to_csv(os.path.join(RESULTS_DIR, "experiment_b_aggregate.csv"), index=False)

    # Run SHAP on best seed (seed=42) on TCGA test
    print("\n[SHAP] Running explainability on TCGA test (seed=42)...")
    set_seed(42)
    cohorts = load_all_cohorts(seed=42)
    feat_cols = cohorts["feature_cols"]
    tcga_train = cohorts["tcga_train"]
    upenn_train = cohorts["upenn_train"]
    tcga_test = cohorts["tcga_test"]

    scaler_shap = FederatedSufficientStatsScaler()
    s1 = scaler_shap.compute_local_stats(tcga_train[feat_cols])
    s2 = scaler_shap.compute_local_stats(upenn_train[feat_cols])
    scaler_shap.aggregate_global_stats([s1, s2])

    X_tr_s = scaler_shap.transform(tcga_train[feat_cols]).values.astype(np.float32)
    X_te_s = scaler_shap.transform(tcga_test[feat_cols]).values.astype(np.float32)
    y_tr = tcga_train["label_12m"].values.astype(np.float32)
    y_te = tcga_test["label_12m"].values.astype(np.float32)
    pos_wt = (1.0 - y_tr.mean()) / max(y_tr.mean(), 1e-6)

    X_upenn_s = scaler_shap.transform(upenn_train[feat_cols]).values.astype(np.float32)
    y_upenn_vals = upenn_train["label_12m"].values.astype(np.float32)

    model_shap = train_fedavg_model(X_tr_s, y_tr, X_upenn_s, y_upenn_vals, input_dim=len(feat_cols), seed=42, pos_weight=pos_wt)

    with torch.no_grad():
        te_logits = model_shap(torch.tensor(X_te_s, dtype=torch.float32)).squeeze().numpy()
    te_probs = logits_to_probs(te_logits)

    try:
        shap_results = run_shap_analysis(
            model=model_shap,
            X_train=X_tr_s,
            X_test=X_te_s,
            y_true=y_te,
            y_probs=te_probs,
            feature_names=feat_cols,
            patient_ids=tcga_test["patient_id"].tolist() if "patient_id" in tcga_test.columns else [f"P{i}" for i in range(len(y_te))],
            model_name="FedAvg_ExptB",
            cohort_name="TCGA_test",
            output_dir=SHAP_DIR,
        )
        print(f"  SHAP summary saved: {shap_results['summary_bar']}")
        print("  Top features:")
        print(shap_results["importance_df"].head(5).to_string(index=False))
    except Exception as e:
        print(f"  SHAP analysis failed (non-critical): {e}")

    print("\n" + "=" * 70)
    print("EXPERIMENT B AGGREGATE RESULTS:")
    print(agg.to_string(index=False))
    print("=" * 70)
    print(f"Results saved to: {RESULTS_DIR}")

    return agg


if __name__ == "__main__":
    run_experiment_b()
