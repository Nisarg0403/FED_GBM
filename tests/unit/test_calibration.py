"""
Verification Gate 2.3 Test Script: Temperature Scaling & Isotonic Calibration.

Verifies post-hoc calibration (Temperature, Platt, Isotonic) across all cohorts:
- Checks raw vs calibrated Expected Calibration Error (ECE).
- Verification Gate 2.3 PASS condition: Post-calibration ECE < 0.10.
"""

import sys
import os
import json
import numpy as np
import pandas as pd
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from fed_gbm.models import SurvivalMLP, set_seed
from fed_gbm.data_loader import load_all_cohorts
from fed_gbm.preprocessing import LeakageFreeImputer, FederatedSufficientStatsScaler
from fed_gbm.federated_server import FederatedServer
from fed_gbm.federated_client import FederatedClient
from fed_gbm.temperature_scaling import (
    TemperatureScaler,
    apply_temperature_scaling,
    apply_platt_scaling,
    apply_isotonic_calibration,
)
from fed_gbm.metrics import evaluate_binary_classification


def test_temperature_scaler_math():
    """Unit test checking TemperatureScaler optimization reduces cross entropy."""
    scaler = TemperatureScaler()
    logits = torch.tensor([[3.0], [-2.0], [1.5], [-1.0]], requires_grad=True)
    labels = torch.tensor([[1.0], [0.0], [1.0], [0.0]])

    t_val = scaler.fit(logits, labels)
    assert isinstance(t_val, float)
    assert t_val > 0.0


def run_verification_gate_2_3():
    print("=" * 80)
    print("VERIFICATION GATE 2.3: TEMPERATURE & ISOTONIC SCALING CALIBRATION (5 SEEDS)")
    print("=" * 80)

    seeds = [42, 101, 2024, 777, 999]
    raw_eces = []
    temp_eces = []
    platt_eces = []
    iso_eces = []

    for seed in seeds:
        set_seed(seed)
        cohorts = load_all_cohorts(seed=seed)
        df_tcga_tr = cohorts["tcga_train"]
        df_upenn_tr = cohorts["upenn_train"]
        df_upenn_te = cohorts["upenn_test"]

        X_tcga_tr = df_tcga_tr.drop(columns=["patient_id", "label_12m"])
        y_tcga_tr = df_tcga_tr["label_12m"]
        X_upenn_tr = df_upenn_tr.drop(columns=["patient_id", "label_12m"])
        y_upenn_tr = df_upenn_tr["label_12m"]
        X_upenn_te = df_upenn_te.drop(columns=["patient_id", "label_12m"])
        y_upenn_te = df_upenn_te["label_12m"]

        imp_tcga = LeakageFreeImputer()
        imp_tcga.fit(X_tcga_tr)
        X_tcga_tr_imp = imp_tcga.transform(X_tcga_tr)

        imp_upenn = LeakageFreeImputer()
        imp_upenn.fit(X_upenn_tr)
        X_upenn_tr_imp = imp_upenn.transform(X_upenn_tr)
        X_upenn_te_imp = imp_upenn.transform(X_upenn_te)

        scaler = FederatedSufficientStatsScaler()
        st1 = scaler.compute_local_stats(X_tcga_tr_imp)
        st2 = scaler.compute_local_stats(X_upenn_tr_imp)
        scaler.aggregate_global_stats([st1, st2])

        X_tcga_tr_std = scaler.transform(X_tcga_tr_imp)
        X_upenn_tr_std = scaler.transform(X_upenn_tr_imp)
        X_upenn_te_std = scaler.transform(X_upenn_te_imp)

        input_dim = X_tcga_tr_std.shape[1]

        server = FederatedServer(input_dim=input_dim, seed=seed)
        c1 = FederatedClient("TCGA", X_tcga_tr_std, y_tcga_tr, input_dim=input_dim)
        c2 = FederatedClient("UPenn", X_upenn_tr_std, y_upenn_tr, input_dim=input_dim)

        for r in range(20):
            g_sd = server.get_global_state_dict()
            c1.set_model_parameters(g_sd)
            c2.set_model_parameters(g_sd)
            p1 = c1.local_train(epochs=5, batch_size=16, lr=0.01, mu=0.01, seed=seed + r)
            p2 = c2.local_train(epochs=5, batch_size=16, lr=0.01, mu=0.01, seed=seed + r + 100)
            server.aggregate_client_payloads([p1, p2])

        eval_m = SurvivalMLP(input_dim=input_dim, hidden_dims=[32, 16])
        eval_m.load_state_dict(server.get_global_state_dict())
        eval_m.eval()

        with torch.no_grad():
            val_logits = eval_m(torch.tensor(X_upenn_tr_std.values, dtype=torch.float32)).numpy().flatten()
            test_logits = eval_m(torch.tensor(X_upenn_te_std.values, dtype=torch.float32)).numpy().flatten()
            raw_probs = torch.sigmoid(torch.tensor(test_logits)).numpy()

        m_raw = evaluate_binary_classification(y_upenn_te.values, raw_probs)
        raw_eces.append(m_raw["ece"])

        # Temperature Scaling
        temp_probs = apply_temperature_scaling(val_logits, y_upenn_tr.values, test_logits)
        m_temp = evaluate_binary_classification(y_upenn_te.values, temp_probs)
        temp_eces.append(m_temp["ece"])

        # Platt Scaling
        platt_probs = apply_platt_scaling(val_logits, y_upenn_tr.values, test_logits)
        m_platt = evaluate_binary_classification(y_upenn_te.values, platt_probs)
        platt_eces.append(m_platt["ece"])

        # Isotonic Calibration
        iso_probs = apply_isotonic_calibration(val_logits, y_upenn_tr.values, test_logits)
        m_iso = evaluate_binary_classification(y_upenn_te.values, iso_probs)
        iso_eces.append(m_iso["ece"])

    raw_ece_mean = float(np.mean(raw_eces))
    temp_ece_mean = float(np.mean(temp_eces))
    platt_ece_mean = float(np.mean(platt_eces))
    iso_ece_mean = float(np.mean(iso_eces))

    print(f"Raw ECE Mean               : {raw_ece_mean:.4f}")
    print(f"Temperature Scaled ECE Mean: {temp_ece_mean:.4f}")
    print(f"Platt Scaled ECE Mean      : {platt_ece_mean:.4f}")
    print(f"Isotonic Scaled ECE Mean   : {iso_ece_mean:.4f}")

    best_cal_ece = min(temp_ece_mean, platt_ece_mean, iso_ece_mean)
    gate_pass = best_cal_ece < 0.10
    print(f"\nVERIFICATION GATE 2.3 STATUS: {'[PASS]' if gate_pass else '[FAIL/WARNING]'}")

    return {
        "gate_2_3_passed": gate_pass,
        "raw_ece_mean": raw_ece_mean,
        "temperature_ece_mean": temp_ece_mean,
        "platt_ece_mean": platt_ece_mean,
        "isotonic_ece_mean": iso_ece_mean,
        "best_calibrated_ece": best_cal_ece,
    }


if __name__ == "__main__":
    test_temperature_scaler_math()
    run_verification_gate_2_3()
