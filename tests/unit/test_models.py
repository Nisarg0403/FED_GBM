import sys, os
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

"""
Phase 2 Comprehensive Unit Tests for FED-GBM v3 Models & Centralized Baselines.
Verifies the 10 strict requirements:
1. Model input dimension versatility
2. Model output shape (batch_size, 1)
3. Logit output (no sigmoid inside model)
4. BCEWithLogitsLoss compatibility
5. Deterministic initialization
6. Centralized baseline input compatibility
7. No test-data access by training functions
8. Feature-schema consistency
9. Probability conversion
10. Metric calculation (ROC-AUC primary, no C-Index)
"""

import os
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from fed_gbm.models import (
    SurvivalMLP,
    set_seed,
    get_centralized_baselines,
    save_experiment_metadata
)
from fed_gbm.metrics import (
    logits_to_probs,
    evaluate_binary_classification,
    calculate_expected_calibration_error
)
from fed_gbm.preprocessing import (
    LeakageFreeImputer,
    VarianceFilter,
    PearsonCorrelationFilter,
    XGBoostFeatureSelector,
    FederatedSufficientStatsScaler
)


def create_mock_data(n_train=77, n_test=20, p=15, seed=42):
    """Generates synthetic TCGA-GBM train (N=77) and test (N=20) data."""
    set_seed(seed)
    feature_names = ['Age', 'Gender', 'KPS', 'Resection_Status'] + [f'rad_{i}' for i in range(p - 4)]
    
    X_tr_mat = np.random.normal(0, 1, size=(n_train, p))
    X_te_mat = np.random.normal(0.2, 1.1, size=(n_test, p))
    
    X_train = pd.DataFrame(X_tr_mat, columns=feature_names)
    X_test = pd.DataFrame(X_te_mat, columns=feature_names)
    
    y_train = pd.Series(np.random.choice([0, 1], size=n_train, p=[0.4, 0.6]), name='label_12m')
    y_test = pd.Series(np.random.choice([0, 1], size=n_test, p=[0.4, 0.6]), name='label_12m')
    
    return X_train, y_train, X_test, y_test, feature_names


def test_1_model_input_dimension():
    """1. Model input dimension: verify SurvivalMLP accepts p in {10, 15, 20, 25, 30}."""
    for p in [10, 15, 20, 25, 30]:
        model = SurvivalMLP(input_dim=p, hidden_dims=[32, 16])
        dummy_x = torch.randn(8, p)
        out = model(dummy_x)
        assert out.shape == (8, 1), f"Failed for input_dim={p}"
    print("[PASS] Test 1: Model input dimension flexibility verified.")


def test_2_model_output_shape():
    """2. Model output shape: verify output shape is (batch_size, 1)."""
    model = SurvivalMLP(input_dim=15, hidden_dims=[32, 16])
    for batch_size in [1, 16, 64, 77]:
        dummy_x = torch.randn(batch_size, 15)
        out = model(dummy_x)
        assert out.shape == (batch_size, 1), f"Failed for batch_size={batch_size}"
    print("[PASS] Test 2: Model output shape (batch_size, 1) verified.")


def test_3_logit_output_unscaled():
    """3. Logit output: verify output values are unscaled raw logits (not bounded by sigmoid)."""
    set_seed(42)
    model = SurvivalMLP(input_dim=15, hidden_dims=[64, 32], dropout_rate=0.0)
    
    # Large inputs to generate out-of-[0,1] bounds logits
    dummy_x = torch.randn(10, 15) * 10.0
    with torch.no_grad():
        out = model(dummy_x).numpy().ravel()
        
    # Verify logits are not constrained to [0, 1]
    has_outside_unit_interval = np.any((out < 0.0) | (out > 1.0))
    assert has_outside_unit_interval, f"Logits appear bounded: {out}"
    print("[PASS] Test 3: Unscaled raw logit output verified (no sigmoid inside model).")


def test_4_bce_with_logits_loss_compatibility():
    """4. BCEWithLogitsLoss compatibility: verify compatibility and gradient flow."""
    set_seed(42)
    model = SurvivalMLP(input_dim=15, hidden_dims=[32, 16])
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    dummy_x = torch.randn(16, 15)
    dummy_y = torch.randint(0, 2, (16, 1)).float()
    
    optimizer.zero_grad()
    logits = model(dummy_x)
    loss = criterion(logits, dummy_y)
    loss.backward()
    optimizer.step()
    
    assert not torch.isnan(loss), "Loss computed as NaN"
    assert loss.item() > 0.0, "Loss value invalid"
    print("[PASS] Test 4: BCEWithLogitsLoss compatibility & gradient update verified.")


def test_5_deterministic_initialization():
    """5. Deterministic initialization: verify set_seed(seed) creates identical model weights."""
    set_seed(42)
    model1 = SurvivalMLP(input_dim=15, hidden_dims=[32, 16])
    w1 = model1.network[0].weight.detach().numpy().copy()
    
    set_seed(42)
    model2 = SurvivalMLP(input_dim=15, hidden_dims=[32, 16])
    w2 = model2.network[0].weight.detach().numpy().copy()
    
    np.testing.assert_allclose(w1, w2, rtol=1e-7, atol=1e-7)
    print("[PASS] Test 5: Deterministic initialization across seeds verified.")


def test_6_centralized_baseline_input_compatibility():
    """6. Centralized baseline input compatibility: test LR, RF, XGB, PyTorch MLP on tabular inputs."""
    X_train, y_train, X_test, y_test, _ = create_mock_data(n_train=77, n_test=20, p=15, seed=42)
    
    baselines = get_centralized_baselines(seed=42, input_dim=15)
    
    # 1. Logistic Regression
    baselines['logistic_regression'].fit(X_train, y_train)
    lr_probs = baselines['logistic_regression'].predict_proba(X_test)[:, 1]
    assert len(lr_probs) == 20
    
    # 2. Random Forest
    baselines['random_forest'].fit(X_train, y_train)
    rf_probs = baselines['random_forest'].predict_proba(X_test)[:, 1]
    assert len(rf_probs) == 20
    
    # 3. XGBoost
    baselines['xgboost'].fit(X_train, y_train)
    xgb_probs = baselines['xgboost'].predict_proba(X_test)[:, 1]
    assert len(xgb_probs) == 20
    
    # 4. PyTorch MLP
    mlp = baselines['pytorch_mlp']
    optimizer = torch.optim.Adam(mlp.parameters(), lr=0.01)
    criterion = nn.BCEWithLogitsLoss()
    
    X_tr_t = torch.tensor(X_train.values, dtype=torch.float32)
    y_tr_t = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    X_te_t = torch.tensor(X_test.values, dtype=torch.float32)
    
    mlp.train()
    for _ in range(5):
        optimizer.zero_grad()
        loss = criterion(mlp(X_tr_t), y_tr_t)
        loss.backward()
        optimizer.step()
        
    mlp.eval()
    with torch.no_grad():
        mlp_logits = mlp(X_te_t).numpy().ravel()
        mlp_probs = logits_to_probs(mlp_logits)
    assert len(mlp_probs) == 20
    
    print("[PASS] Test 6: All 4 centralized baseline models fit and predict on preprocessed tabular data.")


def test_7_no_test_data_access_by_training_functions():
    """7. No test-data access by training functions: verify test partition is untouched during fit."""
    X_train, y_train, X_test, y_test, _ = create_mock_data(n_train=77, n_test=20, p=15, seed=42)
    
    # Copy original test set for strict immutability check
    X_test_orig = X_test.copy(deep=True)
    y_test_orig = y_test.copy(deep=True)
    
    # Preprocessing fit on train only
    imputer = LeakageFreeImputer().fit(X_train)
    vfilter = VarianceFilter().fit(imputer.transform(X_train))
    cfilter = PearsonCorrelationFilter().fit(vfilter.transform(imputer.transform(X_train)))
    
    # Fit model on train only
    baselines = get_centralized_baselines(seed=42, input_dim=cfilter.transform(vfilter.transform(imputer.transform(X_train))).shape[1])
    baselines['logistic_regression'].fit(cfilter.transform(vfilter.transform(imputer.transform(X_train))), y_train)
    
    # Verify test set remained strictly identical
    pd.testing.assert_frame_equal(X_test, X_test_orig)
    pd.testing.assert_series_equal(y_test, y_test_orig)
    print("[PASS] Test 7: Strict test set isolation during preprocessing & training verified.")


def test_8_feature_schema_consistency():
    """8. Feature-schema consistency: verify all baseline models operate on identical feature schemas."""
    X_train, y_train, _, _, feature_names = create_mock_data(n_train=77, n_test=20, p=15, seed=42)
    baselines = get_centralized_baselines(seed=42, input_dim=15)
    
    for name, model in baselines.items():
        if name == 'pytorch_mlp':
            assert model.input_dim == len(feature_names)
        else:
            model.fit(X_train, y_train)
            assert model.n_features_in_ == len(feature_names)
            
    print("[PASS] Test 8: Feature-schema consistency across all 4 baseline models verified.")


def test_9_probability_conversion():
    """9. Probability conversion: verify logits_to_probs maps logits to valid probabilities in (0, 1)."""
    logits = np.array([-10.0, -2.0, 0.0, 2.0, 10.0])
    probs = logits_to_probs(logits)
    
    assert np.all((probs >= 0.0) & (probs <= 1.0)), "Probabilities outside [0, 1]"
    assert probs[2] == 0.5, "Sigmoid(0) != 0.5"
    assert probs[0] < probs[1] < probs[2] < probs[3] < probs[4], "Monotonicity violated"
    print("[PASS] Test 9: Probability conversion (logits_to_probs) verified.")


def test_10_metric_calculation():
    """10. Metric calculation: verify calculation of ROC-AUC, PR-AUC, Acc, F1, Sens, Spec, Brier, ECE."""
    y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0, 1, 0])
    y_probs = np.array([0.1, 0.9, 0.2, 0.8, 0.7, 0.3, 0.65, 0.15, 0.85, 0.4])
    
    metrics = evaluate_binary_classification(y_true, y_probs)
    
    # Required keys present
    expected_keys = ['roc_auc', 'pr_auc', 'accuracy', 'f1', 'sensitivity', 'specificity', 'brier_score', 'ece']
    for k in expected_keys:
        assert k in metrics, f"Missing metric key: {k}"
        assert isinstance(metrics[k], float)
        assert not np.isnan(metrics[k])
        
    # C-Index strictly EXCLUDED
    assert 'c_index' not in metrics
    assert 'concordance_index' not in metrics
    
    print("[PASS] Test 10: Binary classification metric calculations verified (C-Index strictly excluded).")


def run_all_phase_2_tests():
    """Executes all 10 Phase 2 unit tests."""
    print("=" * 70)
    print("RUNNING FED-GBM V3 PHASE 2 UNIT TESTS")
    print("=" * 70)
    test_1_model_input_dimension()
    test_2_model_output_shape()
    test_3_logit_output_unscaled()
    test_4_bce_with_logits_loss_compatibility()
    test_5_deterministic_initialization()
    test_6_centralized_baseline_input_compatibility()
    test_7_no_test_data_access_by_training_functions()
    test_8_feature_schema_consistency()
    test_9_probability_conversion()
    test_10_metric_calculation()
    print("=" * 70)
    print("ALL 10 PHASE 2 UNIT TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == '__main__':
    run_all_phase_2_tests()
