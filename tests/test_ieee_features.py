import sys, os
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

"""
Unit tests for IEEE Journal submission enhancements:
1. CoxPHLoss (DeepSurv survival loss) correctness, gradients, and loss math.
2. partition_data_dirichlet data split integrity, client count, and non-IID properties.
"""

import pytest
import torch
import numpy as np
import pandas as pd

from fed_gbm.models import SurvivalMLP, CoxPHLoss
from fed_gbm.data_loader import partition_data_dirichlet, load_all_cohorts


def test_cox_ph_loss_basic_computation():
    """Verifies CoxPHLoss produces positive scalar loss and backward gradients."""
    loss_fn = CoxPHLoss()

    risk_scores = torch.tensor([[1.5], [0.8], [-0.2], [0.4]], requires_grad=True)
    durations = torch.tensor([[100.0], [250.0], [400.0], [150.0]])
    events = torch.tensor([[1.0], [1.0], [0.0], [1.0]])  # 3 events, 1 censored

    loss = loss_fn(risk_scores, durations, events)

    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0
    assert torch.isfinite(loss)
    assert loss.item() > 0.0

    loss.backward()
    assert risk_scores.grad is not None
    assert risk_scores.grad.shape == risk_scores.shape


def test_cox_ph_loss_zero_events():
    """Verifies CoxPHLoss handles zero events (all censored) without crash."""
    loss_fn = CoxPHLoss()
    risk_scores = torch.tensor([[1.0], [2.0]], requires_grad=True)
    durations = torch.tensor([[100.0], [200.0]])
    events = torch.tensor([[0.0], [0.0]])

    loss = loss_fn(risk_scores, durations, events)
    assert loss.item() == 0.0


def test_partition_data_dirichlet_integrity():
    """Verifies partition_data_dirichlet preserves total row count and patient IDs."""
    cohorts = load_all_cohorts(seed=42)
    df_train = cohorts["tcga_train"]

    n_clients = 3
    alpha = 0.5
    client_dfs = partition_data_dirichlet(df_train, n_clients=n_clients, alpha=alpha, seed=42)

    assert len(client_dfs) == n_clients
    total_partitioned_rows = sum(len(cdf) for cdf in client_dfs)
    assert total_partitioned_rows == len(df_train)

    all_partitioned_ids = []
    for cdf in client_dfs:
        all_partitioned_ids.extend(cdf["patient_id"].tolist())

    assert sorted(all_partitioned_ids) == sorted(df_train["patient_id"].tolist())


def test_partition_data_dirichlet_reproducibility():
    """Verifies partition_data_dirichlet produces deterministic splits with fixed seed."""
    cohorts = load_all_cohorts(seed=42)
    df = cohorts["tcga_train"]

    split_a = partition_data_dirichlet(df, n_clients=3, alpha=0.5, seed=123)
    split_b = partition_data_dirichlet(df, n_clients=3, alpha=0.5, seed=123)

    for c_a, c_b in zip(split_a, split_b):
        assert c_a["patient_id"].tolist() == c_b["patient_id"].tolist()
