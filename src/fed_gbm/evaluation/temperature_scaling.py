"""
Temperature Scaling & Platt Scaling Post-Hoc Calibration Module for FED-GBM v4.

Provides post-hoc calibration methods:
1. Temperature Scaling: p_cal = sigmoid(logits / T)
2. Platt Scaling (Logistic Calibration): p_cal = sigmoid(a * logits + b)
3. Isotonic Regression Calibration: non-parametric isotonic mapping
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression


class TemperatureScaler(nn.Module):
    """Post-hoc temperature scaling model."""
    def __init__(self):
        super(TemperatureScaler, self).__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature

    def fit(self, logits: torch.Tensor, labels: torch.Tensor, lr: float = 0.05, max_iter: int = 100):
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def eval_loss():
            optimizer.zero_grad()
            loss = criterion(self.forward(logits), labels)
            loss.backward()
            return loss

        optimizer.step(eval_loss)
        with torch.no_grad():
            self.temperature.clamp_(min=0.01)

        return float(self.temperature.item())


def apply_temperature_scaling(
    val_logits: np.ndarray,
    val_labels: np.ndarray,
    test_logits: np.ndarray
) -> np.ndarray:
    """Fits TemperatureScaler on validation set and returns calibrated test set probabilities."""
    scaler = TemperatureScaler()
    t_val_logits = torch.tensor(val_logits, dtype=torch.float32).unsqueeze(1) if val_logits.ndim == 1 else torch.tensor(val_logits, dtype=torch.float32)
    t_val_labels = torch.tensor(val_labels, dtype=torch.float32).unsqueeze(1) if val_labels.ndim == 1 else torch.tensor(val_labels, dtype=torch.float32)
    t_test_logits = torch.tensor(test_logits, dtype=torch.float32).unsqueeze(1) if test_logits.ndim == 1 else torch.tensor(test_logits, dtype=torch.float32)

    scaler.fit(t_val_logits, t_val_labels)

    with torch.no_grad():
        calibrated_test_logits = scaler(t_test_logits)
        calibrated_test_probs = torch.sigmoid(calibrated_test_logits).numpy().flatten()

    return calibrated_test_probs


def apply_platt_scaling(
    val_logits: np.ndarray,
    val_labels: np.ndarray,
    test_logits: np.ndarray
) -> np.ndarray:
    """Fits Platt scaling (Logistic Regression on logits) to calibrate probabilities."""
    val_x = val_logits.reshape(-1, 1)
    test_x = test_logits.reshape(-1, 1)

    lr = LogisticRegression(C=1.0, solver="lbfgs")
    lr.fit(val_x, val_labels)

    calibrated_probs = lr.predict_proba(test_x)[:, 1]
    return calibrated_probs


def apply_isotonic_calibration(
    val_logits: np.ndarray,
    val_labels: np.ndarray,
    test_logits: np.ndarray
) -> np.ndarray:
    """Fits non-parametric Isotonic Regression to calibrate probabilities."""
    val_x = torch.sigmoid(torch.tensor(val_logits)).numpy().flatten()
    test_x = torch.sigmoid(torch.tensor(test_logits)).numpy().flatten()

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.001, y_max=0.999)
    iso.fit(val_x, val_labels)

    calibrated_probs = iso.transform(test_x)
    return calibrated_probs
