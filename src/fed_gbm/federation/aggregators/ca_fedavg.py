"""
CA-FedAvg: Calibration-Aware Federated Averaging Module.

Novel federated aggregation method that weights client state_dict updates based on
both local sample size (N_k) and local validation Expected Calibration Error (ECE_k):

w_{t+1} = sum_k [ ( N_k * exp(-lambda * ECE_k) / Z ) * w_k ]
where Z = sum_j N_j * exp(-lambda * ECE_j) and lambda in [0.5, 1.0, 2.0].

Clients with lower calibration error receive higher aggregation weight,
improving overall global model calibration and accuracy under client drift.
"""

import copy
import numpy as np
import torch
from typing import List, Dict, Any


def aggregate_ca_fedavg(
    payloads: List[Dict[str, Any]],
    client_eces: List[float],
    lambda_param: float = 1.0
) -> Dict[str, torch.Tensor]:
    """
    Aggregates client model state_dicts using Calibration-Aware weighting.

    Args:
        payloads: List of client payloads containing 'state_dict' and 'n_samples'
        client_eces: List of float ECE values corresponding to each client
        lambda_param: Scaling hyperparameter for calibration weighting (default = 1.0)

    Returns:
        Aggregated global model state_dict
    """
    assert len(payloads) > 0, "Cannot aggregate empty payload list."
    assert len(payloads) == len(client_eces), "Mismatch between payloads and client ECEs."

    # Compute unnormalized weights: N_k * exp(-lambda * ECE_k)
    raw_weights = []
    for p, ece in zip(payloads, client_eces):
        n_k = float(p["n_samples"])
        raw_w = n_k * np.exp(-lambda_param * float(ece))
        raw_weights.append(raw_w)

    total_weight = sum(raw_weights)
    if total_weight <= 0.0:
        normalized_weights = [1.0 / len(payloads)] * len(payloads)
    else:
        normalized_weights = [rw / total_weight for rw in raw_weights]

    first_sd = payloads[0]["state_dict"]
    global_sd = {}

    for key in first_sd.keys():
        key_tensor = torch.zeros_like(first_sd[key], dtype=torch.float32)
        for p, norm_w in zip(payloads, normalized_weights):
            client_t = p["state_dict"][key].to(dtype=torch.float32)
            key_tensor += norm_w * client_t
        global_sd[key] = key_tensor

    return global_sd


class CalibrationAwareFedAvg:
    """
    Server-side wrapper for Calibration-Aware Federated Averaging (CA-FedAvg).
    """
    def __init__(self, lambda_param: float = 1.0):
        self.lambda_param = lambda_param

    def aggregate(
        self,
        payloads: List[Dict[str, Any]],
        client_eces: List[float]
    ) -> Dict[str, torch.Tensor]:
        return aggregate_ca_fedavg(payloads, client_eces, lambda_param=self.lambda_param)
