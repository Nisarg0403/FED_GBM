"""
Heterogeneity-Aware Proximal Aggregation (HAPA) Module for FED-GBM v4.

Provides adaptive per-client proximal parameter calculation and drift-aware server aggregation:
1. Per-client adaptive mu: mu_k = mu_base * (1 + grad_variance_k)
2. Server aggregation: w_k \propto n_k / (1 + client_drift_k), where client_drift_k is L2 distance to global model.
"""

import copy
import torch
import numpy as np


class HeterogeneityAwareProximalAggregation:
    """Heterogeneity-Aware Proximal Aggregation (HAPA) server-side aggregator."""

    def __init__(self, mu_base: float = 0.01):
        self.mu_base = mu_base

    @staticmethod
    def compute_client_drift(client_state_dict: dict, global_state_dict: dict) -> float:
        """Computes Euclidean L2 distance between client parameter payload and global model parameters."""
        total_sq_dist = 0.0
        for k in global_state_dict:
            if k in client_state_dict and torch.is_floating_point(global_state_dict[k]):
                diff = client_state_dict[k].float() - global_state_dict[k].float()
                total_sq_dist += torch.sum(diff ** 2).item()
        return float(np.sqrt(total_sq_dist))

    def aggregate(self, client_payloads: list, global_state_dict: dict) -> dict:
        """
        Aggregates client payloads using drift-weighted averaging:
        w_k \propto n_k / (1 + drift_k)
        """
        if not client_payloads:
            raise ValueError("Empty client_payloads list provided to HAPA aggregator.")

        drifts = [
            self.compute_client_drift(p["state_dict"], global_state_dict)
            for p in client_payloads
        ]

        raw_weights = [
            float(p["n_samples"]) / (1.0 + d)
            for p, d in zip(client_payloads, drifts)
        ]
        total_weight = sum(raw_weights)
        norm_weights = [w / total_weight for w in raw_weights]

        aggregated_state = {}
        for key in global_state_dict.keys():
            if not torch.is_floating_point(global_state_dict[key]):
                aggregated_state[key] = copy.deepcopy(global_state_dict[key])
                continue

            param_type = global_state_dict[key].dtype
            weighted_param = torch.zeros_like(global_state_dict[key], dtype=torch.float32)

            for p, w in zip(client_payloads, norm_weights):
                weighted_param += w * p["state_dict"][key].to(torch.float32)

            aggregated_state[key] = weighted_param.to(param_type)

        return aggregated_state


def compute_adaptive_mu(base_mu: float, loss_variance: float) -> float:
    """Computes per-client adaptive proximal mu based on loss/gradient variance."""
    return float(base_mu * (1.0 + float(loss_variance)))
