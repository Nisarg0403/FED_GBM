"""
FedNova Normalized Aggregator for FED-GBM v5.

Reference: Wang et al., "Tackling Objective Inconsistency in Federated Optimization", NeurIPS 2020.
Normalizes local parameter updates by client local steps to eliminate objective inconsistency.
"""

import copy
import torch


class FedNovaServer:
    """Server for FedNova aggregating normalized updates based on effective local steps."""

    def aggregate(self, global_state_dict: dict, client_payloads: list, client_local_steps: list) -> dict:
        if not client_payloads:
            return global_state_dict

        total_samples = sum(p["n_samples"] for p in client_payloads)
        p_weights = [float(p["n_samples"]) / total_samples for p in client_payloads]

        # Calculate normalized updates: delta_k = (w_global - w_k) / tau_k
        normalized_deltas = []
        for payload, tau_k in zip(client_payloads, client_local_steps):
            tau = max(1, tau_k)
            delta_k = {}
            for k, param in global_state_dict.items():
                if torch.is_floating_point(param):
                    diff = (param.float() - payload["state_dict"][k].float()) / tau
                    delta_k[k] = diff
            normalized_deltas.append(delta_k)

        tau_effective = sum(w * tau for w, tau in zip(p_weights, client_local_steps))

        new_global_sd = copy.deepcopy(global_state_dict)
        for key in global_state_dict.keys():
            if not torch.is_floating_point(global_state_dict[key]):
                continue

            avg_normalized_delta = torch.zeros_like(global_state_dict[key], dtype=torch.float32)
            for w, delta_k in zip(p_weights, normalized_deltas):
                avg_normalized_delta += w * delta_k[key]

            # w_global = w_global - tau_effective * avg_normalized_delta
            new_global_sd[key] = (global_state_dict[key].float() - tau_effective * avg_normalized_delta).to(global_state_dict[key].dtype)

        return new_global_sd
