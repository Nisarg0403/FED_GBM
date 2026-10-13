"""
FedPer Personalized Head Aggregator for FED-GBM v5.

Reference: Arivazhagan et al., "Federated Learning with Personalization Layers", arXiv 2019.
Aggregates shared backbone representation layers while keeping classification head local per client.
"""

import copy
import torch


class FedPerServer:
    """Server for FedPer aggregating shared backbone while preserving personalized head layers."""

    @staticmethod
    def aggregate_backbone(global_state_dict: dict, client_payloads: list, head_keys: list = None) -> dict:
        if head_keys is None:
            # By default for SurvivalMLP, head_keys is ['net.3.weight', 'net.3.bias'] (the output linear layer)
            head_keys = ["net.3.weight", "net.3.bias"]

        if not client_payloads:
            return global_state_dict

        total_samples = sum(p["n_samples"] for p in client_payloads)
        norm_weights = [float(p["n_samples"]) / total_samples for p in client_payloads]

        new_global_sd = {}
        for key in global_state_dict.keys():
            if key in head_keys:
                # Retain server head or don't overwrite local head
                new_global_sd[key] = copy.deepcopy(global_state_dict[key])
                continue

            if not torch.is_floating_point(global_state_dict[key]):
                new_global_sd[key] = copy.deepcopy(global_state_dict[key])
                continue

            weighted_param = torch.zeros_like(global_state_dict[key], dtype=torch.float32)
            for p, w in zip(client_payloads, norm_weights):
                weighted_param += w * p["state_dict"][key].to(torch.float32)

            new_global_sd[key] = weighted_param.to(global_state_dict[key].dtype)

        return new_global_sd
