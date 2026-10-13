"""
SCAFFOLD Federated Aggregator & Local Control Variates for FED-GBM v5.

Reference: Karimireddy et al., "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning", ICML 2020.
Corrects for client drift in non-IID settings via control variates c_k and c.
"""

import copy
import torch
import torch.nn as nn
import numpy as np


class SCAFFOLDServer:
    """Server for SCAFFOLD maintaining global model and global control variate c."""

    def __init__(self, global_state_dict: dict):
        self.c_global = {
            k: torch.zeros_like(v, dtype=torch.float32)
            for k, v in global_state_dict.items()
            if torch.is_floating_point(v)
        }

    def aggregate(
        self,
        global_state_dict: dict,
        client_payloads: list,
        client_c_updates: list
    ) -> tuple:
        """
        Aggregates model parameter updates and updates global control variate c.
        Returns: (new_global_state_dict, new_c_global)
        """
        num_clients = len(client_payloads)
        if num_clients == 0:
            return global_state_dict, self.c_global

        new_global_sd = copy.deepcopy(global_state_dict)

        # 1. Aggregate model parameters
        for key in global_state_dict.keys():
            if not torch.is_floating_point(global_state_dict[key]):
                continue
            avg_param = torch.zeros_like(global_state_dict[key], dtype=torch.float32)
            for p in client_payloads:
                avg_param += p["state_dict"][key].to(torch.float32)
            new_global_sd[key] = (avg_param / num_clients).to(global_state_dict[key].dtype)

        # 2. Update global control variate c = c + (1 / K) * sum(delta_c_k)
        for key in self.c_global.keys():
            avg_delta_c = torch.zeros_like(self.c_global[key])
            for delta_c in client_c_updates:
                if key in delta_c:
                    avg_delta_c += delta_c[key]
            self.c_global[key] += avg_delta_c / num_clients

        return new_global_sd, self.c_global


def scaffold_local_train(
    model: nn.Module,
    X_tr: torch.Tensor,
    y_tr: torch.Tensor,
    c_local: dict,
    c_global: dict,
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 0.01
) -> tuple:
    """
    Local training step with SCAFFOLD gradient correction: grad = grad - c_k + c.
    Returns: (new_state_dict, delta_c_k, new_c_local)
    """
    initial_sd = copy.deepcopy(model.state_dict())
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    dataset = torch.utils.data.TensorDataset(X_tr, y_tr.unsqueeze(1) if y_tr.ndim == 1 else y_tr)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    total_steps = 0
    for _ in range(epochs):
        for bx, by in loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()

            # Apply SCAFFOLD gradient correction: g = g - c_k + c
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if param.grad is not None and name in c_local and name in c_global:
                        param.grad += (c_global[name] - c_local[name])

            optimizer.step()
            total_steps += 1

    new_sd = model.state_dict()

    # Compute delta_c_k = c_k^+ - c_k = (initial - new) / (steps * lr) - c + c_k
    steps_lr = max(1, total_steps) * lr
    new_c_local = {}
    delta_c_k = {}

    for k in c_local.keys():
        if k in initial_sd and k in new_sd:
            param_diff = (initial_sd[k].float() - new_sd[k].float()) / steps_lr
            c_plus = param_diff - c_global[k] + c_local[k]
            delta_c_k[k] = c_plus - c_local[k]
            new_c_local[k] = c_plus
        else:
            new_c_local[k] = copy.deepcopy(c_local[k])
            delta_c_k[k] = torch.zeros_like(c_local[k])

    return new_sd, delta_c_k, new_c_local
