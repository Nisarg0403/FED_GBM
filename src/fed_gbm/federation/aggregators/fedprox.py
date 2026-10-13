"""
FED-GBM v3 FedProx Module.
Implements proximal term regularization L_prox = (mu / 2) * sum(||theta - theta_global||^2)
to limit local client model drift during local SGD training.
"""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn


def compute_proximal_penalty(
    model: nn.Module,
    global_ref_params: Dict[str, torch.Tensor],
    mu: float
) -> torch.Tensor:
    """
    Computes the FedProx proximal penalty term:
    L_prox = (mu / 2.0) * sum_{param} ||param - ref_param||_2^2

    Args:
        model: Local PyTorch model instance
        global_ref_params: Dict mapping param name to fixed global reference tensor
        mu: Proximal regularization coefficient (>= 0.0)

    Returns:
        Scalar PyTorch Tensor representing the proximal penalty loss
    """
    if mu <= 0.0:
        return torch.tensor(0.0, device=next(model.parameters()).device)

    prox_sum = torch.tensor(0.0, device=next(model.parameters()).device)
    for name, param in model.named_parameters():
        if param.requires_grad and name in global_ref_params:
            ref_param = global_ref_params[name].to(device=param.device)
            prox_sum = prox_sum + torch.sum((param - ref_param) ** 2)

    return (mu / 2.0) * prox_sum
