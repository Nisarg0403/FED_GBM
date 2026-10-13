"""
FED-GBM v3 Patient-Level DP-SGD Module.
Implements per-example gradient computation, per-example L2 norm clipping (C=1.0),
calibrated Gaussian noise injection, and exact mini-batch optimizer updates.
"""

import math
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import torch
import torch.nn as nn

from fed_gbm.fedprox import compute_proximal_penalty


def compute_per_sample_clipped_gradients(
    model: nn.Module,
    bx: torch.Tensor,
    by: torch.Tensor,
    criterion: nn.Module,
    C: float = 1.0
) -> Tuple[Dict[str, torch.Tensor], List[float]]:
    """
    Computes per-example gradients independently for each example i in mini-batch bx, by.
    Clips each patient's gradient vector independently to L2 norm bound C.
    
    Args:
        model: PyTorch model instance
        bx: Mini-batch input tensor (batch_size, input_dim)
        by: Mini-batch target tensor (batch_size, 1)
        criterion: Loss function (nn.BCEWithLogitsLoss)
        C: Per-example clipping norm bound (default = 1.0)
        
    Returns:
        Tuple of (clipped_gradient_sums_dict, per_sample_raw_norms_list)
    """
    batch_size = bx.size(0)
    trainable_params = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    
    # Store per-sample gradient dicts
    per_sample_grads = []
    per_sample_norms = []
    
    for i in range(batch_size):
        xi = bx[i:i+1]
        yi = by[i:i+1]
        
        logit_i = model(xi)
        loss_i = criterion(logit_i, yi)
        
        # Compute gradient for example i independently
        grads_i = torch.autograd.grad(
            loss_i,
            [p for _, p in trainable_params],
            retain_graph=True,
            create_graph=False
        )
        
        # Compute L2 norm across all trainable parameters for example i
        total_sq_norm = 0.0
        sample_g_dict = {}
        for (name, _), g in zip(trainable_params, grads_i):
            sample_g_dict[name] = g
            total_sq_norm += torch.sum(g ** 2).item()
            
        sample_norm = math.sqrt(total_sq_norm)
        per_sample_norms.append(float(sample_norm))
        
        # Calculate per-example clip factor
        clip_factor = min(1.0, float(C) / (sample_norm + 1e-12))
        
        # Clip sample gradients independently
        clipped_g_dict = {name: g * clip_factor for name, g in sample_g_dict.items()}
        per_sample_grads.append(clipped_g_dict)
        
    # Sum clipped gradients across mini-batch examples
    clipped_sums = {}
    for name, param in trainable_params:
        clipped_sums[name] = torch.zeros_like(param)
        for i in range(batch_size):
            clipped_sums[name] += per_sample_grads[i][name]
            
    return clipped_sums, per_sample_norms


def train_step_dp_sgd(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    bx: torch.Tensor,
    by: torch.Tensor,
    C: float = 1.0,
    sigma: float = 0.0,
    mu: float = 0.0,
    global_ref_params: Optional[Dict[str, torch.Tensor]] = None
) -> float:
    """
    Executes one mini-batch Patient-Level DP-SGD training step:
    1. Per-example gradient computation
    2. Per-example L2 norm clipping (C = 1.0)
    3. Calibrated Gaussian noise addition (N(0, sigma^2 C^2 I))
    4. Optional FedProx proximal regularization addition
    5. Mini-batch averaged parameter update via optimizer.step()
    """
    batch_size = bx.size(0)
    optimizer.zero_grad()
    
    # 1 & 2. Compute per-example gradients and clip per patient
    clipped_sums, _ = compute_per_sample_clipped_gradients(model, bx, by, criterion, C=C)
    
    # 3. Add Gaussian noise calibrated to clipping bound C
    trainable_params = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    
    for name, param in trainable_params:
        grad_sum = clipped_sums[name]
        
        if sigma > 0.0:
            noise_std = sigma * C
            noise = torch.randn_like(param) * noise_std
            noisy_grad_sum = grad_sum + noise
        else:
            noisy_grad_sum = grad_sum
            
        # Mini-batch average gradient
        param.grad = noisy_grad_sum / float(batch_size)
        
    # 4. If FedProx enabled, add proximal penalty gradient
    if mu > 0.0 and global_ref_params is not None:
        prox_loss = compute_proximal_penalty(model, global_ref_params, mu)
        if prox_loss.requires_grad:
            prox_loss.backward()
            
    # Compute batch loss for logging
    with torch.no_grad():
        batch_logits = model(bx)
        batch_loss = criterion(batch_logits, by).item()
        
    optimizer.step()
    return float(batch_loss)
