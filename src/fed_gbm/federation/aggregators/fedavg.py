"""
FED-GBM v3 Parameter-Level FedAvg Module.
Implements pure PyTorch state_dict parameter aggregation.
Formulation: theta_global = sum_k (n_k / sum_j n_j) * theta_k
"""

import copy
from typing import List, Tuple, Dict, Any, Optional
import torch


def aggregate_state_dicts(
    client_updates: List[Tuple[Dict[str, torch.Tensor], int]],
    weighting_scheme: str = 'sample_size'
) -> Dict[str, torch.Tensor]:
    """
    Performs true parameter-level FedAvg on PyTorch state_dicts.
    
    Args:
        client_updates: List of tuples (state_dict, n_samples)
        weighting_scheme: 'sample_size' (default) or 'equal'
        
    Returns:
        Aggregated global state_dict
    """
    if not client_updates:
        raise ValueError("client_updates list cannot be empty")
        
    n_clients = len(client_updates)
    ref_state_dict, _ = client_updates[0]
    
    # Validate keys and tensor shapes across all clients
    for idx, (s_dict, n_s) in enumerate(client_updates):
        if set(s_dict.keys()) != set(ref_state_dict.keys()):
            raise KeyError(f"Client {idx} state_dict keys do not match reference keys.")
        for key in ref_state_dict.keys():
            if s_dict[key].shape != ref_state_dict[key].shape:
                raise ValueError(
                    f"Client {idx} tensor shape mismatch for key '{key}': "
                    f"expected {ref_state_dict[key].shape}, got {s_dict[key].shape}"
                )
                
    # Calculate aggregation weights
    if weighting_scheme == 'sample_size':
        total_samples = sum(n_s for _, n_s in client_updates)
        if total_samples <= 0:
            raise ValueError("Total sample count across clients must be > 0")
        weights = [float(n_s) / float(total_samples) for _, n_s in client_updates]
    elif weighting_scheme == 'equal':
        weights = [1.0 / float(n_clients) for _ in client_updates]
    else:
        raise ValueError(f"Unsupported weighting scheme: {weighting_scheme}")
        
    # Initialize aggregated state_dict with zero tensors matching reference shapes & dtypes
    aggregated_state_dict = {}
    for key, tensor in ref_state_dict.items():
        if tensor.is_floating_point():
            aggregated_state_dict[key] = torch.zeros_like(tensor, dtype=tensor.dtype)
        else:
            # For integer/boolean non-trainable buffers, use reference value if unweighted
            aggregated_state_dict[key] = tensor.clone()
            
    # Accumulate weighted parameter tensors
    for (s_dict, _), weight in zip(client_updates, weights):
        for key in ref_state_dict.keys():
            if ref_state_dict[key].is_floating_point():
                aggregated_state_dict[key] += s_dict[key].to(dtype=ref_state_dict[key].dtype) * weight
                
    return aggregated_state_dict
