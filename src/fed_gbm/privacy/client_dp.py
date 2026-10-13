"""
FED-GBM v3 Client-Level DP-FedAvg Module.
Implements Client-Level Differential Privacy for Federated Learning (Phase 6 & 6.1 Audit):

1. Complete parameter update vector calculation: Delta_k = theta_t^k - theta_t
2. L2 norm calculation across all trainable floating-point parameters
3. Client update clipping: Delta_k_clipped = Delta_k * min(1, S / ||Delta_k||_2)
4. Client update aggregation (Equal-weighted or Sample-size weighted)
5. Server-side Gaussian noise injection calibrated to exact aggregation sensitivity
6. Global model parameter update: theta_{t+1} = theta_t + Delta_private
7. Multi-round client-level privacy accounting across 30 federated rounds under
   Fixed Participating Clients / Dataset-Modification Adjacency (Adjacency A).
"""

import copy
import math
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import torch
import torch.nn as nn

from fed_gbm.models import set_seed


class ClientUpdateClipper:
    """
    Computes complete client model update vectors, L2 norms, clipping factors,
    and returns clipped parameter update vectors.
    """
    def __init__(self, clip_norm: float = 1.0):
        self.clip_norm = float(clip_norm)

    def compute_client_update(
        self,
        global_state_dict: Dict[str, torch.Tensor],
        client_state_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Computes update vector Delta_k = theta_t^k - theta_t for all trainable floating-point parameters.
        """
        return {
            key: (client_state_dict[key].detach() - g_tensor.detach())
            if torch.is_floating_point(g_tensor) else torch.zeros_like(g_tensor)
            for key, g_tensor in global_state_dict.items()
        }

    def compute_update_l2_norm(self, update_dict: Dict[str, torch.Tensor]) -> float:
        """
        Computes complete L2 norm across all floating-point parameter updates:
        ||Delta_k||_2 = sqrt( sum_p ||Delta_{k, p}||_2^2 )
        """
        total_sq_norm = sum(torch.sum(tensor ** 2).item() for tensor in update_dict.values() if torch.is_floating_point(tensor))
        return float(math.sqrt(total_sq_norm))

    def clip_update(
        self,
        update_dict: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], float, float]:
        """
        Clips update vector Delta_k to maximum L2 norm S:
        Delta_k_clipped = Delta_k * min(1, S / ||Delta_k||_2)
        
        Returns:
            Tuple of (clipped_update_dict, raw_l2_norm, clip_factor)
        """
        raw_norm = self.compute_update_l2_norm(update_dict)
        clip_factor = 1.0 if (raw_norm == 0.0 or math.isnan(raw_norm)) else min(1.0, self.clip_norm / (raw_norm + 1e-12))
            
        clipped_dict = {
            key: (tensor * clip_factor if torch.is_floating_point(tensor) else tensor.clone())
            for key, tensor in update_dict.items()
        }
        return clipped_dict, raw_norm, clip_factor


class ClientDPAccountant:
    """
    Client-Level Privacy Accountant for Deterministic Multi-Round Client Participation.
    Calculates cumulative (epsilon, delta)-DP bounds across T federated communication rounds
    under Fixed Participating Clients / Dataset-Modification Adjacency (Adjacency A).
    """
    def __init__(self, target_delta: float = 1e-5):
        self.target_delta = target_delta

    def compute_sensitivity(
        self,
        n_clients: int,
        clip_norm: float = 1.0,
        weighting_scheme: str = 'sample_size',
        sample_weights: Optional[List[float]] = None,
        adjacency_mode: str = 'replacement'
    ) -> float:
        """
        Derives exact L2 Global Sensitivity under Fixed Participating Clients Adjacency (Adjacency A):
        
        1. Equal Weighting:
           - zero_reference (change relative to 0 update): Delta S = S / K
           - replacement (worst-case update swap): Delta S = 2 * S / K
           
        2. Sample-Size Weighting (w_k = n_k / N fixed per client):
           - zero_reference: Delta S = max_k(w_k) * S
           - replacement: Delta S = 2 * max_k(w_k) * S
        """
        if weighting_scheme == 'equal':
            base_s = clip_norm / float(n_clients)
        elif weighting_scheme == 'sample_size' and sample_weights is not None:
            w_max = max(sample_weights) / float(sum(sample_weights))
            base_s = float(w_max) * clip_norm
        else:
            base_s = clip_norm / float(n_clients)
            
        if adjacency_mode == 'replacement':
            return 2.0 * base_s
        return base_s

    def compute_privacy(
        self,
        n_clients: int,
        rounds: int,
        sigma: float,
        clip_norm: float = 1.0,
        weighting_scheme: str = 'sample_size',
        sample_weights: Optional[List[float]] = None,
        adjacency_mode: str = 'replacement'
    ) -> Dict[str, Any]:
        """
        Computes formal client-level DP guarantee across T federated rounds.
        
        Uses closed-form analytical RDP minimizer for un-amplified Gaussian mechanism (q_client=1.0):
        RDP(alpha) = alpha * T / (2 * sigma^2)
        
        Analytical optimal alpha:
        alpha_opt = 1 + sqrt(2 * sigma^2 * ln(1/delta) / T)
        
        Exact minimal epsilon:
        epsilon(delta) = T / (2 * sigma^2) + sqrt(2 * T * ln(1/delta)) / sigma
        """
        if sigma <= 0.0:
            return {
                'n_clients': n_clients,
                'rounds': rounds,
                'sigma': 0.0,
                'clip_norm': clip_norm,
                'delta': self.target_delta,
                'epsilon': float('inf'),
                'accountant': 'Non-private baseline (sigma=0)'
            }
            
        sensitivity = self.compute_sensitivity(
            n_clients=n_clients,
            clip_norm=clip_norm,
            weighting_scheme=weighting_scheme,
            sample_weights=sample_weights,
            adjacency_mode=adjacency_mode
        )
        
        # Closed-form analytical RDP minimization for Gaussian mechanism across T rounds
        C1 = float(rounds) / (2.0 * (sigma ** 2))
        C2 = math.log(1.0 / self.target_delta)
        
        optimal_alpha = 1.0 + math.sqrt(C2 / C1)
        exact_epsilon = C1 * optimal_alpha + C2 / (optimal_alpha - 1.0)
        
        # Verify against fine continuous search grid
        fine_orders = np.linspace(1.0001, 100.0, 100000)
        eps_vals = [C1 * a + C2 / (a - 1.0) for a in fine_orders]
        fine_grid_epsilon = float(np.min(eps_vals))
        
        # Also record coarse grid value for historical audit reference
        coarse_orders = [1.1 + 0.1 * i for i in range(500)]
        coarse_vals = [C1 * a + C2 / (a - 1.0) for a in coarse_orders]
        coarse_grid_epsilon = float(np.min(coarse_vals))
        
        return {
            'n_clients': n_clients,
            'rounds': rounds,
            'sigma': float(sigma),
            'clip_norm': float(clip_norm),
            'sensitivity': float(sensitivity),
            'weighting_scheme': weighting_scheme,
            'adjacency_mode': adjacency_mode,
            'delta': float(self.target_delta),
            'epsilon': exact_epsilon,
            'fine_grid_epsilon': fine_grid_epsilon,
            'coarse_grid_epsilon': coarse_grid_epsilon,
            'optimal_order': float(optimal_alpha),
            'accountant': 'Closed-Form Gaussian RDP Accountant (Adjacency A)',
            'participation_model': f'Fixed {n_clients}-Client Deterministic Participation (q_client=1.0)'
        }


class ClientDPFedAvg:
    """
    Client-Level DP-FedAvg Server Orchestrator.
    Handles client update calculation, clipping, aggregation, server noise injection,
    and parameter updating.
    """
    def __init__(
        self,
        global_state_dict: Dict[str, torch.Tensor],
        clip_norm: float = 1.0,
        sigma: float = 0.0,
        weighting_scheme: str = 'sample_size',
        target_delta: float = 1e-5,
        adjacency_mode: str = 'replacement'
    ):
        self.global_state_dict = copy.deepcopy(global_state_dict)
        self.clipper = ClientUpdateClipper(clip_norm=clip_norm)
        self.sigma = float(sigma)
        self.weighting_scheme = weighting_scheme
        self.target_delta = target_delta
        self.adjacency_mode = adjacency_mode
        self.accountant = ClientDPAccountant(target_delta=target_delta)

    def aggregate_client_updates(
        self,
        client_payloads: List[Dict[str, Any]],
        seed: int = 42
    ) -> Dict[str, Any]:
        """
        Executes Server-Side Client-Level DP-FedAvg Aggregation:
        1. Extract client state_dicts and sample counts n_k.
        2. Compute complete update Delta_k = theta_t^k - theta_t for each client.
        3. Clip each update Delta_k to norm S.
        4. Aggregate clipped updates (Equal or Sample-size weighted).
        5. Add server-side Gaussian noise calibrated to exact aggregation sensitivity.
        6. Update global state dict: theta_{t+1} = theta_t + Delta_private.
        """
        set_seed(seed)
        K = len(client_payloads)
        if K == 0:
            raise ValueError("No client payloads received for aggregation.")
            
        sample_counts = [p['n_samples'] for p in client_payloads]
        total_samples = sum(sample_counts)
        
        # Calculate sensitivity and weights under Adjacency A
        sensitivity = self.accountant.compute_sensitivity(
            n_clients=K,
            clip_norm=self.clipper.clip_norm,
            weighting_scheme=self.weighting_scheme,
            sample_weights=sample_counts,
            adjacency_mode=self.adjacency_mode
        )
        
        if self.weighting_scheme == 'equal':
            weights = [1.0 / float(K)] * K
        else: # sample_size
            weights = [float(n) / float(total_samples) for n in sample_counts]
            
        clipped_updates = []
        raw_norms = []
        clip_factors = []
        
        # Process each client update against UNMUTATED global parameters
        for payload in client_payloads:
            c_state = payload['state_dict']
            update_dict = self.clipper.compute_client_update(self.global_state_dict, c_state)
            clipped_dict, raw_norm, c_factor = self.clipper.clip_update(update_dict)
            
            clipped_updates.append(clipped_dict)
            raw_norms.append(raw_norm)
            clip_factors.append(c_factor)
            
        # Weighted sum of clipped client updates
        aggregated_update = {}
        first_keys = self.global_state_dict.keys()
        
        for key in first_keys:
            g_tensor = self.global_state_dict[key]
            if torch.is_floating_point(g_tensor):
                agg_t = torch.zeros_like(g_tensor)
                for k in range(K):
                    agg_t += weights[k] * clipped_updates[k][key]
                aggregated_update[key] = agg_t
            else:
                # Non-floating buffers receive no update
                aggregated_update[key] = torch.zeros_like(g_tensor)
                
        # Server Gaussian noise injection if sigma > 0
        noisy_aggregated_update = {}
        if self.sigma > 0.0:
            noise_std = self.sigma * sensitivity
            for key, tensor in aggregated_update.items():
                if torch.is_floating_point(tensor):
                    noise = torch.randn_like(tensor) * noise_std
                    noisy_aggregated_update[key] = tensor + noise
                else:
                    noisy_aggregated_update[key] = tensor.clone()
        else:
            noisy_aggregated_update = aggregated_update
            
        # Update global parameters: theta_{t+1} = theta_t + Delta_private
        new_global_state_dict = {}
        for key in first_keys:
            g_tensor = self.global_state_dict[key]
            if torch.is_floating_point(g_tensor):
                new_global_state_dict[key] = g_tensor + noisy_aggregated_update[key]
            else:
                # Copy non-floating buffers from first client
                new_global_state_dict[key] = client_payloads[0]['state_dict'][key].clone()
                
        self.global_state_dict = copy.deepcopy(new_global_state_dict)
        
        return {
            'n_clients': K,
            'total_samples': total_samples,
            'raw_norms': raw_norms,
            'clip_factors': clip_factors,
            'weights': weights,
            'sensitivity': sensitivity,
            'sigma': self.sigma,
            'clip_norm': self.clipper.clip_norm,
            'weighting_scheme': self.weighting_scheme,
            'adjacency_mode': self.adjacency_mode,
            'global_state_dict': copy.deepcopy(self.global_state_dict)
        }
