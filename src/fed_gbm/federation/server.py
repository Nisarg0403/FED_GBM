"""
FED-GBM v3 Federated Server Module.
Orchestrates global model distribution, client participation sampling,
and sample-size weighted parameter aggregation (FedAvg).
"""

import copy
import random
import numpy as np
from typing import List, Dict, Any, Optional

import torch

from fed_gbm.models import SurvivalMLP, set_seed
from fed_gbm.federated_client import FederatedClient
from fed_gbm.fedavg import aggregate_state_dicts


class FederatedServer:
    """
    Centralized Federated Server managing global PyTorch model parameters.
    Orchestrates round-based parameter distribution and sample-weighted FedAvg aggregation.
    Does NOT store or process raw patient data.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [32, 16],
        dropout_rate: float = 0.2,
        use_layer_norm: bool = True,
        seed: int = 42
    ):
        set_seed(seed)
        self.input_dim = input_dim
        self.seed = seed
        
        # Instantiate global PyTorch model
        self.global_model = SurvivalMLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate,
            use_layer_norm=use_layer_norm
        )

    def get_global_state_dict(self) -> Dict[str, torch.Tensor]:
        """
        Returns a deep copy of the current global model's state_dict.
        """
        return copy.deepcopy(self.global_model.state_dict())

    def aggregate_client_payloads(
        self,
        client_payloads: List[Dict[str, Any]],
        weighting_scheme: str = 'sample_size'
    ) -> Dict[str, Any]:
        """
        Aggregates PyTorch parameter state_dicts returned by participating clients.
        Updates global model parameters in-place.
        """
        if not client_payloads:
            raise ValueError("client_payloads list cannot be empty")
            
        client_updates = [(payload['state_dict'], payload['n_samples']) for payload in client_payloads]
        
        # Aggregate parameters using FedAvg
        new_global_state_dict = aggregate_state_dicts(client_updates, weighting_scheme=weighting_scheme)
        
        # Load new parameters into global model
        self.global_model.load_state_dict(new_global_state_dict)
        
        total_samples = sum(p['n_samples'] for p in client_payloads)
        participating_ids = [p['client_id'] for p in client_payloads]
        
        return {
            'participating_clients': participating_ids,
            'total_participating_samples': total_samples,
            'weighting_scheme': weighting_scheme,
            'client_losses': {p['client_id']: p['final_loss'] for p in client_payloads}
        }

    def run_federated_round(
        self,
        clients: List[FederatedClient],
        round_idx: int = 1,
        local_epochs: int = 5,
        batch_size: int = 16,
        lr: float = 0.01,
        weight_decay: float = 1e-4,
        mu: float = 0.0,
        sigma: float = 0.0,
        C: float = 1.0,
        client_sampling_rate: float = 1.0,
        weighting_scheme: str = 'sample_size',
        seed: int = 42
    ) -> Dict[str, Any]:
        """
        Executes one full communication round:
        1. Select participating clients (rate q)
        2. Distribute global state_dict
        3. Execute local training independently on each client (with optional DP-SGD sigma/C, FedProx mu)
        4. Collect client payloads
        5. Aggregate client parameters centrally via sample-weighted FedAvg
        """
        set_seed(seed + round_idx)
        
        # Sample participating clients
        n_clients = len(clients)
        k_sample = max(1, int(np.ceil(client_sampling_rate * n_clients)))
        participating_clients = random.sample(clients, k_sample)
        
        global_state_dict = self.get_global_state_dict()
        client_payloads = []
        
        for client in participating_clients:
            # 1. Distribute global parameters
            client.set_model_parameters(global_state_dict)
            
            # 2. Local training
            payload = client.local_train(
                epochs=local_epochs,
                batch_size=batch_size,
                lr=lr,
                weight_decay=weight_decay,
                mu=mu,
                sigma=sigma,
                C=C,
                seed=seed + round_idx
            )
            client_payloads.append(payload)
            
        # 3. Parameter aggregation
        agg_meta = self.aggregate_client_payloads(client_payloads, weighting_scheme=weighting_scheme)
        agg_meta['round'] = round_idx
        agg_meta['client_sampling_rate'] = client_sampling_rate
        agg_meta['mu'] = mu
        agg_meta['sigma'] = sigma
        agg_meta['C'] = C
        
        return agg_meta
