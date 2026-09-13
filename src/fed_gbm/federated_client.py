"""
FED-GBM v3 Federated Client Module.
Encapsulates local client data, isolated local PyTorch training,
and parameter state_dict payload generation.
"""

import copy
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from fed_gbm.models import SurvivalMLP, set_seed
from fed_gbm.fedprox import compute_proximal_penalty
from fed_gbm.dp_sgd import train_step_dp_sgd


class FederatedClient:
    """
    Isolated Federated Client holding local training data for one site (e.g. TCGA or UPenn).
    Performs local PyTorch training without sharing raw patient records.
    Supports FedAvg (mu=0, sigma=0), FedProx (mu > 0), and Patient-Level DP-SGD (sigma > 0, C=1.0).
    """
    def __init__(
        self,
        client_id: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        input_dim: int,
        hidden_dims: list = [32, 16],
        dropout_rate: float = 0.2,
        use_layer_norm: bool = True
    ):
        self.client_id = client_id
        self.n_samples = len(X_train)
        self.input_dim = input_dim
        
        # Store preprocessed local features & labels strictly
        self.X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
        self.y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
        
        # Instantiate local PyTorch model
        self.model = SurvivalMLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate,
            use_layer_norm=use_layer_norm
        )

    def set_model_parameters(self, global_state_dict: Dict[str, torch.Tensor]):
        """
        Loads global model parameter state_dict into local model.
        """
        self.model.load_state_dict(copy.deepcopy(global_state_dict))

    def get_model_parameters(self) -> Dict[str, torch.Tensor]:
        """
        Returns a deep copy of local model's current state_dict.
        """
        return copy.deepcopy(self.model.state_dict())

    def local_train(
        self,
        epochs: int = 5,
        batch_size: int = 16,
        lr: float = 0.01,
        weight_decay: float = 1e-4,
        mu: float = 0.0,
        sigma: float = 0.0,
        C: float = 1.0,
        seed: int = 42
    ) -> Dict[str, Any]:
        """
        Performs local PyTorch training using local data for E epochs.
        Supports Patient-Level DP-SGD (per-example clipping C=1.0, Gaussian noise sigma)
        and FedProx proximal term penalty (mu > 0.0).
        """
        set_seed(seed)
        self.model.train()
        
        # Freeze reference global parameters for FedProx proximal calculation
        global_ref_params = {
            name: param.detach().clone()
            for name, param in self.model.named_parameters()
        }
        
        dataset = TensorDataset(self.X_train_tensor, self.y_train_tensor)
        loader = DataLoader(
            dataset,
            batch_size=min(batch_size, self.n_samples),
            shuffle=True
        )
        
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        criterion = nn.BCEWithLogitsLoss()
        
        if sigma > 0.0:
            from opacus import PrivacyEngine
            gen = torch.Generator()
            gen.manual_seed(seed)
            privacy_engine = PrivacyEngine(accountant='rdp')
            
            # Deepcopy model for local training round to prevent re-hooking issues across rounds/calls
            local_model = copy.deepcopy(self.model)
            local_opt = torch.optim.Adam(
                local_model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
            
            # Opacus make_private wraps dataset with DPDataLoader & UniformWithReplacementSampler (Poisson Subsampling)
            # Each patient is independently included with probability q = batch_size / n_samples
            dp_objects = privacy_engine.make_private(
                module=local_model,
                optimizer=local_opt,
                data_loader=loader,
                noise_multiplier=sigma,
                max_grad_norm=C,
                poisson_sampling=True
            )
            if len(dp_objects) == 4:
                model_dp, optimizer_dp, _, loader_dp = dp_objects
            else:
                model_dp, optimizer_dp, loader_dp = dp_objects
            
            actual_steps = 0
            epoch_losses = []
            for _ in range(epochs):
                batch_losses = []
                for bx, by in loader_dp:
                    if bx.size(0) == 0:
                        continue  # Skip empty Poisson mini-batches
                    optimizer_dp.zero_grad()
                    logits = model_dp(bx)
                    bce_loss = criterion(logits, by)
                    bce_loss.backward()
                    optimizer_dp.step()
                    actual_steps += 1
                    batch_losses.append(bce_loss.item())
                epoch_losses.append(np.mean(batch_losses) if batch_losses else 0.0)
                
            final_loss = epoch_losses[-1] if epoch_losses else 0.0
            epsilon = privacy_engine.get_epsilon(delta=1e-5)
            best_alpha = getattr(privacy_engine.accountant, 'best_alpha', None)
            
            # Update self.model with trained weights (unwrapping GradSampleModule)
            self.model.load_state_dict(model_dp._module.state_dict() if hasattr(model_dp, '_module') else model_dp.state_dict())
            
            privacy_meta = {
                'n_samples': self.n_samples,
                'expected_batch_size': min(batch_size, self.n_samples),
                'epochs': epochs,
                'q': float(min(batch_size, self.n_samples)) / float(self.n_samples),
                'actual_steps': actual_steps,
                'C': float(C),
                'sigma': float(sigma),
                'delta': 1e-5,
                'epsilon': float(epsilon),
                'optimal_order': float(best_alpha) if best_alpha is not None else None,
                'accountant': 'Opacus v1.6.0 Subsampled Gaussian RDP Accountant',
                'sampling_mechanism': 'Poisson Subsampling (UniformWithReplacementSampler)'
            }
        else:
            epoch_losses = []
            for _ in range(epochs):
                batch_losses = []
                for bx, by in loader:
                    optimizer.zero_grad()
                    logits = self.model(bx)
                    bce_loss = criterion(logits, by)
                    prox_loss = compute_proximal_penalty(self.model, global_ref_params, mu)
                    total_loss = bce_loss + prox_loss
                    total_loss.backward()
                    optimizer.step()
                    b_loss = total_loss.item()
                    batch_losses.append(b_loss)
                epoch_losses.append(np.mean(batch_losses))
                
            final_loss = epoch_losses[-1] if epoch_losses else 0.0
            privacy_meta = {}
            
        # Construct aggregation payload (NO RAW DATA ATTACHED)
        payload = {
            'client_id': self.client_id,
            'n_samples': self.n_samples,
            'state_dict': copy.deepcopy(self.model.state_dict()),
            'local_epochs': epochs,
            'mu': mu,
            'sigma': sigma,
            'C': C,
            'privacy_meta': privacy_meta,
            'final_loss': float(final_loss)
        }
        
        return payload
