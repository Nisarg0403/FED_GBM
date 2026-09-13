"""
PyTorch Prognosis Model & Centralized Reference Baselines for FED-GBM v3.
- SurvivalMLP: Compact binary classification MLP returning ONE logit (for BCEWithLogitsLoss).
- Centralized reference models: Logistic Regression, Random Forest, XGBoost, PyTorch MLP.
- Reproducibility: Deterministic seed control & experiment metadata logging.
"""

import os
import sys
import json
import random
import datetime
import platform
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional

import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def set_seed(seed: int = 42):
    """
    Enforces deterministic seed handling across Python random, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SurvivalMLP(nn.Module):
    """
    PyTorch MLP for 12-month survival prognosis.
    Outputs ONE raw logit per sample for compatibility with BCEWithLogitsLoss.
    Does NOT apply sigmoid inside forward().
    Uses LayerNorm instead of BatchNorm to prevent small-batch instability.

    Default hidden_dims=[128, 64] matches the V3 architecture spec and published
    FL-GBM papers (Stripelis et al. 2024, BMC Medical Informatics).
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [128, 64],
        dropout_rate: float = 0.3,
        use_layer_norm: bool = True
    ):
        super(SurvivalMLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.use_layer_norm = use_layer_norm
        
        layers = []
        in_dim = input_dim
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.ReLU())
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(dropout_rate))
            in_dim = h_dim
            
        # Final linear output projection to 1 logit
        layers.append(nn.Linear(in_dim, 1))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning unscaled logits of shape (batch_size, 1).
        """
        return self.network(x)

    def get_config(self) -> Dict[str, Any]:
        """
        Returns model hyperparameter configuration.
        """
        return {
            'architecture': 'SurvivalMLP',
            'input_dim': self.input_dim,
            'hidden_dims': self.hidden_dims,
            'dropout_rate': self.dropout_rate,
            'use_layer_norm': self.use_layer_norm,
            'output_dim': 1,
            'activation': 'ReLU',
            'loss_function': 'BCEWithLogitsLoss'
        }


def get_pos_weight(y_train: np.ndarray) -> float:
    """
    Computes BCE pos_weight = (N_neg / N_pos) for class-imbalanced datasets.
    Passed to BCEWithLogitsLoss(pos_weight=...) to up-weight minority class.
    """
    n_pos = float(y_train.sum())
    n_neg = float(len(y_train) - n_pos)
    if n_pos == 0:
        return 1.0
    return n_neg / n_pos


def get_centralized_baselines(seed: int = 42, input_dim: int = 15) -> Dict[str, Any]:
    """
    Factory function returning initialized centralized reference models for Experiment A.
    All models receive identical preprocessed tabular feature vectors.
    """
    set_seed(seed)
    
    # 1. Logistic Regression (class_weight='balanced' for imbalanced labels)
    log_reg = LogisticRegression(
        C=1.0,
        solver='liblinear',
        class_weight='balanced',
        random_state=seed
    )
    
    # 2. Random Forest (more estimators + balanced for better AUC)
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        class_weight='balanced',
        random_state=seed
    )
    
    # 3. XGBoost Classifier (scale_pos_weight set at fit time from data)
    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        eval_metric='auc'
    )
    
    # 4. PyTorch MLP (upgraded to [128, 64] for real data)
    mlp = SurvivalMLP(
        input_dim=input_dim,
        hidden_dims=[128, 64],
        dropout_rate=0.3,
        use_layer_norm=True
    )
    
    return {
        'logistic_regression': log_reg,
        'random_forest': rf,
        'xgboost': xgb,
        'pytorch_mlp': mlp
    }


def save_experiment_metadata(
    artifact_dir: str,
    model_name: str,
    seed: int,
    feature_schema: List[str],
    preproc_config: Dict[str, Any],
    model_config: Dict[str, Any]
) -> str:
    """
    Saves complete experiment reproducibility metadata under artifacts/v3/.
    """
    os.makedirs(artifact_dir, exist_ok=True)
    
    meta = {
        'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'model_name': model_name,
        'seed': seed,
        'system_metadata': {
            'python_version': sys.version.split()[0],
            'torch_version': torch.__version__,
            'platform': platform.platform()
        },
        'feature_schema': {
            'feature_count': len(feature_schema),
            'feature_list': feature_schema
        },
        'preprocessing_config': preproc_config,
        'model_config': model_config
    }
    
    file_path = os.path.join(artifact_dir, f"meta_{model_name}_seed_{seed}.json")
    with open(file_path, 'w') as f:
        json.dump(meta, f, indent=2)
        
    return file_path
