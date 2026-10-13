"""
Federated Architecture & Client/Server Coordination for FED-GBM
"""
from fed_gbm.federation.client import FederatedClient
from fed_gbm.federation.server import FederatedServer
from fed_gbm.federation.aggregators import (
    aggregate_state_dicts,
    CalibrationAwareFedAvg,
    compute_proximal_penalty,
    FedNovaServer,
    FedPerServer,
    SCAFFOLDServer,
    scaffold_local_train,
    HeterogeneityAwareProximalAggregation,
    compute_adaptive_mu,
)

__all__ = [
    "FederatedClient",
    "FederatedServer",
    "aggregate_state_dicts",
    "CalibrationAwareFedAvg",
    "compute_proximal_penalty",
    "FedNovaServer",
    "FedPerServer",
    "SCAFFOLDServer",
    "scaffold_local_train",
    "HeterogeneityAwareProximalAggregation",
    "compute_adaptive_mu",
]
