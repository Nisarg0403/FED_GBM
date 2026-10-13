"""
Federated Aggregation Algorithms for FED-GBM
"""
from fed_gbm.federation.aggregators.fedavg import aggregate_state_dicts
from fed_gbm.federation.aggregators.ca_fedavg import CalibrationAwareFedAvg
from fed_gbm.federation.aggregators.fedprox import compute_proximal_penalty
from fed_gbm.federation.aggregators.fednova import FedNovaServer
from fed_gbm.federation.aggregators.fedper import FedPerServer
from fed_gbm.federation.aggregators.scaffold import SCAFFOLDServer, scaffold_local_train
from fed_gbm.federation.aggregators.hapa import HeterogeneityAwareProximalAggregation, compute_adaptive_mu

__all__ = [
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
