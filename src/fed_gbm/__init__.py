"""
FED-GBM: Privacy-Preserving Federated Learning for Glioblastoma Multiforme Prognosis
Research-Grade Multimodal Framework
"""

__version__ = "3.0.0"

# Top-level direct imports for clean research API
from fed_gbm.models import (
    SurvivalMLP,
    CoxPHLoss,
    set_seed,
    get_pos_weight,
    get_centralized_baselines,
    save_experiment_metadata,
)
from fed_gbm.federation import (
    FederatedServer,
    FederatedClient,
    aggregate_state_dicts,
    CalibrationAwareFedAvg,
    compute_proximal_penalty,
    FedNovaServer,
    FedPerServer,
    SCAFFOLDServer,
    scaffold_local_train,
    HeterogeneityAwareProximalAggregation,
)
from fed_gbm.privacy import (
    compute_per_sample_clipped_gradients,
    train_step_dp_sgd,
    ClientUpdateClipper,
    ClientDPFedAvg,
    ClientDPAccountant,
    compute_rdp,
    convert_rdp_to_dp,
    RDPAccountant,
)
from fed_gbm.evaluation import (
    logits_to_probs,
    calculate_expected_calibration_error,
    evaluate_binary_classification,
    compute_nri_and_idi,
    compute_brier_decomposition,
    compute_time_dependent_auc,
    apply_platt_scaling,
    find_optimal_threshold_youden,
    generate_km_curves,
)
from fed_gbm.preprocessing import (
    LeakageFreeImputer,
    FederatedSufficientStatsScaler,
)
from fed_gbm.guards import (
    LeakageGuard,
    ProvenanceGuard,
    NoSyntheticGuard,
)

__all__ = [
    "__version__",
    "SurvivalMLP",
    "CoxPHLoss",
    "set_seed",
    "get_pos_weight",
    "get_centralized_baselines",
    "save_experiment_metadata",
    "FederatedServer",
    "FederatedClient",
    "aggregate_state_dicts",
    "CalibrationAwareFedAvg",
    "compute_proximal_penalty",
    "FedNovaServer",
    "FedPerServer",
    "SCAFFOLDServer",
    "scaffold_local_train",
    "HeterogeneityAwareProximalAggregation",
    "compute_per_sample_clipped_gradients",
    "train_step_dp_sgd",
    "ClientUpdateClipper",
    "ClientDPFedAvg",
    "ClientDPAccountant",
    "compute_rdp",
    "convert_rdp_to_dp",
    "RDPAccountant",
    "logits_to_probs",
    "calculate_expected_calibration_error",
    "evaluate_binary_classification",
    "compute_nri_and_idi",
    "compute_brier_decomposition",
    "compute_time_dependent_auc",
    "apply_platt_scaling",
    "find_optimal_threshold_youden",
    "generate_km_curves",
    "LeakageFreeImputer",
    "FederatedSufficientStatsScaler",
    "LeakageGuard",
    "ProvenanceGuard",
    "NoSyntheticGuard",
]
