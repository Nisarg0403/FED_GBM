"""
data_loader.py — Cached loaders for Phase 7.4 final artifacts and real clinical cohort benchmarks.
"""
import json
import os
from typing import Optional, Tuple, Dict, Any

import pandas as pd
import streamlit as st

try:
    import app.utils.paths as paths
except ModuleNotFoundError:
    import utils.paths as paths

# Path definitions with resilient fallbacks
AGGREGATE_CSV = getattr(paths, "AGGREGATE_CSV", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "results", "phase7_final", "phase7_aggregate_results.csv")))
PER_SEED_CSV = getattr(paths, "PER_SEED_CSV", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "results", "phase7_final", "phase7_per_seed_results.csv")))
PRIVACY_CSV = getattr(paths, "PRIVACY_CSV", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "results", "phase7_final", "phase7_privacy_results.csv")))
CONFIG_JSON = getattr(paths, "CONFIG_JSON", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "results", "phase7_final", "phase7_config.json")))
SPLIT_VERIFY_JSON = getattr(paths, "SPLIT_VERIFY_JSON", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "results", "phase7_final", "split_verification.json")))
FEATURE_PROV_JSON = getattr(paths, "FEATURE_PROV_JSON", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "results", "phase7_final", "feature_provenance.json")))
AGG_VERIFY_JSON = getattr(paths, "AGG_VERIFY_JSON", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "results", "phase7_final", "aggregation_verification.json")))
IMPROVED_AGGREGATE_CSV = getattr(paths, "IMPROVED_AGGREGATE_CSV", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "results", "improved_experiments", "aggregate_results.csv")))
IMPROVED_PER_SEED_CSV = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "results", "improved_experiments", "per_seed_results.csv"))

REQUIRED_MODELS = [
    "Centralized Logistic Regression",
    "Centralized Random Forest",
    "Centralized XGBoost",
    "Centralized MLP",
    "FedAvg MLP",
    "FedProx MLP (mu=0.01)",
    "Patient DP (sigma=0.05)",
    "Patient DP (sigma=0.10)",
    "Patient DP (sigma=0.20)",
    "Client DP (sigma=0.05)",
    "Client DP (sigma=0.10)",
    "Client DP (sigma=0.20)",
]

REQUIRED_SEEDS = {42, 123, 2024, 3407, 9999}
REQUIRED_EVAL_SETS = {"TCGA_Locked_Test", "UPenn_External_Test"}


def phase7_final_available() -> Tuple[bool, str]:
    """Returns (True, "") if required Phase 7.4 files are present."""
    required = [AGGREGATE_CSV, PER_SEED_CSV, PRIVACY_CSV, CONFIG_JSON, SPLIT_VERIFY_JSON, FEATURE_PROV_JSON]
    missing = [f for f in required if not os.path.exists(f)]
    if missing:
        names = [os.path.basename(f) for f in missing]
        return False, f"Missing files: {', '.join(names)}"
    return True, ""


@st.cache_data(ttl=3600)
def load_config() -> Tuple[Optional[Dict[str, Any]], str]:
    if not os.path.exists(CONFIG_JSON):
        return None, "phase7_config.json not found."
    with open(CONFIG_JSON, "r") as f:
        cfg = json.load(f)
    return cfg, ""


@st.cache_data(ttl=3600)
def load_aggregate() -> Tuple[Optional[pd.DataFrame], str]:
    ok, err = phase7_final_available()
    if not ok:
        return None, err
    df = pd.read_csv(AGGREGATE_CSV)
    return df, ""


@st.cache_data(ttl=3600)
def load_per_seed() -> Tuple[Optional[pd.DataFrame], str]:
    ok, err = phase7_final_available()
    if not ok:
        return None, err
    df = pd.read_csv(PER_SEED_CSV)
    return df, ""


@st.cache_data(ttl=3600)
def load_privacy() -> Tuple[Optional[pd.DataFrame], str]:
    if not os.path.exists(PRIVACY_CSV):
        return None, "privacy CSV not found."
    df = pd.read_csv(PRIVACY_CSV)
    return df, ""


@st.cache_data(ttl=3600)
def load_split_verification() -> Tuple[Optional[Dict], str]:
    if not os.path.exists(SPLIT_VERIFY_JSON):
        return None, "split_verification.json not found."
    with open(SPLIT_VERIFY_JSON) as f:
        return json.load(f), ""


@st.cache_data(ttl=3600)
def load_feature_provenance() -> Tuple[Optional[Dict], str]:
    if not os.path.exists(FEATURE_PROV_JSON):
        return None, "feature_provenance.json not found."
    with open(FEATURE_PROV_JSON) as f:
        return json.load(f), ""


@st.cache_data(ttl=3600)
def load_aggregation_verification() -> Tuple[Optional[Dict], str]:
    if not os.path.exists(AGG_VERIFY_JSON):
        return None, "aggregation_verification.json not found."
    with open(AGG_VERIFY_JSON) as f:
        return json.load(f), ""


@st.cache_data(ttl=3600)
def load_tcga_fused() -> Optional[pd.DataFrame]:
    candidate = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw", "clinical_unique.csv")
    )
    if os.path.exists(candidate):
        return pd.read_csv(candidate)
    return None


@st.cache_data(ttl=3600)
def load_improved_aggregate() -> Tuple[Optional[pd.DataFrame], str]:
    """
    Loads multi-cohort aggregate results (TCGA, UPenn, CGGA) on real patient clinical data.
    """
    if not os.path.exists(IMPROVED_AGGREGATE_CSV):
        return None, "improved_experiments/aggregate_results.csv not found."
    df = pd.read_csv(IMPROVED_AGGREGATE_CSV)
    return df, ""


@st.cache_data(ttl=3600)
def load_improved_per_seed() -> Tuple[Optional[pd.DataFrame], str]:
    """
    Loads multi-cohort per-seed run results (TCGA, UPenn, CGGA) on real patient clinical data.
    """
    if not os.path.exists(IMPROVED_PER_SEED_CSV):
        return None, "improved_experiments/per_seed_results.csv not found."
    df = pd.read_csv(IMPROVED_PER_SEED_CSV)
    return df, ""


@st.cache_data(ttl=3600)
def load_deepsurv_benchmark() -> Tuple[Optional[pd.DataFrame], str]:
    """
    Loads DeepSurv benchmark results (Dirichlet Non-IID & DeepSurv Cox Partial Likelihood).
    """
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "results", "ieee_benchmark_results.csv"))
    if not os.path.exists(path):
        return None, "ieee_benchmark_results.csv not found."
    df = pd.read_csv(path)
    return df, ""


@st.cache_data(ttl=3600)
def load_v10_centralized() -> Tuple[Optional[pd.DataFrame], str]:
    """Loads v10 Centralized DeepSurv baselines."""
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "reports", "v10_centralized_baselines.csv"))
    if not os.path.exists(path):
        return None, "reports/v10_centralized_baselines.csv not found."
    return pd.read_csv(path), ""


@st.cache_data(ttl=3600)
def load_v10_federated() -> Tuple[Optional[pd.DataFrame], str]:
    """Loads v10 Multi-Seed Federated evaluation results."""
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "reports", "v10_federated_results.csv"))
    if not os.path.exists(path):
        return None, "reports/v10_federated_results.csv not found."
    return pd.read_csv(path), ""


@st.cache_data(ttl=3600)
def load_v10_significance() -> Tuple[Optional[pd.DataFrame], str]:
    """Loads v10 10-Seed Statistical Significance test results."""
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "reports", "v10_significance.csv"))
    if not os.path.exists(path):
        return None, "reports/v10_significance.csv not found."
    return pd.read_csv(path), ""


@st.cache_data(ttl=3600)
def load_v10_ablations() -> Tuple[Optional[pd.DataFrame], str]:
    """Loads v10 5-Dimension Ablation Matrix results."""
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "reports", "v10_ablation_matrix.csv"))
    if not os.path.exists(path):
        return None, "reports/v10_ablation_matrix.csv not found."
    return pd.read_csv(path), ""


