"""
paths.py — Centralised path resolution for FED-GBM dashboard.

app/utils/paths.py is located at:
    FED_GBM_ANTIgravity_project/app/utils/paths.py

Two levels up reaches PROJECT_ROOT:
    FED_GBM_ANTIgravity_project/
"""
import os

# FED_GBM_ANTIgravity_project/ directory (two levels above app/utils/)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Phase 7.4 results directory (authoritative)
PHASE7_FINAL_DIR = os.path.join(PROJECT_ROOT, "results", "phase7_final")

# Splits manifests
SPLITS_DIR = os.path.abspath(os.path.join(PROJECT_ROOT, "data", "splits"))

# Core source package
SRC_DIR = os.path.abspath(os.path.join(PROJECT_ROOT, "src", "fed_gbm"))

# Artifact file paths
AGGREGATE_CSV       = os.path.join(PHASE7_FINAL_DIR, "phase7_aggregate_results.csv")
PER_SEED_CSV        = os.path.join(PHASE7_FINAL_DIR, "phase7_per_seed_results.csv")
PRIVACY_CSV         = os.path.join(PHASE7_FINAL_DIR, "phase7_privacy_results.csv")
CONFIG_JSON         = os.path.join(PHASE7_FINAL_DIR, "phase7_config.json")
SPLIT_VERIFY_JSON   = os.path.join(PHASE7_FINAL_DIR, "split_verification.json")
FEATURE_PROV_JSON   = os.path.join(PHASE7_FINAL_DIR, "feature_provenance.json")
AGG_VERIFY_JSON     = os.path.join(PHASE7_FINAL_DIR, "aggregation_verification.json")
AUDIT_MD            = os.path.join(PHASE7_FINAL_DIR, "PHASE_7.4_FINAL_AUDIT.md")
RESULTS_REPORT_MD   = os.path.join(PHASE7_FINAL_DIR, "PHASE_7_FINAL_RESULTS_REPORT.md")
PREDICTIONS_DIR     = os.path.join(PHASE7_FINAL_DIR, "predictions")
IMPROVED_AGGREGATE_CSV = os.path.join(PROJECT_ROOT, "results", "improved_experiments", "aggregate_results.csv")

