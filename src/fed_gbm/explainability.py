"""
SHAP Explainability Engine for FED-GBM v3 SurvivalMLP.

Uses shap.DeepExplainer on the trained PyTorch SurvivalMLP to generate:
1. Global feature importance summary plot (bar + beeswarm)
2. Per-patient waterfall charts for top-N patients
3. Force plots for Streamlit integration

Output: results/v3/phase7_final/shap/
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import List, Optional, Dict, Any

import torch
import shap


def _model_predict(model: torch.nn.Module, X: np.ndarray) -> np.ndarray:
    """
    Wrapper converting numpy input to probability predictions from SurvivalMLP.
    Always returns 1D array of shape (n_samples,) — required by KernelExplainer.
    """
    model.eval()
    X_t = np.atleast_2d(X)  # ensure (n, features) even for single sample
    with torch.no_grad():
        logits = model(torch.tensor(X_t, dtype=torch.float32))
    logits_np = logits.detach().cpu().numpy().reshape(-1)  # always 1D
    return 1.0 / (1.0 + np.exp(-logits_np.astype(np.float64)))


def compute_shap_values(
    model: torch.nn.Module,
    X_train: np.ndarray,
    X_explain: np.ndarray,
    feature_names: List[str],
    n_background: int = 30,
    seed: int = 42,
) -> np.ndarray:
    """
    Computes SHAP values using KernelExplainer (model-agnostic).
    KernelExplainer works with any PyTorch module including LayerNorm.

    Args:
        model:          Trained SurvivalMLP (torch.nn.Module in eval mode)
        X_train:        Training features (numpy array) for background
        X_explain:      Samples to explain (numpy array)
        feature_names:  List of feature names
        n_background:   Number of kmeans-summarized background samples
        seed:           Random seed for background selection

    Returns:
        shap_values: numpy array of shape (n_explain, n_features)
    """
    model.eval()
    np.random.seed(seed)

    # Summarize background with kmeans for speed
    n_bg = min(n_background, len(X_train))
    background = shap.kmeans(X_train, n_bg)

    predict_fn = lambda X: _model_predict(model, X)
    explainer = shap.KernelExplainer(predict_fn, background)

    # Use fewer nsamples for speed on small test sets
    n_samples = min(64, len(X_explain))
    shap_vals = explainer.shap_values(X_explain, nsamples=n_samples, silent=True)

    return np.array(shap_vals)


def plot_shap_summary(
    shap_values: np.ndarray,
    X_explain: np.ndarray,
    feature_names: List[str],
    output_dir: str,
    model_name: str,
    cohort_name: str,
    plot_type: str = "bar",
) -> str:
    """
    Generates and saves SHAP summary plot (bar or beeswarm).
    Uses matplotlib directly for compatibility with KernelExplainer outputs.

    Returns path to saved PNG.
    """
    os.makedirs(output_dir, exist_ok=True)

    sv = np.array(shap_values)
    if sv.ndim == 1:
        sv = sv.reshape(1, -1)

    mean_abs = np.abs(sv).mean(axis=0)
    sorted_idx = np.argsort(mean_abs)  # ascending for horizontal bar
    sorted_features = [feature_names[i] for i in sorted_idx]
    sorted_vals = mean_abs[sorted_idx]

    fig, ax = plt.subplots(figsize=(9, max(4, len(feature_names) * 0.55 + 1)))
    fig.patch.set_facecolor("#0D1117")
    ax.set_facecolor("#161B22")

    if plot_type == "beeswarm":
        # Beeswarm-style: scatter mean SHAP by feature value
        for fi, feat_i in enumerate(sorted_idx):
            vals = sv[:, feat_i]
            feat_vals = X_explain[:, feat_i]
            norm_feat = (feat_vals - feat_vals.min()) / ((feat_vals.max() - feat_vals.min()) + 1e-9)
            colors = plt.cm.RdBu_r(norm_feat)
            ax.scatter(vals, np.full_like(vals, fi), c=colors, alpha=0.7, s=30)
        ax.set_yticks(range(len(sorted_features)))
        ax.set_yticklabels(sorted_features, color="#AAAAAA", fontsize=10)
        ax.axvline(0, color="#555", linewidth=1)
        ax.set_xlabel("SHAP value (impact on model output)", color="#AAAAAA", fontsize=10)
    else:
        # Bar plot: mean |SHAP|
        bars = ax.barh(sorted_features, sorted_vals, color="#58A6FF", alpha=0.85, edgecolor="none")
        ax.set_xlabel("Mean |SHAP| (feature importance)", color="#AAAAAA", fontsize=10)
        ax.tick_params(colors="#AAAAAA")
        for bar_val, bar in zip(sorted_vals, bars):
            ax.text(bar_val + sorted_vals.max() * 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{bar_val:.4f}", va="center", ha="left", color="#AAAAAA", fontsize=8)

    ax.tick_params(colors="#AAAAAA")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")

    ax.set_title(
        f"SHAP Feature Importance ({plot_type.capitalize()})"
        f"\n{model_name} | {cohort_name}",
        color="white", fontsize=12, fontweight="bold", pad=10
    )
    ax.grid(axis="x", color="#2A2A2A", linestyle="--", alpha=0.5)

    plt.tight_layout()
    safe_name = model_name.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")
    out_path = os.path.join(output_dir, f"shap_summary_{cohort_name}_{safe_name}_{plot_type}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0D1117")
    plt.close(fig)

    return out_path


def plot_shap_waterfall(
    shap_values: np.ndarray,
    X_explain: np.ndarray,
    feature_names: List[str],
    base_value: float,
    patient_idx: int,
    patient_id: str,
    output_dir: str,
    model_name: str,
) -> str:
    """Generates per-patient SHAP waterfall chart using matplotlib."""
    os.makedirs(output_dir, exist_ok=True)

    sv_patient = np.array(shap_values[patient_idx])    # (n_features,)
    feat_vals = X_explain[patient_idx]

    # Sort by absolute value
    sorted_idx = np.argsort(np.abs(sv_patient))[::-1]
    top_k = min(len(feature_names), 8)
    top_idx = sorted_idx[:top_k]

    labels = [f"{feature_names[i]}\n= {feat_vals[i]:.2f}" for i in top_idx]
    vals = sv_patient[top_idx]
    colors = ["#FF4B4B" if v < 0 else "#00D4AA" for v in vals]

    fig, ax = plt.subplots(figsize=(9, max(4, top_k * 0.6 + 1.5)))
    fig.patch.set_facecolor("#0D1117")
    ax.set_facecolor("#161B22")

    bars = ax.barh(labels[::-1], vals[::-1], color=colors[::-1], alpha=0.85, edgecolor="none")
    ax.axvline(0, color="#666", linewidth=1)
    ax.set_xlabel("SHAP value", color="#AAAAAA", fontsize=10)
    ax.tick_params(colors="#AAAAAA")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")

    prob = base_value + float(sv_patient.sum())
    ax.set_title(
        f"Patient {patient_id}\nP(survive ≥12m) = {prob:.3f} | {model_name}",
        color="white", fontsize=11, fontweight="bold"
    )
    ax.grid(axis="x", color="#2A2A2A", linestyle="--", alpha=0.5)

    plt.tight_layout()
    safe_pid = patient_id.replace("-", "_")
    safe_name = model_name.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")
    out_path = os.path.join(output_dir, f"shap_waterfall_{safe_pid}_{safe_name}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0D1117")
    plt.close(fig)

    return out_path



def run_shap_analysis(
    model: torch.nn.Module,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_true: np.ndarray,
    y_probs: np.ndarray,
    feature_names: List[str],
    patient_ids: List[str],
    model_name: str,
    cohort_name: str,
    output_dir: str,
    n_waterfall_patients: int = 5,
) -> Dict[str, Any]:
    """
    Full SHAP analysis pipeline:
    1. Compute SHAP values
    2. Summary bar plot
    3. Beeswarm plot
    4. Waterfall charts for top-N patients by prediction confidence

    Returns dict with paths to all generated plots and top feature importance ranking.
    """
    results = {}

    shap_vals = compute_shap_values(
        model=model,
        X_train=X_train,
        X_explain=X_test,
        feature_names=feature_names,
    )
    # Guarantee 2D shape: (n_samples, n_features)
    shap_vals = np.atleast_2d(np.array(shap_vals))
    if shap_vals.shape[0] == len(feature_names) and shap_vals.shape[1] != len(feature_names):
        shap_vals = shap_vals.T  # transpose if accidentally transposed
    results["shap_values"] = shap_vals

    # Base value = mean model output on background
    model.eval()
    with torch.no_grad():
        bg_logits = model(torch.tensor(X_train[:min(50, len(X_train))], dtype=torch.float32)).squeeze()
        if bg_logits.dim() == 0:
            bg_logits = bg_logits.unsqueeze(0)
        bg_logits = bg_logits.numpy()
    base_value = float(np.mean(1.0 / (1.0 + np.exp(-bg_logits.astype(np.float64)))))
    results["base_value"] = base_value

    # Summary plots
    results["summary_bar"] = plot_shap_summary(
        shap_vals, X_test, feature_names, output_dir, model_name, cohort_name, "bar"
    )
    results["summary_beeswarm"] = plot_shap_summary(
        shap_vals, X_test, feature_names, output_dir, model_name, cohort_name, "beeswarm"
    )

    # Top-N patient waterfall charts (most confident high-risk + low-risk)
    sorted_idx = np.argsort(y_probs)
    top_high_risk = sorted_idx[:n_waterfall_patients // 2]      # lowest P(survive) = highest risk
    top_low_risk = sorted_idx[-(n_waterfall_patients - n_waterfall_patients // 2):]

    waterfall_paths = []
    for idx in np.concatenate([top_high_risk, top_low_risk]):
        if idx < len(X_test):
            pid = patient_ids[idx] if idx < len(patient_ids) else f"Patient_{idx}"
            wpath = plot_shap_waterfall(
                shap_vals, X_test, feature_names, base_value,
                patient_idx=int(idx), patient_id=str(pid),
                output_dir=output_dir, model_name=model_name,
            )
            waterfall_paths.append(wpath)

    results["waterfall_paths"] = waterfall_paths

    # Global feature importance ranking
    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
    if mean_abs_shap.ndim == 0:
        mean_abs_shap = np.array([float(mean_abs_shap)] * len(feature_names))
    importance_df = pd.DataFrame({
        "feature": feature_names[:len(mean_abs_shap)],
        "mean_abs_shap": mean_abs_shap[:len(feature_names)],
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    results["importance_df"] = importance_df

    imp_path = os.path.join(output_dir, f"shap_importance_{cohort_name}_{model_name.replace(' ', '_')}.csv")
    importance_df.to_csv(imp_path, index=False)
    results["importance_csv"] = imp_path

    return results
