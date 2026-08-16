"""
plotting.py
===========
Publication-grade Matplotlib plotting suite for quantitative benchmark experiments.
Generates vector-crisp visualizations cleanly styled for academic reporting.
"""

from typing import Optional, Dict, Any, List, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PALETTE: Dict[str, str] = {
    "VanillaNN": "#9CA3AF",
    "PINN": "#2563EB",
    "ICPINN": "#7C3AED",
    "MonteCarlo": "#F59E0B",
    "FDM": "#10B981",
    "BinomialTree": "#EF4444",
    "MonteCarlo(8k paths)": "#F59E0B",
}


def _style_ax(ax: plt.Axes) -> None:
    """Apply clean, minimalist academic chart styling."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.35, color="#D1D5DB")
    ax.tick_params(axis="both", which="major", labelsize=9.5)


def plot_experiment1(df: pd.DataFrame, savepath: Optional[str] = None) -> plt.Figure:
    """Plot Relative Pricing Error scaling across asset dimensions d in {1, 2, 3, 5}."""
    dims = sorted(df["dim"].unique())
    methods = ["VanillaNN", "PINN", "ICPINN", "MonteCarlo", "FDM", "BinomialTree"]
    fig, ax = plt.subplots(figsize=(9.0, 5.5))
    width = 0.13
    x = np.arange(len(dims))
    present_methods = [m for m in methods if m in df["method"].unique()]

    for i, m in enumerate(present_methods):
        sub = df[df["method"] == m].set_index("dim").reindex(dims)["rel_error"]
        offset = (i - len(present_methods) / 2.0) * width + width / 2.0
        ax.bar(x + offset, sub.values, width, label=m, color=PALETTE.get(m, "#6B7280"), edgecolor="white", linewidth=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{d}D Basket" for d in dims], fontsize=10.5, fontweight="medium")
    ax.set_ylabel("Normalized Relative Pricing Error", fontsize=10.5)
    ax.set_title("Experiment 1: Pricing Accuracy vs. Asset Dimension (Curse of Dimensionality)", fontsize=12.0, fontweight="semibold", pad=12)
    ax.legend(frameon=False, ncol=3, fontsize=9.5, loc="upper left")
    _style_ax(ax)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=200, bbox_inches="tight")
    return fig


def plot_experiment2(df: pd.DataFrame, savepath: Optional[str] = None) -> plt.Figure:
    """Plot model robustness curves across varying market quote microstructure noise levels."""
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    for m in ["VanillaNN", "PINN", "ICPINN"]:
        if m not in df["method"].unique():
            continue
        sub = df[df["method"] == m].sort_values("noise")
        ax.plot(
            sub["noise"] * 100.0, sub["rel_error"] * 100.0,
            marker="o", linewidth=2.2, markersize=6.5, label=m, color=PALETTE.get(m, "#6B7280")
        )

    ax.set_xlabel("Market Quote Noise Standard Deviation (%)", fontsize=10.5)
    ax.set_ylabel("Relative Pricing Error (%)", fontsize=10.5)
    ax.set_title(f"Experiment 2: Microstructure Noise Robustness ({df['dim'].iloc[0]}D Basket)", fontsize=12.0, fontweight="semibold", pad=12)
    ax.legend(frameon=False, fontsize=10.0)
    _style_ax(ax)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=200, bbox_inches="tight")
    return fig


def plot_experiment3(df: pd.DataFrame, savepath: Optional[str] = None) -> plt.Figure:
    """Plot sample data efficiency curves as labeled training quote density is reduced."""
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    for m in ["VanillaNN", "PINN", "ICPINN"]:
        if m not in df["method"].unique():
            continue
        sub = df[df["method"] == m].sort_values("frac", ascending=False)
        ax.plot(
            sub["frac"] * 100.0, sub["rel_error"] * 100.0,
            marker="s", linewidth=2.2, markersize=6.5, label=m, color=PALETTE.get(m, "#6B7280")
        )

    ax.set_xlabel("Training Quote Availability Ratio (%)", fontsize=10.5)
    ax.set_ylabel("Relative Pricing Error (%)", fontsize=10.5)
    ax.set_title(f"Experiment 3: Extreme Data Sparsity Resilience ({df['dim'].iloc[0]}D Basket)", fontsize=12.0, fontweight="semibold", pad=12)
    ax.invert_xaxis()
    ax.legend(frameon=False, fontsize=10.0)
    _style_ax(ax)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=200, bbox_inches="tight")
    return fig


def plot_experiment4(df: pd.DataFrame, savepath: Optional[str] = None) -> plt.Figure:
    """Plot logarithmic inference latency comparison per query across dimensions."""
    dims = sorted(df["dim"].unique())
    methods = [m for m in ["VanillaNN", "PINN", "ICPINN", "MonteCarlo(8k paths)"] if m in df["method"].unique()]
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    width = 0.18
    x = np.arange(len(dims))

    for i, m in enumerate(methods):
        sub = df[df["method"] == m].set_index("dim").reindex(dims)["per_query_ms"]
        offset = (i - len(methods) / 2.0) * width + width / 2.0
        ax.bar(x + offset, sub.values, width, label=m, color=PALETTE.get(m, "#6B7280"), edgecolor="white", linewidth=0.6)

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{d}D Basket" for d in dims], fontsize=10.5, fontweight="medium")
    ax.set_ylabel("Inference Latency per Query (ms, log scale)", fontsize=10.5)
    ax.set_title("Experiment 4: Sub-Millisecond Inference Speedup", fontsize=12.0, fontweight="semibold", pad=12)
    ax.legend(frameon=False, fontsize=9.5)
    _style_ax(ax)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=200, bbox_inches="tight")
    return fig


def plot_experiment5(df: pd.DataFrame, savepath: Optional[str] = None) -> plt.Figure:
    """Bar chart comparing Relative Pricing Error in Interpolation vs Extrapolation zones."""
    if "dim" in df.columns and len(df["dim"].unique()) > 1:
        target_dim = 3 if 3 in df["dim"].unique() else df["dim"].iloc[0]
        subdf = df[df["dim"] == target_dim]
        dim_label = f" ({target_dim}D Basket)"
    else:
        subdf = df.copy()
        dim_label = f" ({subdf['dim'].iloc[0]}D Basket)" if "dim" in subdf.columns else ""

    methods = [m for m in ["VanillaNN", "PINN", "ICPINN"] if m in subdf["method"].unique()]
    zones = ["interpolation", "extrapolation"]
    zone_labels = ["Interpolation Zone\n(Observed NTM Quotes)", "Extrapolation Zone\n(Tail-Risk OOD Manifold)"]

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    width = 0.22
    x = np.arange(len(zones))

    for i, m in enumerate(methods):
        vals = []
        for z in zones:
            row = subdf[(subdf["method"] == m) & (subdf["zone"] == z)]
            vals.append(float(row["rel_error"].values[0]) if len(row) > 0 else 0.0)
        offset = (i - len(methods) / 2.0) * width + width / 2.0
        ax.bar(x + offset, vals, width, label=m, color=PALETTE.get(m, "#6B7280"), edgecolor="white", linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(zone_labels, fontsize=10.5, fontweight="medium")
    ax.set_ylabel("Normalized Relative Pricing Error", fontsize=10.5)
    ax.set_title(f"Experiment 5: Domain Extrapolation vs. Interpolation Fidelity{dim_label}", fontsize=12.0, fontweight="semibold", pad=12)
    ax.legend(frameon=False, fontsize=10.0)
    _style_ax(ax)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=200, bbox_inches="tight")
    return fig


def plot_extrapolation_curve_1d(
    extrap_models: Dict[Tuple[int, str], Any],
    mkt: Dict[str, Any],
    moneyness_bound: float = 0.15,
    savepath: Optional[str] = None
) -> plt.Figure:
    """1D cross-section profile of u(S, tau=0.5) vs asset price S_1 across the domain continuum."""
    from data.monte_carlo import mc_basket_price_batch_with_se
    from engine.trainer import predict

    dim = mkt["dim"]
    y_grid = np.linspace(-0.55, 0.55, 140)
    S_slice = np.full((140, dim), mkt["S0"])
    S_slice[:, 0] = mkt["S0"][0] * np.exp(y_grid)
    tau_slice = np.full(140, 0.5)

    rng = np.random.default_rng(9999)
    true_price, _ = mc_basket_price_batch_with_se(
        S_slice, tau_slice, mkt["K"], mkt["r"], mkt["sigma"], mkt["corr"], mkt["weights"],
        option_type="call", n_samples=30000, rng=rng
    )

    fig, ax = plt.subplots(figsize=(9.2, 5.5))
    ax.plot(
        S_slice[:, 0], true_price, color="#111827", linestyle="--",
        linewidth=2.4, label="Exact Monte Carlo (Ground Truth)"
    )

    for model_type in ["VanillaNN", "PINN", "ICPINN"]:
        model = extrap_models.get((dim, model_type))
        if model is None:
            continue
        pred = predict(model, S_slice, tau_slice, mkt)
        ax.plot(
            S_slice[:, 0], pred, label=model_type,
            color=PALETTE.get(model_type, "#6B7280"), linewidth=2.2, alpha=0.9
        )

    S_low = mkt["S0"][0] * np.exp(-moneyness_bound)
    S_high = mkt["S0"][0] * np.exp(moneyness_bound)
    ax.axvspan(S_low, S_high, color="#10B981", alpha=0.14, label="Observed Liquid NTM Manifold (Training Data)")
    ax.axvline(S_low, color="#059669", linestyle=":", alpha=0.85, linewidth=1.4)
    ax.axvline(S_high, color="#059669", linestyle=":", alpha=0.85, linewidth=1.4)

    ax.set_xlabel(f"Asset 1 Price ($S_1$, Reference Spot $S_0={mkt['S0'][0]:.0f}$)", fontsize=10.5)
    ax.set_ylabel("Option Price $u(S, \\tau=0.5)$", fontsize=10.5)
    ax.set_title(f"1D Cross-Section: Out-Of-Distribution Tail-Risk Pricing ({dim}D Basket)", fontsize=12.0, fontweight="semibold", pad=12)
    ax.legend(frameon=False, fontsize=9.5, loc="upper left")
    _style_ax(ax)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=200, bbox_inches="tight")
    return fig
