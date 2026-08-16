"""
run_all_experiments.py
=======================
Master CLI driver script executing the entire 5-part quantitative benchmark suite.
Automatically resolves repository root paths, outputs tabular CSV logs to `results/`,
and renders vector-crisp publication charts into `figures/`.
"""

import os
import sys
import time
from typing import Dict, Any, Tuple

# Ensure `src/` modules can be imported cleanly from anywhere
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SRC_DIR)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from config import make_market, get_device
from evaluation.benchmarks import (
    run_experiment_1_dimension, run_experiment_2_noise,
    run_experiment_3_sparsity, run_experiment_4_speed,
    run_experiment_5_extrapolation
)
from evaluation.plotting import (
    plot_experiment1, plot_experiment2, plot_experiment3,
    plot_experiment4, plot_experiment5, plot_extrapolation_curve_1d
)


def main() -> None:
    t_master_start = time.perf_counter()
    results_dir = os.path.join(_REPO_ROOT, "results")
    figures_dir = os.path.join(_REPO_ROOT, "figures")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    device = get_device()
    print("=" * 70)
    print("PINN Basket Option Benchmark — Quantitative Execution Suite")
    print(f"Hardware Acceleration Device: {device.type.upper()}")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Experiment 1: Accuracy vs. Dimension (Curse of Dimensionality)
    # ------------------------------------------------------------------
    print("\n[1/5] EXPERIMENT 1: Accuracy vs. Dimension (Curse of Dimensionality)")
    print("-" * 70)
    t0 = time.perf_counter()
    df1, trained_models = run_experiment_1_dimension(verbose=True)
    df1.to_csv(os.path.join(results_dir, "exp1_dimension.csv"), index=False)
    plot_experiment1(df1, savepath=os.path.join(figures_dir, "exp1_dimension.png"))
    print(f"-> Exp 1 Completed in {time.perf_counter() - t0:.2f}s | Saved to results/ & figures/")

    # ------------------------------------------------------------------
    # Experiment 2: Robustness to Market Microstructure Noise
    # ------------------------------------------------------------------
    print("\n[2/5] EXPERIMENT 2: Robustness to Market Quote Microstructure Noise")
    print("-" * 70)
    t0 = time.perf_counter()
    df2 = run_experiment_2_noise(verbose=True)
    df2.to_csv(os.path.join(results_dir, "exp2_noise.csv"), index=False)
    plot_experiment2(df2, savepath=os.path.join(figures_dir, "exp2_noise.png"))
    print(f"-> Exp 2 Completed in {time.perf_counter() - t0:.2f}s | Saved to results/ & figures/")

    # ------------------------------------------------------------------
    # Experiment 3: Extreme Data Sparsity (Scarce Quote Regime)
    # ------------------------------------------------------------------
    print("\n[3/5] EXPERIMENT 3: Extreme Data Sparsity & Sample Efficiency")
    print("-" * 70)
    t0 = time.perf_counter()
    df3 = run_experiment_3_sparsity(verbose=True)
    df3.to_csv(os.path.join(results_dir, "exp3_sparsity.csv"), index=False)
    plot_experiment3(df3, savepath=os.path.join(figures_dir, "exp3_sparsity.png"))
    print(f"-> Exp 3 Completed in {time.perf_counter() - t0:.2f}s | Saved to results/ & figures/")

    # ------------------------------------------------------------------
    # Experiment 4: Sub-Millisecond Inference Speedup
    # ------------------------------------------------------------------
    print("\n[4/5] EXPERIMENT 4: Sub-Millisecond Inference Latency Benchmarking")
    print("-" * 70)
    t0 = time.perf_counter()
    df4 = run_experiment_4_speed(trained_models, verbose=True)
    df4.to_csv(os.path.join(results_dir, "exp4_speed.csv"), index=False)
    plot_experiment4(df4, savepath=os.path.join(figures_dir, "exp4_speed.png"))
    print(f"-> Exp 4 Completed in {time.perf_counter() - t0:.2f}s | Saved to results/ & figures/")

    # ------------------------------------------------------------------
    # Experiment 5: Domain Extrapolation & Tail-Risk Generalization
    # ------------------------------------------------------------------
    print("\n[5/5] EXPERIMENT 5: Domain Extrapolation (NTM to Tail-Risk OOD Pricing)")
    print("-" * 70)
    t0 = time.perf_counter()
    df5, extrap_models = run_experiment_5_extrapolation(verbose=True)
    df5.to_csv(os.path.join(results_dir, "exp5_extrapolation.csv"), index=False)
    plot_experiment5(df5, savepath=os.path.join(figures_dir, "exp5_extrapolation.png"))

    mkt3 = make_market(3)
    plot_extrapolation_curve_1d(
        extrap_models, mkt3,
        savepath=os.path.join(figures_dir, "exp5_extrap_1d.png")
    )
    print(f"-> Exp 5 Completed in {time.perf_counter() - t0:.2f}s | Saved to results/ & figures/")

    print("\n" + "=" * 70)
    print(f"ALL BENCHMARKS SUCCESSFULLY EXECUTED IN {time.perf_counter() - t_master_start:.2f}s.")
    print(f"Results: {results_dir}")
    print(f"Figures: {figures_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
