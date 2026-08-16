"""
evaluation package
==================
Quantitative evaluation benchmarks and vector-crisp visualization suite.
"""

from .benchmarks import (
    run_experiment_1_dimension, run_experiment_2_noise,
    run_experiment_3_sparsity, run_experiment_4_speed,
    run_experiment_5_extrapolation, run_experiment_extrapolation
)
from .plotting import (
    plot_experiment1, plot_experiment2, plot_experiment3,
    plot_experiment4, plot_experiment5, plot_extrapolation_curve_1d,
    PALETTE
)

__all__ = [
    "run_experiment_1_dimension", "run_experiment_2_noise",
    "run_experiment_3_sparsity", "run_experiment_4_speed",
    "run_experiment_5_extrapolation", "run_experiment_extrapolation",
    "plot_experiment1", "plot_experiment2", "plot_experiment3",
    "plot_experiment4", "plot_experiment5", "plot_extrapolation_curve_1d",
    "PALETTE"
]
