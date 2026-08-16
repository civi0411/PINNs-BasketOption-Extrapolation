"""
config.py
=========
Central configuration and parameter definitions for the PINN Basket Option Benchmark.

All global market conventions, network architectures, loss weighting coefficients,
and experiment grid definitions reside here to guarantee reproducibility and
single-source-of-truth consistency across all modules.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import torch

# ----------------------------------------------------------------------
# Global Market Parameters
# ----------------------------------------------------------------------
R_FREE: float = 0.05               # Risk-free annual interest rate
T_MAX: float = 1.0                 # Option maturity in years
S0_BASE: float = 100.0             # Reference initial spot price per asset
K_STRIKE: float = 100.0            # Option strike price (At-The-Money basket)
PAIRWISE_RHO: float = 0.30         # Flat pairwise correlation across assets
SEED: int = 42                     # Master random seed for reproducibility


def make_market(dim: int, seed: int = SEED) -> Dict[str, Any]:
    """
    Construct a reproducible, multi-asset financial market specification.

    Args:
        dim: Number of underlying assets in the basket (d >= 1).
        seed: Random seed for generating asset volatilities and spot offsets.

    Returns:
        Dict[str, Any] containing:
            - S0: (dim,) initial spot prices
            - sigma: (dim,) annualized volatilities in [0.10, 0.35]
            - corr: (dim, dim) pairwise correlation matrix with unit diagonal
            - weights: (dim,) equal basket weights summing to 1.0
            - K: scalar strike price
            - r: scalar risk-free rate
            - T: scalar time to maturity (years)
            - dim: integer dimension
            - option_type: default option type ("call")
    """
    rng = np.random.default_rng(seed)

    S0 = S0_BASE * (1.0 + 0.05 * rng.uniform(-1.0, 1.0, size=dim))
    sigma = 0.20 + 0.10 * rng.uniform(-1.0, 1.0, size=dim)
    sigma = np.clip(sigma, 0.10, 0.35)

    corr = np.full((dim, dim), PAIRWISE_RHO, dtype=np.float64)
    np.fill_diagonal(corr, 1.0)

    weights = np.ones(dim, dtype=np.float64) / float(dim)

    return {
        "S0": S0,
        "sigma": sigma,
        "corr": corr,
        "weights": weights,
        "K": K_STRIKE,
        "r": R_FREE,
        "T": T_MAX,
        "dim": dim,
        "option_type": "call",
    }


# ----------------------------------------------------------------------
# Neural Network & Training Hyperparameters
# ----------------------------------------------------------------------
NET_HIDDEN: int = 64               # Hidden units per layer
NET_LAYERS: int = 4                # Number of hidden layers in MLP
EPOCHS_DEFAULT: int = 2000         # Total optimization steps per model
LR_DEFAULT: float = 2e-3           # Adam learning rate
BATCH_DATA: int = 128              # Labeled data points sampled per step
BATCH_COLLOC: int = 128            # Unlabeled collocation points sampled per step

LAMBDA_DATA: float = 1.0
LAMBDA_PDE: float = 1.0
LAMBDA_IC: float = 1.0
LAMBDA_BC: float = 0.5

# ----------------------------------------------------------------------
# Data Generation & Sampling Bounds
# ----------------------------------------------------------------------
N_TRAIN_DEFAULT: int = 800         # Base labeled dataset size (100% regime)
N_TEST_DEFAULT: int = 300          # Held-out evaluation points

MC_PATHS_TRAIN_LABEL: int = 4000   # Monte Carlo paths for training label quotes
MC_PATHS_TEST_LABEL: int = 60000   # Monte Carlo paths for exact ground-truth test labels

LOG_MONEY_HALF_WIDTH: float = 0.55   # Full domain half-width
TRAIN_LOG_MONEY_HALF_WIDTH: float = 0.15  # Liquid NTM zone half-width

# ----------------------------------------------------------------------
# Experiment Configurations & Aliases for Notebook Compatibility
# ----------------------------------------------------------------------
EXPERIMENT_DIMS: List[int] = [1, 2, 3, 5]
NOISE_LEVELS: List[float] = [0.0, 0.01, 0.05, 0.10]
SPARSITY_FRACS: List[float] = [1.0, 0.5, 0.25, 0.10, 0.05, 0.02]
FIXED_DIM_FOR_NOISE_SPARSITY: int = 3
FIXED_DIM_FOR_EXTRAPOLATION: int = 3

# Aliases preserving 100% backward compatibility with notebooks/pinn_story.ipynb
EXTRAP_DIMS: List[int] = EXPERIMENT_DIMS
N_SEEDS_EXTRAP: int = 1
TRAIN_MONEY_HALF_WIDTH: float = TRAIN_LOG_MONEY_HALF_WIDTH
TRAIN_TAU_RANGE: Tuple[float, float] = (0.05, 1.0)
FULL_MONEY_HALF_WIDTH: float = LOG_MONEY_HALF_WIDTH
FULL_TAU_RANGE: Tuple[float, float] = (0.0, 1.0)


def get_device() -> torch.device:
    """
    Detect and return the optimal hardware acceleration device.
    Prioritizes Apple Silicon MPS, then CUDA, falling back to CPU.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
