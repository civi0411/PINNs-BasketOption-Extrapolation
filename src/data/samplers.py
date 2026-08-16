"""
samplers.py
===========
Domain point samplers generating labeled and unsupervised collocation datasets
in log-moneyness coordinate space across asset dimensions.
"""

from typing import Dict, Any, Tuple, Optional, Union
import numpy as np
import torch
from .monte_carlo import mc_basket_price_batch_with_se, basket_payoff


def sample_domain_points(
    mkt: Dict[str, Any],
    n_points: int,
    rng: np.random.Generator,
    tau_min_frac: float = 0.0,
    moneyness_half_width: Optional[float] = None,
    tau_max: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample uniform points inside the full space (or a specified NTM sub-box)."""
    dim = mkt["dim"]
    if moneyness_half_width is None:
        moneyness_half_width = mkt["log_moneyness_range"][1]
    tau_high = tau_max if tau_max is not None else mkt["T"]
    tau_low = tau_min_frac * tau_high

    tau = rng.uniform(tau_low, tau_high, size=n_points)
    y = rng.uniform(-moneyness_half_width, moneyness_half_width, size=(n_points, dim))
    S = mkt["S0"][None, :] * np.exp(y)
    return S, tau


def sample_extrapolation_train_points(
    mkt: Dict[str, Any],
    n_points: int,
    rng: np.random.Generator,
    moneyness_bound: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample training points exclusively from the liquid Near-The-Money (NTM) sub-domain."""
    tau_high = mkt["T"]
    tau_low = 0.05 * tau_high
    tau = rng.uniform(tau_low, tau_high, size=n_points)
    y = rng.uniform(-moneyness_bound, moneyness_bound, size=(n_points, mkt["dim"]))
    S = mkt["S0"][None, :] * np.exp(y)
    return S, tau


def make_labeled_dataset(
    mkt: Dict[str, Any],
    n_points: int,
    mc_paths: int,
    seed: int = 42,
    noise_std: float = 0.0,
    moneyness_half_width: Optional[float] = None,
    tau_max: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate high-precision labeled quote pairs (S_i, tau_i) -> price_i via Monte Carlo."""
    rng = np.random.default_rng(seed)
    S, tau = sample_domain_points(mkt, n_points, rng, tau_min_frac=0.05,
                                   moneyness_half_width=moneyness_half_width, tau_max=tau_max)
    prices, _ = mc_basket_price_batch_with_se(
        S, tau, mkt["K"], mkt["r"], mkt["sigma"], mkt["corr"], mkt["weights"],
        option_type=mkt.get("option_type", "call"), n_samples=mc_paths, rng=rng
    )
    if noise_std > 0.0:
        prices += rng.normal(0.0, noise_std * np.maximum(prices, 0.5))
        prices = np.maximum(prices, 0.0)
    return S, tau, prices


def make_extrapolation_labeled_dataset(
    mkt: Dict[str, Any],
    n_points: int,
    mc_paths: int,
    moneyness_bound: float,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate labeled training quotes strictly bounded within the NTM region."""
    rng = np.random.default_rng(seed)
    S, tau = sample_extrapolation_train_points(mkt, n_points, rng, moneyness_bound=moneyness_bound)
    prices, _ = mc_basket_price_batch_with_se(
        S, tau, mkt["K"], mkt["r"], mkt["sigma"], mkt["corr"], mkt["weights"],
        option_type=mkt.get("option_type", "call"), n_samples=mc_paths, rng=rng
    )
    return S, tau, prices


def make_collocation_points(
    mkt: Dict[str, Any],
    n_points: int,
    seed: int = 43,
    moneyness_half_width: Optional[float] = None,
    tau_max: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate unsupervised interior collocation coordinates (S, tau) for PDE residual minimization."""
    rng = np.random.default_rng(seed)
    return sample_domain_points(mkt, n_points, rng, tau_min_frac=0.01,
                                moneyness_half_width=moneyness_half_width, tau_max=tau_max)


def make_boundary_points(
    mkt: Dict[str, Any],
    n_points: int,
    seed: int = 44,
    moneyness_half_width: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate boundary condition evaluation coordinates using exact asymptotic Dirichlet bounds."""
    rng = np.random.default_rng(seed)
    dim = mkt["dim"]
    if moneyness_half_width is None:
        moneyness_half_width = mkt["log_moneyness_range"][1]
    tau = rng.uniform(0.01, mkt["T"], size=n_points)
    y = rng.uniform(-moneyness_half_width, moneyness_half_width, size=(n_points, dim))

    # Force a random subset of dimensions to extreme bounds
    chosen_dim = rng.integers(0, dim, size=n_points)
    high_side = rng.choice([True, False], size=n_points)
    y[np.arange(n_points), chosen_dim] = np.where(high_side, moneyness_half_width, -moneyness_half_width)

    S = mkt["S0"][None, :] * np.exp(y)
    basket_spot = np.dot(S, mkt["weights"])

    if mkt.get("option_type", "call") == "call":
        target = np.where(
            high_side,
            np.maximum(basket_spot - mkt["K"] * np.exp(-mkt["r"] * tau), 0.0),
            0.0
        )
    else:
        target = np.where(
            high_side,
            0.0,
            np.maximum(mkt["K"] * np.exp(-mkt["r"] * tau) - basket_spot, 0.0)
        )
    return S, tau, target


def to_tensors(
    S: np.ndarray,
    tau: np.ndarray,
    K: float,
    device: Union[str, torch.device] = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert spot prices and maturities into PyTorch tensors in log-moneyness coordinates."""
    S_arr = np.asarray(S, dtype=np.float32)
    tau_arr = np.asarray(tau, dtype=np.float32)
    if S_arr.ndim == 1:
        S_arr = S_arr[np.newaxis, :]
    tau_arr = np.atleast_1d(tau_arr)
    y_arr = np.log(S_arr) - np.log(float(K))
    y_t = torch.tensor(y_arr, dtype=torch.float32, device=device, requires_grad=True)
    tau_t = torch.tensor(tau_arr, dtype=torch.float32, device=device, requires_grad=True).unsqueeze(-1)
    return y_t, tau_t


def price_to_tensor(
    prices: np.ndarray,
    device: Union[str, torch.device] = "cpu"
) -> torch.Tensor:
    """Convert option prices into a PyTorch tensor vector."""
    p_arr = np.asarray(prices, dtype=np.float32)
    p_arr = np.atleast_1d(p_arr)
    return torch.tensor(p_arr, dtype=torch.float32, device=device).unsqueeze(-1)


# Aliases preserving 100% backward compatibility with notebooks/pinn_story.ipynb
make_labeled_dataset_in_box = make_extrapolation_labeled_dataset
make_collocation_points_full_domain = make_collocation_points
make_boundary_points_full_domain = make_boundary_points

