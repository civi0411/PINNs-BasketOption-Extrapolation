"""
monte_carlo.py
==============
Vectorized Monte Carlo pricer with exact numerical standard error (SE) tracking
and multi-asset terminal option payoff evaluation.
"""

from typing import Tuple, Optional, Union
import numpy as np


def basket_payoff(
    S: np.ndarray,
    K: float,
    weights: np.ndarray,
    option_type: str = "call"
) -> np.ndarray:
    """Evaluate exact terminal payoff of a European basket option across sample trajectories."""
    S_arr = np.asarray(S, dtype=np.float64)
    w_arr = np.asarray(weights, dtype=np.float64)
    basket_val = np.dot(S_arr, w_arr)
    if option_type == "call":
        return np.maximum(basket_val - float(K), 0.0)
    elif option_type == "put":
        return np.maximum(float(K) - basket_val, 0.0)
    else:
        raise ValueError(f"Unknown option_type '{option_type}'. Must be 'call' or 'put'.")


def mc_basket_price_batch_with_se(
    S0_array: np.ndarray,
    tau_array: np.ndarray,
    K: float,
    r: float,
    sigma: np.ndarray,
    corr: np.ndarray,
    weights: np.ndarray,
    option_type: str = "call",
    n_samples: int = 60000,
    rng: Optional[np.random.Generator] = None,
    chunk_size: int = 20000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute exact Monte Carlo prices and standard errors for N query coordinates (S0_i, tau_i).
    Employs chunked path evaluation to prevent memory explosion during high-sample evaluation.
    """
    S0_arr = np.asarray(S0_array, dtype=np.float64)
    tau_arr = np.asarray(tau_array, dtype=np.float64)
    if S0_arr.ndim == 1:
        S0_arr = S0_arr[np.newaxis, :]
    tau_arr = np.atleast_1d(tau_arr)

    n_points, dim = S0_arr.shape
    if rng is None:
        rng = np.random.default_rng()

    # Cholesky decomposition of asset correlation matrix with numerical stability safeguard
    corr_arr = np.asarray(corr, dtype=np.float64)
    try:
        L = np.linalg.cholesky(corr_arr)
    except np.linalg.LinAlgError:
        jitter = 1e-10 * np.eye(dim, dtype=np.float64)
        L = np.linalg.cholesky(corr_arr + jitter)

    sigma_arr = np.asarray(sigma, dtype=np.float64)
    drift = (float(r) - 0.5 * (sigma_arr ** 2))

    prices = np.zeros(n_points, dtype=np.float64)
    standard_errors = np.zeros(n_points, dtype=np.float64)

    for i in range(n_points):
        s0_i = S0_arr[i]
        t_i = max(float(tau_arr[i]), 1e-8)
        sqrt_t = np.sqrt(t_i)
        disc = np.exp(-float(r) * t_i)

        sum_payoff = 0.0
        sum_payoff_sq = 0.0
        samples_left = n_samples

        while samples_left > 0:
            batch = min(samples_left, chunk_size)
            Z = rng.normal(size=(batch, dim))
            W = Z @ L.T
            ST = s0_i * np.exp(drift * t_i + sigma_arr * sqrt_t * W)
            pay = basket_payoff(ST, K, weights, option_type)
            sum_payoff += np.sum(pay)
            sum_payoff_sq += np.sum(pay ** 2)
            samples_left -= batch

        mean_pay = sum_payoff / float(n_samples)
        var_pay = max(0.0, (sum_payoff_sq / float(n_samples)) - (mean_pay ** 2))
        prices[i] = disc * mean_pay
        standard_errors[i] = disc * np.sqrt(var_pay / float(n_samples))

    return prices, standard_errors
