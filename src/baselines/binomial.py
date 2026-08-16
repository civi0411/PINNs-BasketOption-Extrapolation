"""
binomial.py
===========
Cox-Ross-Rubinstein (CRR) binomial tree option pricer.

Restricted by construction to single-asset (d=1) options: extending binomial
lattices to correlated baskets requires 2^d or (steps+1)^d branching nodes
per time slice, illustrating the "curse of dimensionality" where classical
lattice algorithms become computationally intractable.
"""

from typing import Union
import numpy as np


def crr_price(
    S0: Union[float, np.ndarray],
    K: float,
    r: float,
    sigma: float,
    tau: Union[float, np.ndarray],
    option_type: str = "call",
    n_steps: int = 250
) -> np.ndarray:
    """
    Vectorized-per-query Cox-Ross-Rubinstein (CRR) binomial tree pricer.
    """
    S0_arr = np.atleast_1d(np.asarray(S0, dtype=np.float64))
    tau_arr = np.atleast_1d(np.asarray(tau, dtype=np.float64))
    out = np.zeros_like(S0_arr)

    for idx in range(len(S0_arr)):
        s0, t = S0_arr[idx], max(float(tau_arr[idx]), 1e-8)
        dt = t / float(n_steps)
        u = np.exp(sigma * np.sqrt(dt))
        d = 1.0 / u
        p = (np.exp(r * dt) - d) / (u - d)
        disc = np.exp(-r * dt)

        j = np.arange(n_steps + 1)
        ST = s0 * (u ** (n_steps - j)) * (d ** j)
        if option_type == "call":
            values = np.maximum(ST - K, 0.0)
        else:
            values = np.maximum(K - ST, 0.0)

        for _ in range(n_steps, 0, -1):
            values = disc * (p * values[:-1] + (1 - p) * values[1:])

        out[idx] = values[0]

    return out
