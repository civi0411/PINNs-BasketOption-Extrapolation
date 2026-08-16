"""
fdm.py
======
Finite-difference numerical solvers for Black-Scholes partial differential equations.

Highlights the fundamental computational limits ("curse of dimensionality"):
    - FDM1D: Second-order Crank-Nicolson implicit-explicit tridiagonal solver (d=1).
    - FDM2D: Explicit finite-difference solver with cross-derivative terms (d=2).
    - FDM3D+: Not implemented by design (grid scaling O(N^d) creates memory explosion).
"""

from typing import Optional
import numpy as np
from scipy.interpolate import RegularGridInterpolator


class FDM1D:
    """Crank-Nicolson tridiagonal solver for single-asset European options."""

    def __init__(
        self,
        K: float,
        r: float,
        sigma: float,
        T: float,
        weights: Optional[np.ndarray] = None,
        y_min: Optional[float] = None,
        y_max: Optional[float] = None,
        n_space: int = 400,
        n_time: int = 400,
        option_type: str = "call"
    ):
        self.K, self.r, self.sigma, self.T = float(K), float(r), float(sigma), float(T)
        self.option_type = option_type

        if y_min is None:
            y_min = np.log(self.K) - 1.5
        if y_max is None:
            y_max = np.log(self.K) + 1.5

        self.y_grid = np.linspace(y_min, y_max, n_space)
        self.tau_grid = np.linspace(0.0, self.T, n_time)
        self._solve()

    def _solve(self) -> None:
        y = self.y_grid
        dy = y[1] - y[0]
        dtau = self.tau_grid[1] - self.tau_grid[0]
        n = len(y)
        sigma, r = self.sigma, self.r

        S = np.exp(y)
        if self.option_type == "call":
            u0 = np.maximum(S - self.K, 0.0)
        else:
            u0 = np.maximum(self.K - S, 0.0)

        a = (r - 0.5 * sigma ** 2) / (2.0 * dy)
        b = 0.5 * (sigma ** 2) / (dy ** 2)

        lower = b - a
        diag = -2.0 * b - r
        upper = b + a

        A = np.zeros((n, n), dtype=np.float64)
        for i in range(1, n - 1):
            A[i, i - 1] = lower
            A[i, i] = diag
            A[i, i + 1] = upper

        I = np.eye(n, dtype=np.float64)
        M1 = I - 0.5 * dtau * A
        M2 = I + 0.5 * dtau * A
        M1_inv = np.linalg.inv(M1)

        u = u0.copy()
        grid = np.zeros((len(self.tau_grid), n), dtype=np.float64)
        grid[0] = u

        for k in range(1, len(self.tau_grid)):
            tau = self.tau_grid[k]
            rhs = M2 @ u
            u_new = M1_inv @ rhs
            if self.option_type == "call":
                u_new[0] = 0.0
                u_new[-1] = S[-1] - self.K * np.exp(-r * tau)
            else:
                u_new[0] = self.K * np.exp(-r * tau) - S[0]
                u_new[-1] = 0.0
            u = u_new
            grid[k] = u

        self.grid = grid
        self._interp = RegularGridInterpolator(
            (self.tau_grid, self.y_grid), self.grid,
            bounds_error=False, fill_value=None
        )

    def price(self, S: np.ndarray, tau: np.ndarray) -> np.ndarray:
        S_arr = np.asarray(S, dtype=np.float64)
        tau_arr = np.asarray(tau, dtype=np.float64)
        y_arr = np.log(S_arr)
        tau_c = np.clip(tau_arr, self.tau_grid[0], self.tau_grid[-1])
        y_c = np.clip(y_arr, self.y_grid[0], self.y_grid[-1])
        tau_c, y_c = np.broadcast_arrays(tau_c, y_c)
        pts = np.stack([tau_c, y_c], axis=-1)
        return self._interp(pts)


class FDM2D:
    """Explicit 2D finite-difference solver for 2-asset correlated baskets."""

    def __init__(
        self,
        K: float,
        r: float,
        sigma: np.ndarray,
        corr: np.ndarray,
        weights: np.ndarray,
        T: float,
        y_min: Optional[float] = None,
        y_max: Optional[float] = None,
        n_space: int = 70,
        n_time: int = 4000,
        option_type: str = "call"
    ):
        assert len(sigma) == 2, "FDM2D solver strictly requires exactly 2 assets."
        self.K, self.r, self.sigma, self.corr, self.weights, self.T = float(K), float(r), sigma, corr, weights, float(T)
        self.option_type = option_type

        if y_min is None:
            y_min = np.log(self.K) - 1.2
        if y_max is None:
            y_max = np.log(self.K) + 1.2

        self.y_grid = np.linspace(y_min, y_max, n_space)
        self.n_time = n_time
        self._solve()

    def _solve(self) -> None:
        y = self.y_grid
        dy = y[1] - y[0]
        n = len(y)
        Y1, Y2 = np.meshgrid(y, y, indexing="ij")
        S1, S2 = np.exp(Y1), np.exp(Y2)

        w = self.weights
        basket = w[0] * S1 + w[1] * S2
        if self.option_type == "call":
            u = np.maximum(basket - self.K, 0.0)
        else:
            u = np.maximum(self.K - basket, 0.0)

        r = self.r
        sig1, sig2 = self.sigma[0], self.sigma[1]
        rho = self.corr[0, 1]

        drift1 = r - 0.5 * sig1 ** 2
        drift2 = r - 0.5 * sig2 ** 2

        max_sig2 = max(sig1 ** 2, sig2 ** 2)
        dtau_stable = 0.4 * (dy ** 2) / max_sig2
        n_time = max(self.n_time, int(np.ceil(self.T / dtau_stable)) + 1)
        dtau = self.T / float(n_time)
        self.tau_grid = np.linspace(0.0, self.T, n_time + 1)

        n_store = min(len(self.tau_grid), 400)
        store_idx = np.unique(np.linspace(0, len(self.tau_grid) - 1, n_store).astype(int))
        self._stored_tau = self.tau_grid[store_idx]
        stored_grids = np.zeros((len(store_idx), n, n), dtype=np.float64)
        store_pos = 0
        if store_idx[0] == 0:
            stored_grids[0] = u
            store_pos = 1

        for step in range(1, len(self.tau_grid)):
            tau = self.tau_grid[step]
            u_new = u.copy()

            du_dy1 = (u[2:, 1:-1] - u[:-2, 1:-1]) / (2.0 * dy)
            du_dy2 = (u[1:-1, 2:] - u[1:-1, :-2]) / (2.0 * dy)
            d2u_dy1 = (u[2:, 1:-1] - 2.0 * u[1:-1, 1:-1] + u[:-2, 1:-1]) / (dy ** 2)
            d2u_dy2 = (u[1:-1, 2:] - 2.0 * u[1:-1, 1:-1] + u[1:-1, :-2]) / (dy ** 2)
            d2u_cross = (u[2:, 2:] - u[2:, :-2] - u[:-2, 2:] + u[:-2, :-2]) / (4.0 * (dy ** 2))

            rhs = (
                drift1 * du_dy1 + drift2 * du_dy2
                + 0.5 * (sig1 ** 2) * d2u_dy1 + 0.5 * (sig2 ** 2) * d2u_dy2
                + rho * sig1 * sig2 * d2u_cross
                - r * u[1:-1, 1:-1]
            )

            u_new[1:-1, 1:-1] = u[1:-1, 1:-1] + dtau * rhs

            if self.option_type == "call":
                u_new[0, :] = 0.0
                u_new[-1, :] = w[0] * S1[-1, :] + w[1] * S2[-1, :] - self.K * np.exp(-r * tau)
                u_new[:, 0] = 0.0
                u_new[:, -1] = w[0] * S1[:, -1] + w[1] * S2[:, -1] - self.K * np.exp(-r * tau)
            else:
                u_new[0, :] = self.K * np.exp(-r * tau)
                u_new[-1, :] = 0.0
                u_new[:, 0] = self.K * np.exp(-r * tau)
                u_new[:, -1] = 0.0

            u = np.maximum(u_new, 0.0)
            if step in store_idx:
                stored_grids[store_pos] = u
                store_pos += 1

        self.grid = stored_grids
        self._interp = RegularGridInterpolator(
            (self._stored_tau, self.y_grid, self.y_grid), self.grid,
            bounds_error=False, fill_value=None
        )

    def price(self, S: np.ndarray, tau: np.ndarray) -> np.ndarray:
        S_arr = np.asarray(S, dtype=np.float64)
        tau_arr = np.asarray(tau, dtype=np.float64)
        y_arr = np.log(S_arr)
        tau_c = np.clip(tau_arr, self._stored_tau[0], self._stored_tau[-1])
        y_c = np.clip(y_arr, self.y_grid[0], self.y_grid[-1])
        tau_b, y1_b = np.broadcast_arrays(tau_c, y_c[:, 0])
        _, y2_b = np.broadcast_arrays(tau_c, y_c[:, 1])
        pts = np.stack([tau_b, y1_b, y2_b], axis=-1)
        return self._interp(pts)
