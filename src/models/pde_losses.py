"""
pde_losses.py
=============
Autograd differential operators calculating exact Black-Scholes partial differential equation
residuals in log-price coordinates y_i = ln(S_i).
"""

import torch
import torch.nn as nn


def pde_residual(
    model: nn.Module,
    y: torch.Tensor,
    tau: torch.Tensor,
    r: float,
    sigma: torch.Tensor,
    corr: torch.Tensor
) -> torch.Tensor:
    """
    Compute the Black-Scholes PDE residual via automatic differentiation.

    Governing PDE in log-price coordinates y = ln(S) with time-to-maturity tau = T - t:
        du/dtau - sum_i (r - 0.5 * sigma_i^2) * du/dy_i
        - 0.5 * sum_{i,j} rho_{i,j} * sigma_i * sigma_j * d2u/(dy_i dy_j)
        + r * u = 0

    Args:
        model: Neural network model instance evaluated at (y, tau).
        y: Tensor of shape [batch, dim] of log-moneyness coordinates.
        tau: Tensor of shape [batch, 1] of time-to-maturity coordinates.
        r: Risk-free rate scalar.
        sigma: Tensor of shape [dim] of asset volatilities.
        corr: Tensor of shape [dim, dim] of asset correlation matrix.

    Returns:
        Tensor of shape [batch, 1] representing the exact PDE residual.
    """
    if not y.requires_grad:
        y = y.clone().detach().requires_grad_(True)
    if not tau.requires_grad:
        tau = tau.clone().detach().requires_grad_(True)

    u = model(y, tau)
    ones = torch.ones_like(u)

    # First derivatives
    du_dtau = torch.autograd.grad(u, tau, grad_outputs=ones, create_graph=True)[0]
    du_dy = torch.autograd.grad(u, y, grad_outputs=ones, create_graph=True)[0]

    # Drift term
    drift_coeff = r - 0.5 * (sigma ** 2)
    drift_term = torch.sum(drift_coeff * du_dy, dim=1, keepdim=True)

    # Diffusion terms (diagonal and cross-derivatives)
    dim = y.shape[1]
    diff_term = torch.zeros_like(u)

    for i in range(dim):
        du_dyi = du_dy[:, i:i + 1]
        d2u_dyi2 = torch.autograd.grad(du_dyi, y, grad_outputs=ones, create_graph=True)[0][:, i:i + 1]
        diff_term = diff_term + 0.5 * (sigma[i] ** 2) * d2u_dyi2

        for j in range(i + 1, dim):
            d2u_dyidyj = torch.autograd.grad(du_dyi, y, grad_outputs=ones, create_graph=True)[0][:, j:j + 1]
            cov_ij = corr[i, j] * sigma[i] * sigma[j]
            diff_term = diff_term + cov_ij * d2u_dyidyj

    return du_dtau - drift_term - diff_term + r * u
