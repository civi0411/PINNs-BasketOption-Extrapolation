"""
networks.py
===========
Neural network architectures for multi-asset European basket options.

Architectures:
    - VanillaNN : Pure data-driven Multi-Layer Perceptron (MLP).
    - PINN      : Physics-Informed Neural Network with soft PDE residual regularization.
    - ICPINN    : Initial-Condition constrained PINN with hard structural factorization:
                  u(y, tau) = Payoff(y) + (1 - exp(-tau)) * N(y, tau)
                  guaranteeing exact zero error at maturity (tau = 0).
"""

from typing import Union
import torch
import torch.nn as nn
from data.monte_carlo import basket_payoff


class MLP(nn.Module):
    """
    Standard Multi-Layer Perceptron backbone with Tanh activations.
    Employs Xavier/Glorot Normal initialization tailored for Tanh gradients.
    """

    def __init__(self, in_features: int, hidden_features: int = 128, num_layers: int = 4):
        super().__init__()
        layers = []
        curr_in = in_features
        for _ in range(num_layers):
            linear = nn.Linear(curr_in, hidden_features)
            nn.init.xavier_normal_(linear.weight)
            nn.init.zeros_(linear.bias)
            layers.append(linear)
            layers.append(nn.Tanh())
            curr_in = hidden_features

        out_linear = nn.Linear(curr_in, 1)
        nn.init.xavier_normal_(out_linear.weight)
        nn.init.zeros_(out_linear.bias)
        layers.append(out_linear)

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VanillaNN(nn.Module):
    """Pure data-driven neural network mapping log-moneyness y and time-to-maturity tau to option price."""

    def __init__(self, dim: int, hidden: int = 128, layers: int = 4):
        super().__init__()
        self.dim = dim
        self.mlp = MLP(in_features=dim + 1, hidden_features=hidden, num_layers=layers)

    def forward(self, y: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        x = torch.cat([y, tau], dim=-1)
        return self.mlp(x)


class PINN(nn.Module):
    """
    Physics-Informed Neural Network.
    Maps (y, tau) to option price u(y, tau). Optimized via hybrid data and Black-Scholes PDE residual loss.
    """

    def __init__(self, dim: int, hidden: int = 128, layers: int = 4):
        super().__init__()
        self.dim = dim
        self.mlp = MLP(in_features=dim + 1, hidden_features=hidden, num_layers=layers)

    def forward(self, y: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        x = torch.cat([y, tau], dim=-1)
        return self.mlp(x)


class ICPINN(nn.Module):
    """
    Initial Condition constrained PINN (ICPINN).
    Enforces the terminal payoff exactly by architectural construction using the functional ansatz:
        u(y, tau) = Payoff(y) + (1 - exp(-tau)) * N(y, tau)
    When tau -> 0, (1 - exp(-tau)) -> 0, guaranteeing u(y, 0) == Payoff(y) exactly across all dimensions.
    """

    def __init__(
        self,
        dim: int,
        K: float,
        weights: Union[list, torch.Tensor],
        option_type: str = "call",
        hidden: int = 128,
        layers: int = 4
    ):
        super().__init__()
        self.dim = dim
        self.K = float(K)
        self.option_type = option_type
        self.register_buffer("weights", torch.as_tensor(weights, dtype=torch.float32))
        self.mlp = MLP(in_features=dim + 1, hidden_features=hidden, num_layers=layers)

    def forward(self, y: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        x = torch.cat([y, tau], dim=-1)
        n_out = self.mlp(x)

        S = torch.exp(y)
        payoff = basket_payoff(S, self.K, self.weights, self.option_type)
        if payoff.ndim == 1:
            payoff = payoff.unsqueeze(-1)

        gate = 1.0 - torch.exp(-tau)
        return payoff + gate * n_out
