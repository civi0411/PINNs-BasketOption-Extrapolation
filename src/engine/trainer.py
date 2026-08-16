"""
trainer.py
==========
Optimization loops and hybrid loss objectives for deep financial models.

Governing Loss Objectives:
    - VanillaNN : L = L_data
    - PINN      : L = w_data * L_data + w_pde * L_pde + w_ic * L_ic + w_bc * L_bc
    - ICPINN    : L = w_data * L_data + w_pde * L_pde + w_bc * L_bc
                  (where L_ic == 0 exactly by architectural construction)
"""

import time
from typing import Dict, Any, Tuple, List, Union
import numpy as np
import torch
import torch.nn as nn

from config import (
    NET_HIDDEN, NET_LAYERS, EPOCHS_DEFAULT, LR_DEFAULT,
    BATCH_DATA, BATCH_COLLOC, LAMBDA_DATA, LAMBDA_PDE,
    LAMBDA_IC, LAMBDA_BC
)
from models.networks import VanillaNN, PINN, ICPINN
from models.pde_losses import pde_residual
from data.samplers import (
    make_collocation_points, make_boundary_points,
    to_tensors, price_to_tensor
)
from data.monte_carlo import basket_payoff

MSE = nn.MSELoss()


def _batch_indices(n: int, batch_size: int, rng: np.random.Generator) -> np.ndarray:
    """Sample uniform mini-batch indices without replacement."""
    if n <= batch_size:
        return np.arange(n)
    return rng.choice(n, size=batch_size, replace=False)


def train_vanilla_nn(
    mkt: Dict[str, Any],
    S_train: np.ndarray,
    tau_train: np.ndarray,
    price_train: np.ndarray,
    epochs: int = EPOCHS_DEFAULT,
    lr: float = LR_DEFAULT,
    batch_size: int = BATCH_DATA,
    seed: int = 0,
    verbose: bool = False,
    device: Union[str, torch.device] = "cpu"
) -> Tuple[VanillaNN, float, List[Tuple[int, float]]]:
    """Train a pure data-driven Multi-Layer Perceptron (VanillaNN) on empirical quotes."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dim = mkt["dim"]

    model = VanillaNN(dim, hidden=NET_HIDDEN, layers=NET_LAYERS).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    y_all, tau_all = to_tensors(S_train, tau_train, mkt["K"], device=device)
    price_all = price_to_tensor(price_train, device=device)
    n = y_all.shape[0]

    history: List[Tuple[int, float]] = []
    t0 = time.perf_counter()

    for ep in range(epochs):
        idx = _batch_indices(n, batch_size, rng)
        y_b, tau_b, p_b = y_all[idx], tau_all[idx], price_all[idx]

        u_pred = model(y_b, tau_b)
        loss = MSE(u_pred, p_b)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if verbose and ep % max(1, epochs // 5) == 0:
            history.append((ep, float(loss.item())))

    train_time = time.perf_counter() - t0
    return model, train_time, history


def train_pinn(
    mkt: Dict[str, Any],
    S_train: np.ndarray,
    tau_train: np.ndarray,
    price_train: np.ndarray,
    model_type: str = "pinn",
    epochs: int = EPOCHS_DEFAULT,
    lr: float = LR_DEFAULT,
    batch_size_data: int = BATCH_DATA,
    batch_size_colloc: int = BATCH_COLLOC,
    seed: int = 0,
    verbose: bool = False,
    lambda_data: float = LAMBDA_DATA,
    lambda_pde: float = LAMBDA_PDE,
    lambda_ic: float = LAMBDA_IC,
    lambda_bc: float = LAMBDA_BC,
    device: Union[str, torch.device] = "cpu"
) -> Tuple[nn.Module, float, List[Tuple[int, float, float, float]]]:
    """Train a Physics-Informed Neural Network (PINN or ICPINN) minimizing hybrid loss."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dim = mkt["dim"]

    if model_type.lower() == "pinn":
        model = PINN(dim, hidden=NET_HIDDEN, layers=NET_LAYERS).to(device)
    elif model_type.lower() == "icpinn":
        model = ICPINN(
            dim, K=mkt["K"], weights=mkt["weights"],
            option_type=mkt.get("option_type", "call"),
            hidden=NET_HIDDEN, layers=NET_LAYERS
        ).to(device)
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Must be 'pinn' or 'icpinn'.")

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sigma_t = torch.tensor(mkt["sigma"], dtype=torch.float32, device=device)
    corr_t = torch.tensor(mkt["corr"], dtype=torch.float32, device=device)
    r = mkt["r"]

    y_data, tau_data = to_tensors(S_train, tau_train, mkt["K"], device=device)
    price_data = price_to_tensor(price_train, device=device)
    n_data = y_data.shape[0]

    S_col, tau_col = make_collocation_points(mkt, n_points=max(2000, batch_size_colloc * 8), seed=seed)
    y_col_all, tau_col_all = to_tensors(S_col, tau_col, mkt["K"], device=device)
    n_col = y_col_all.shape[0]

    S_ic, _ = make_collocation_points(mkt, n_points=512, seed=seed + 1)
    y_ic_all, _ = to_tensors(S_ic, np.zeros(S_ic.shape[0]), mkt["K"], device=device)
    tau_ic_all = torch.zeros_like(y_ic_all[:, :1])
    payoff_ic_np = basket_payoff(S_ic, mkt["K"], mkt["weights"], mkt.get("option_type", "call"))
    payoff_ic = torch.tensor(payoff_ic_np, dtype=torch.float32, device=device).unsqueeze(-1)

    S_bc, tau_bc, target_bc = make_boundary_points(mkt, n_points=512, seed=seed + 2)
    y_bc_all, tau_bc_all = to_tensors(S_bc, tau_bc, mkt["K"], device=device)
    target_bc_all = price_to_tensor(target_bc, device=device)

    history: List[Tuple[int, float, float, float]] = []
    t0 = time.perf_counter()

    for ep in range(epochs):
        idx_d = _batch_indices(n_data, batch_size_data, rng)
        y_d, tau_d, p_d = y_data[idx_d], tau_data[idx_d], price_data[idx_d]

        idx_c = _batch_indices(n_col, batch_size_colloc, rng)
        y_c, tau_c = y_col_all[idx_c], tau_col_all[idx_c]

        idx_ic = _batch_indices(y_ic_all.shape[0], 64, rng)
        idx_bc = _batch_indices(y_bc_all.shape[0], 64, rng)

        u_pred_data = model(y_d, tau_d)
        data_loss = MSE(u_pred_data, p_d)

        residual = pde_residual(model, y_c, tau_c, r, sigma_t, corr_t)
        pde_loss = (residual ** 2).mean()

        if model_type.lower() == "icpinn":
            ic_loss = torch.tensor(0.0, device=device)
        else:
            u_ic_pred = model(y_ic_all[idx_ic], tau_ic_all[idx_ic])
            ic_loss = MSE(u_ic_pred, payoff_ic[idx_ic])

        u_bc_pred = model(y_bc_all[idx_bc], tau_bc_all[idx_bc])
        bc_loss = MSE(u_bc_pred, target_bc_all[idx_bc])

        loss = (
            lambda_data * data_loss
            + lambda_pde * pde_loss
            + lambda_ic * ic_loss
            + lambda_bc * bc_loss
        )

        opt.zero_grad()
        loss.backward()
        opt.step()

        if verbose and ep % max(1, epochs // 5) == 0:
            history.append((ep, float(loss.item()), float(data_loss.item()), float(pde_loss.item())))

    train_time = time.perf_counter() - t0
    return model, train_time, history


@torch.no_grad()
def predict(
    model: nn.Module,
    S: np.ndarray,
    tau: np.ndarray,
    mkt: Dict[str, Any],
    device: Union[str, torch.device] = "cpu"
) -> np.ndarray:
    """Vectorized inference query mapping NumPy arrays to option prices."""
    model.eval()
    y, t = to_tensors(S, tau, mkt["K"], device=device)
    preds = model(y, t).squeeze(-1)
    if preds.device.type != "cpu":
        preds = preds.cpu()
    return preds.numpy()
