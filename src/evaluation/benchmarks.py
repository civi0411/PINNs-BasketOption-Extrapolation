"""
benchmarks.py
=============
Execution routines for the five benchmark experiments of the quantitative study.
Each returns a structured pandas DataFrame ready for tabular reporting and plotting.
"""

import time
from typing import Dict, Any, Tuple, List, Optional, Union
import numpy as np
import pandas as pd

from config import (
    make_market, get_device, EXPERIMENT_DIMS, NOISE_LEVELS, SPARSITY_FRACS,
    FIXED_DIM_FOR_NOISE_SPARSITY, N_TRAIN_DEFAULT, N_TEST_DEFAULT,
    MC_PATHS_TRAIN_LABEL, MC_PATHS_TEST_LABEL, EPOCHS_DEFAULT,
    TRAIN_LOG_MONEY_HALF_WIDTH
)
from data.samplers import (
    make_labeled_dataset, make_extrapolation_labeled_dataset, sample_domain_points
)
from data.monte_carlo import mc_basket_price_batch_with_se
from engine.trainer import train_vanilla_nn, train_pinn, predict
from baselines.fdm import FDM1D, FDM2D
from baselines.binomial import crr_price


def _make_test_set(
    mkt: Dict[str, Any],
    n_test: int,
    seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate high-precision Monte Carlo ground-truth test points over the evaluation continuum."""
    rng = np.random.default_rng(seed)
    S_te, tau_te = sample_domain_points(mkt, n_test, rng, tau_min_frac=0.05)
    price_te, _ = mc_basket_price_batch_with_se(
        S_te, tau_te, mkt["K"], mkt["r"], mkt["sigma"], mkt["corr"], mkt["weights"],
        option_type=mkt.get("option_type", "call"),
        n_samples=MC_PATHS_TEST_LABEL, rng=rng
    )
    return S_te, tau_te, price_te


def _rel_error(pred: np.ndarray, true: np.ndarray) -> float:
    """Compute aggregate normalized relative error."""
    true_arr = np.asarray(true, dtype=np.float64)
    pred_arr = np.asarray(pred, dtype=np.float64)
    mae = np.mean(np.abs(pred_arr - true_arr))
    scale = np.maximum(np.mean(np.abs(true_arr)), 1e-6)
    return float(mae / scale)


def _mae(pred: np.ndarray, true: np.ndarray) -> float:
    """Compute Mean Absolute Error across the queried sample points."""
    return float(np.mean(np.abs(np.asarray(pred) - np.asarray(true))))


# ------------------------------------------------------------------
# Experiment 1: Accuracy vs Dimension
# ------------------------------------------------------------------
def run_experiment_1_dimension(
    dims: List[int] = EXPERIMENT_DIMS,
    n_train: int = N_TRAIN_DEFAULT,
    n_test: int = N_TEST_DEFAULT,
    epochs: int = EPOCHS_DEFAULT,
    seed: int = 10,
    verbose: bool = True
) -> Tuple[pd.DataFrame, Dict[Tuple[int, str], Any]]:
    """Evaluate pricing accuracy scaling across dimensions d in {1, 2, 3, 5}."""
    device = get_device()
    rows: List[Dict[str, Any]] = []
    trained_models: Dict[Tuple[int, str], Any] = {}

    for dim in dims:
        mkt = make_market(dim, seed=seed + dim)
        mkt["option_type"] = "call"
        if verbose:
            print(f"\n[Exp1] Dimension d={dim} -- Building training & evaluation sets...")

        S_tr, tau_tr, price_tr = make_labeled_dataset(
            mkt, n_points=n_train, mc_paths=MC_PATHS_TRAIN_LABEL, seed=seed
        )
        S_te, tau_te, price_te = _make_test_set(mkt, n_test, seed=seed + 1000)

        for model_type, trainer in [
            ("VanillaNN", lambda: train_vanilla_nn(mkt, S_tr, tau_tr, price_tr, epochs=epochs, seed=seed, device=device)),
            ("PINN", lambda: train_pinn(mkt, S_tr, tau_tr, price_tr, model_type="pinn", epochs=epochs, seed=seed, device=device)),
            ("ICPINN", lambda: train_pinn(mkt, S_tr, tau_tr, price_tr, model_type="icpinn", epochs=epochs, seed=seed, device=device)),
        ]:
            t0 = time.perf_counter()
            model, train_time, _ = trainer()
            t0i = time.perf_counter()
            pred = predict(model, S_te, tau_te, mkt, device=device)
            infer_time = time.perf_counter() - t0i

            rows.append({
                "experiment": "dimension", "dim": dim, "method": model_type,
                "rel_error": _rel_error(pred, price_te), "mae": _mae(pred, price_te),
                "train_time_s": train_time, "infer_time_s": infer_time, "n_test": n_test,
            })
            trained_models[(dim, model_type)] = model
            if verbose:
                print(f"    {model_type:12s} | RelErr={rows[-1]['rel_error']:.4f} | MAE={rows[-1]['mae']:.4f} | Train={train_time:.1f}s")

        rng = np.random.default_rng(seed + 2000)
        t0 = time.perf_counter()
        mc_pred, _ = mc_basket_price_batch_with_se(
            S_te, tau_te, mkt["K"], mkt["r"], mkt["sigma"], mkt["corr"], mkt["weights"],
            option_type="call", n_samples=8000, rng=rng
        )
        mc_time = time.perf_counter() - t0
        rows.append({
            "experiment": "dimension", "dim": dim, "method": "MonteCarlo",
            "rel_error": _rel_error(mc_pred, price_te), "mae": _mae(mc_pred, price_te),
            "train_time_s": 0.0, "infer_time_s": mc_time, "n_test": n_test,
        })
        if verbose:
            print(f"    {'MonteCarlo':12s} | RelErr={rows[-1]['rel_error']:.4f} | MAE={rows[-1]['mae']:.4f} | Infer={mc_time*1000:.1f}ms")

        if dim == 1:
            t0 = time.perf_counter()
            fdm = FDM1D(K=mkt["K"], r=mkt["r"], sigma=mkt["sigma"][0], T=mkt["T"], n_space=300, n_time=300)
            fdm_build_time = time.perf_counter() - t0
            t0 = time.perf_counter()
            fdm_pred = fdm.price(S_te[:, 0], tau_te)
            fdm_infer_time = time.perf_counter() - t0
            rows.append({
                "experiment": "dimension", "dim": dim, "method": "FDM",
                "rel_error": _rel_error(fdm_pred, price_te), "mae": _mae(fdm_pred, price_te),
                "train_time_s": fdm_build_time, "infer_time_s": fdm_infer_time, "n_test": n_test,
            })

            t0 = time.perf_counter()
            bin_pred = crr_price(S_te[:, 0], mkt["K"], mkt["r"], mkt["sigma"][0], tau_te, n_steps=250)
            bin_time = time.perf_counter() - t0
            rows.append({
                "experiment": "dimension", "dim": dim, "method": "BinomialTree",
                "rel_error": _rel_error(bin_pred, price_te), "mae": _mae(bin_pred, price_te),
                "train_time_s": 0.0, "infer_time_s": bin_time, "n_test": n_test,
            })
            if verbose:
                print(f"    {'FDM (1D)':12s} | RelErr={rows[-2]['rel_error']:.4f} | Build={fdm_build_time:.2f}s")
                print(f"    {'BinomialTree':12s} | RelErr={rows[-1]['rel_error']:.4f} | Time={bin_time:.2f}s")

        elif dim == 2:
            t0 = time.perf_counter()
            fdm = FDM2D(K=mkt["K"], r=mkt["r"], sigma=mkt["sigma"], corr=mkt["corr"],
                        weights=mkt["weights"], T=mkt["T"], n_space=70)
            fdm_build_time = time.perf_counter() - t0
            t0 = time.perf_counter()
            fdm_pred = fdm.price(S_te, tau_te)
            fdm_infer_time = time.perf_counter() - t0
            rows.append({
                "experiment": "dimension", "dim": dim, "method": "FDM",
                "rel_error": _rel_error(fdm_pred, price_te), "mae": _mae(fdm_pred, price_te),
                "train_time_s": fdm_build_time, "infer_time_s": fdm_infer_time, "n_test": n_test,
            })
            if verbose:
                print(f"    {'FDM (2D)':12s} | RelErr={rows[-1]['rel_error']:.4f} | Build={fdm_build_time:.2f}s")

    return pd.DataFrame(rows), trained_models


# ------------------------------------------------------------------
# Experiment 2: Robustness to Microstructure Noise
# ------------------------------------------------------------------
def run_experiment_2_noise(
    dim: int = FIXED_DIM_FOR_NOISE_SPARSITY,
    noise_levels: List[float] = NOISE_LEVELS,
    n_train: int = N_TRAIN_DEFAULT,
    n_test: int = N_TEST_DEFAULT,
    epochs: int = EPOCHS_DEFAULT,
    seed: int = 20,
    verbose: bool = True
) -> pd.DataFrame:
    """Evaluate pricing stability under corrupted training quotes."""
    device = get_device()
    mkt = make_market(dim, seed=seed)
    mkt["option_type"] = "call"
    S_te, tau_te, price_te = _make_test_set(mkt, n_test, seed=seed + 1000)

    rows: List[Dict[str, Any]] = []
    for noise in noise_levels:
        S_tr, tau_tr, price_tr = make_labeled_dataset(
            mkt, n_points=n_train, mc_paths=MC_PATHS_TRAIN_LABEL, noise_std=noise, seed=seed
        )

        for model_type, trainer in [
            ("VanillaNN", lambda: train_vanilla_nn(mkt, S_tr, tau_tr, price_tr, epochs=epochs, seed=seed, device=device)),
            ("PINN", lambda: train_pinn(mkt, S_tr, tau_tr, price_tr, model_type="pinn", epochs=epochs, seed=seed, device=device)),
        ]:
            model, _, _ = trainer()
            pred = predict(model, S_te, tau_te, mkt, device=device)
            rows.append({
                "experiment": "noise", "dim": dim, "noise": noise, "method": model_type,
                "rel_error": _rel_error(pred, price_te), "mae": _mae(pred, price_te),
            })
            if verbose:
                print(f"[Exp2] Noise={noise:.2f} | {model_type:10s} | RelErr={rows[-1]['rel_error']:.4f}")

    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# Experiment 3: Extreme Data Sparsity
# ------------------------------------------------------------------
def run_experiment_3_sparsity(
    dim: int = FIXED_DIM_FOR_NOISE_SPARSITY,
    fracs: List[float] = SPARSITY_FRACS,
    n_train_full: int = N_TRAIN_DEFAULT,
    n_test: int = N_TEST_DEFAULT,
    epochs: int = EPOCHS_DEFAULT,
    seed: int = 30,
    verbose: bool = True
) -> pd.DataFrame:
    """Evaluate generalization when labeled quote availability is scarce."""
    device = get_device()
    mkt = make_market(dim, seed=seed)
    mkt["option_type"] = "call"
    S_te, tau_te, price_te = _make_test_set(mkt, n_test, seed=seed + 1000)

    rows: List[Dict[str, Any]] = []
    for frac in fracs:
        n_points = max(20, int(round(n_train_full * frac)))
        S_tr, tau_tr, price_tr = make_labeled_dataset(
            mkt, n_points=n_points, mc_paths=MC_PATHS_TRAIN_LABEL, seed=seed
        )

        for model_type, trainer in [
            ("VanillaNN", lambda: train_vanilla_nn(mkt, S_tr, tau_tr, price_tr, epochs=epochs, seed=seed, batch_size=min(128, n_points), device=device)),
            ("PINN", lambda: train_pinn(mkt, S_tr, tau_tr, price_tr, model_type="pinn", epochs=epochs, seed=seed, batch_size_data=min(128, n_points), device=device)),
        ]:
            model, _, _ = trainer()
            pred = predict(model, S_te, tau_te, mkt, device=device)
            rows.append({
                "experiment": "sparsity", "dim": dim, "frac": frac, "n_points": n_points, "method": model_type,
                "rel_error": _rel_error(pred, price_te), "mae": _mae(pred, price_te),
            })
            if verbose:
                print(f"[Exp3] Frac={frac:.2f} (N={n_points:3d}) | {model_type:10s} | RelErr={rows[-1]['rel_error']:.4f}")

    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# Experiment 4: Sub-Millisecond Inference Speedup
# ------------------------------------------------------------------
def run_experiment_4_speed(
    trained_models: Dict[Tuple[int, str], Any],
    dims: List[int] = EXPERIMENT_DIMS,
    n_query: int = 2000,
    seed: int = 40,
    verbose: bool = True
) -> pd.DataFrame:
    """Benchmark neural network inference latency against vectorized Monte Carlo simulation."""
    device = get_device()
    rows: List[Dict[str, Any]] = []

    for dim in dims:
        mkt = make_market(dim, seed=seed + dim)
        mkt["option_type"] = "call"
        rng = np.random.default_rng(seed)
        S_q, tau_q = sample_domain_points(mkt, n_query, rng)

        for model_type in ["VanillaNN", "PINN", "ICPINN"]:
            model = trained_models.get((dim, model_type))
            if model is None:
                continue
            t0 = time.perf_counter()
            _ = predict(model, S_q, tau_q, mkt, device=device)
            dt = time.perf_counter() - t0
            rows.append({
                "experiment": "speed", "dim": dim, "method": model_type,
                "n_query": n_query, "total_time_s": dt, "per_query_ms": (dt / n_query) * 1000.0
            })

        rng = np.random.default_rng(seed + 1)
        t0 = time.perf_counter()
        _, _ = mc_basket_price_batch_with_se(
            S_q, tau_q, mkt["K"], mkt["r"], mkt["sigma"], mkt["corr"], mkt["weights"],
            option_type="call", n_samples=8000, rng=rng
        )
        dt = time.perf_counter() - t0
        rows.append({
            "experiment": "speed", "dim": dim, "method": "MonteCarlo(8k paths)",
            "n_query": n_query, "total_time_s": dt, "per_query_ms": (dt / n_query) * 1000.0
        })
        if verbose:
            print(f"[Exp4] Dimension d={dim} | Speed benchmarks completed.")

    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# Experiment 5: Domain Extrapolation (Out-Of-Distribution Tail-Risk)
# ------------------------------------------------------------------
def run_experiment_5_extrapolation(
    dims: List[int] = EXPERIMENT_DIMS,
    n_train: int = N_TRAIN_DEFAULT,
    n_test: int = 3000,
    epochs: int = EPOCHS_DEFAULT,
    moneyness_bound: float = TRAIN_LOG_MONEY_HALF_WIDTH,
    seed: int = 50,
    verbose: bool = True
) -> Tuple[pd.DataFrame, Dict[Tuple[int, str], Any]]:
    """Evaluate structural domain extrapolation capability outside observable training data."""
    device = get_device()
    rows: List[Dict[str, Any]] = []
    extrap_models: Dict[Tuple[int, str], Any] = {}

    for dim in dims:
        mkt = make_market(dim, seed=seed + dim)
        mkt["option_type"] = "call"
        if verbose:
            print(f"\n[Exp5] Dimension d={dim} -- Evaluating Domain Extrapolation & Tail-Risk...")

        S_tr, tau_tr, price_tr = make_extrapolation_labeled_dataset(
            mkt, n_points=n_train, mc_paths=MC_PATHS_TRAIN_LABEL,
            moneyness_bound=moneyness_bound, seed=seed
        )
        S_te, tau_te, price_te = _make_test_set(mkt, n_test, seed=seed + 1000)

        y_te = np.log(S_te) - np.log(mkt["S0"])[None, :]
        is_interp = np.all(np.abs(y_te) <= moneyness_bound, axis=1)
        is_extrap = ~is_interp

        for model_type, trainer in [
            ("VanillaNN", lambda: train_vanilla_nn(mkt, S_tr, tau_tr, price_tr, epochs=epochs, seed=seed, device=device)),
            ("PINN", lambda: train_pinn(mkt, S_tr, tau_tr, price_tr, model_type="pinn", epochs=epochs, seed=seed, device=device)),
            ("ICPINN", lambda: train_pinn(mkt, S_tr, tau_tr, price_tr, model_type="icpinn", epochs=epochs, seed=seed, device=device)),
        ]:
            t0 = time.perf_counter()
            model, train_time, _ = trainer()
            pred = predict(model, S_te, tau_te, mkt, device=device)

            for zone_name, mask in [
                ("interpolation", is_interp),
                ("extrapolation", is_extrap),
                ("overall", np.ones_like(is_interp, dtype=bool))
            ]:
                if np.sum(mask) == 0:
                    continue
                rows.append({
                    "experiment": "extrapolation", "dim": dim, "method": model_type, "zone": zone_name,
                    "rel_error": _rel_error(pred[mask], price_te[mask]),
                    "mae": _mae(pred[mask], price_te[mask]),
                    "train_time_s": train_time, "n_test": int(np.sum(mask))
                })
            extrap_models[(dim, model_type)] = model
            if verbose:
                err_in = _rel_error(pred[is_interp], price_te[is_interp]) if np.any(is_interp) else 0.0
                err_out = _rel_error(pred[is_extrap], price_te[is_extrap]) if np.any(is_extrap) else 0.0
                print(f"    {model_type:10s} | Interp (NTM)={err_in:.4f} | Extrap (OOD)={err_out:.4f} | Train={train_time:.1f}s")

    return pd.DataFrame(rows), extrap_models


# Alias preserving 100% backward compatibility with notebooks/pinn_story.ipynb
run_experiment_extrapolation = run_experiment_5_extrapolation

