"""
pinn_basket package (src)
=========================
Physics-Informed Neural Networks for High-Dimensional Multi-Asset Basket Options.
Organized into standard quantitative machine learning packages:
    - models/      : Deep learning network architectures & Black-Scholes PDE losses
    - baselines/   : Finite-difference solvers & CRR binomial trees
    - data/        : Domain collocation point generators & exact Monte Carlo pricer
    - engine/      : Optimization routines & physics-informed training loops
    - evaluation/  : Benchmark experiment drivers & publication plotting suite
"""

from . import config, models, baselines, data, engine, evaluation

__all__ = ["config", "models", "baselines", "data", "engine", "evaluation"]
