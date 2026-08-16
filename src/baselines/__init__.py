"""
baselines package
=================
Classical numerical pricing baselines (finite-difference solvers and binomial trees).
"""

from .fdm import FDM1D, FDM2D
from .binomial import crr_price

__all__ = ["FDM1D", "FDM2D", "crr_price"]
