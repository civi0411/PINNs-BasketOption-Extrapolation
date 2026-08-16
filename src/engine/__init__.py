"""
engine package
==============
Optimization routines and hybrid objective computation engines for neural models.
"""

from .trainer import train_vanilla_nn, train_pinn, predict

__all__ = ["train_vanilla_nn", "train_pinn", "predict"]
