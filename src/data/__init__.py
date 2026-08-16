"""
data package
============
Data sampling utilities, domain collocation point generators, and exact Monte Carlo pricing.
"""

from .monte_carlo import basket_payoff, mc_basket_price_batch_with_se
from .samplers import (
    sample_domain_points, sample_extrapolation_train_points,
    make_labeled_dataset, make_extrapolation_labeled_dataset,
    make_collocation_points, make_boundary_points,
    to_tensors, price_to_tensor,
    make_labeled_dataset_in_box, make_collocation_points_full_domain,
    make_boundary_points_full_domain
)

__all__ = [
    "basket_payoff", "mc_basket_price_batch_with_se",
    "sample_domain_points", "sample_extrapolation_train_points",
    "make_labeled_dataset", "make_extrapolation_labeled_dataset",
    "make_collocation_points", "make_boundary_points",
    "to_tensors", "price_to_tensor",
    "make_labeled_dataset_in_box", "make_collocation_points_full_domain",
    "make_boundary_points_full_domain"
]
