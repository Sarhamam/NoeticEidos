"""Algebraic operations for dual transport modes."""

from .additive import (
    gaussian_kernel,
    gaussian_affinity_matrix,
    heat_kernel,
    diffusion_distance
)
from .multiplicative import (
    log_map,
    poisson_kernel_log,
    haar_measure_weight,
    multiplicative_distance,
    poisson_affinity_matrix,
    multiplicative_heat_kernel,
    log_ratio_distance
)
from .mellin import (
    mellin_transform_discrete,
    mellin_unitarity_test,
    mellin_balance_score,
    analytical_mellin_pairs
)

__all__ = [
    # Additive
    "gaussian_kernel",
    "gaussian_affinity_matrix",
    "heat_kernel",
    "diffusion_distance",
    # Multiplicative
    "log_map",
    "poisson_kernel_log",
    "haar_measure_weight",
    "multiplicative_distance",
    "poisson_affinity_matrix",
    "multiplicative_heat_kernel",
    "log_ratio_distance",
    # Mellin
    "mellin_transform_discrete",
    "mellin_unitarity_test",
    "mellin_balance_score",
    "analytical_mellin_pairs",
]