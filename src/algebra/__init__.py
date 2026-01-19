"""Algebraic operations for dual transport modes."""

from .additive import (
    diffusion_distance,
    gaussian_affinity_matrix,
    gaussian_kernel,
    heat_kernel,
)
from .mellin import (
    analytical_mellin_pairs,
    mellin_balance_score,
    mellin_transform_discrete,
    mellin_unitarity_test,
)
from .multiplicative import (
    haar_measure_weight,
    log_map,
    log_ratio_distance,
    multiplicative_distance,
    multiplicative_heat_kernel,
    poisson_affinity_matrix,
    poisson_kernel_log,
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
