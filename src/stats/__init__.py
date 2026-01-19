"""Statistical analysis for dual transport modes and Mellin balance."""

from .balance import (
    additive_multiplicative_interpolation,
    balance_curve,
    balance_score,
    cross_mode_balance_analysis,
    mellin_coupled_stat,
    mellin_stability_test,
)
from .separability import (
    effect_size_interpretation,
    mode_comparison_matrix,
    power_analysis,
    separability_test,
)
from .spectra import (
    effective_resistance_sum,
    eigenvalue_distribution,
    spectral_entropy,
    spectral_entropy_additive,
    spectral_entropy_multiplicative,
    spectral_gap,
    spectral_gap_additive,
    spectral_gap_multiplicative,
)
from .stability import (
    bootstrap_perturbation,
    coordinate_perturbation,
    multi_statistic_stability,
    noise_perturbation,
    relative_stability,
    stability_curve,
    stability_score,
    subsample_perturbation,
)

__all__ = [
    # Spectral measures
    "spectral_gap",
    "spectral_entropy",
    "spectral_gap_additive",
    "spectral_entropy_additive",
    "spectral_gap_multiplicative",
    "spectral_entropy_multiplicative",
    "eigenvalue_distribution",
    "effective_resistance_sum",
    # Stability analysis
    "stability_score",
    "noise_perturbation",
    "subsample_perturbation",
    "bootstrap_perturbation",
    "coordinate_perturbation",
    "stability_curve",
    "multi_statistic_stability",
    "relative_stability",
    # Separability testing
    "separability_test",
    "mode_comparison_matrix",
    "effect_size_interpretation",
    "power_analysis",
    # Mellin balance
    "mellin_coupled_stat",
    "balance_curve",
    "balance_score",
    "mellin_stability_test",
    "cross_mode_balance_analysis",
    "additive_multiplicative_interpolation",
]
