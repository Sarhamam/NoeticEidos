"""Statistical analysis for dual transport modes and Mellin balance."""

from .spectra import (
    spectral_gap,
    spectral_entropy,
    spectral_gap_additive,
    spectral_entropy_additive,
    spectral_gap_multiplicative,
    spectral_entropy_multiplicative,
    eigenvalue_distribution,
    effective_resistance_sum
)
from .stability import (
    stability_score,
    noise_perturbation,
    subsample_perturbation,
    bootstrap_perturbation,
    coordinate_perturbation,
    stability_curve,
    multi_statistic_stability,
    relative_stability
)
from .separability import (
    separability_test,
    mode_comparison_matrix,
    effect_size_interpretation,
    power_analysis
)
from .balance import (
    mellin_coupled_stat,
    balance_curve,
    balance_score,
    mellin_stability_test,
    cross_mode_balance_analysis,
    additive_multiplicative_interpolation
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