"""Critical validation framework for geometric ML pipeline robustness.

This module provides essential guards against common failure modes in geometric ML:
- Mathematical validity (connectivity, transversality, mass conservation)
- Numerical stability (precision, conditioning, convergence)
- Statistical rigor (sample sizes, multiple testing, reproducibility)
- Performance safety (memory limits, scaling cliffs, runtime monitoring)

Usage:
    from validation import (
        check_graph_connectivity,
        validate_transversality,
        monitor_mass_conservation,
        ensure_reproducibility
    )
"""

from .mathematical import (
    check_graph_connectivity,
    validate_transversality,
    TransversalityError,
    ConnectivityError
)
from .numerical import (
    monitor_mass_conservation,
    validate_cg_convergence,
    check_eigenvalue_validity,
    NumericalStabilityError
)
from .statistical import (
    validate_bootstrap_size,
    check_separability_null,
    apply_multiple_testing_correction,
    StatisticalValidityError
)
from .reproducibility import (
    ensure_reproducibility,
    compute_data_hash,
    log_experiment_config,
    ReproducibilityError
)
from .performance import (
    check_memory_limits,
    monitor_runtime_complexity,
    detect_scaling_cliffs,
    PerformanceError
)

__all__ = [
    # Mathematical guards
    "check_graph_connectivity",
    "validate_transversality",
    "TransversalityError",
    "ConnectivityError",
    # Numerical stability
    "monitor_mass_conservation",
    "validate_cg_convergence",
    "check_eigenvalue_validity",
    "NumericalStabilityError",
    # Statistical rigor
    "validate_bootstrap_size",
    "check_separability_null",
    "apply_multiple_testing_correction",
    "StatisticalValidityError",
    # Reproducibility
    "ensure_reproducibility",
    "compute_data_hash",
    "log_experiment_config",
    "ReproducibilityError",
    # Performance monitoring
    "check_memory_limits",
    "monitor_runtime_complexity",
    "detect_scaling_cliffs",
    "PerformanceError",
]