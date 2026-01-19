"""Numerical stability guards for geometric ML computations."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Union

import numpy as np
from scipy.sparse import issparse

if TYPE_CHECKING:
    from scipy.sparse import csr_matrix


class NumericalStabilityError(Exception):
    """Raised when numerical stability conditions are violated."""

    pass


def monitor_mass_conservation(
    u_initial: np.ndarray,
    u_final: np.ndarray,
    tolerance: float = 1e-6,
    relative: bool = True,
) -> Dict[str, Any]:
    """Monitor mass conservation in diffusion processes.

    Parameters
    ----------
    u_initial : ndarray
        Initial state vector
    u_final : ndarray
        Final state vector after diffusion
    tolerance : float
        Tolerance for mass conservation violation
    relative : bool
        If True, use relative tolerance; if False, use absolute

    Returns
    -------
    conservation_report : dict
        Mass conservation analysis with violation flags

    Raises
    ------
    NumericalStabilityError
        If mass conservation is violated beyond tolerance

    Notes
    -----
    For diffusion processes du/dt = -Lu, total mass ∑u(t) should be conserved
    for connected graphs. Violations indicate numerical errors or non-physical behavior.
    """
    u_initial = np.asarray(u_initial).flatten()
    u_final = np.asarray(u_final).flatten()

    if len(u_initial) != len(u_final):
        raise ValueError("Initial and final states must have same length")

    # Compute masses
    mass_initial = np.sum(u_initial)
    mass_final = np.sum(u_final)
    mass_change = mass_final - mass_initial

    # Assess conservation
    if relative and abs(mass_initial) > 1e-12:
        relative_change = abs(mass_change) / abs(mass_initial)
        conserved = relative_change <= tolerance
        error_metric = relative_change
        error_type = "relative"
    else:
        absolute_change = abs(mass_change)
        conserved = absolute_change <= tolerance
        error_metric = absolute_change
        error_type = "absolute"

    # Additional diagnostics
    finite_initial = np.all(np.isfinite(u_initial))
    finite_final = np.all(np.isfinite(u_final))

    # Check for negative values (unphysical for many diffusion processes)
    negative_initial = np.any(u_initial < -1e-12)
    negative_final = np.any(u_final < -1e-12)

    report = {
        "conserved": conserved,
        "mass_initial": float(mass_initial),
        "mass_final": float(mass_final),
        "mass_change": float(mass_change),
        "error_metric": float(error_metric),
        "error_type": error_type,
        "tolerance": tolerance,
        "finite_initial": finite_initial,
        "finite_final": finite_final,
        "negative_initial": negative_initial,
        "negative_final": negative_final,
    }

    if not conserved:
        raise NumericalStabilityError(
            f"Mass conservation violated: {error_type} change = {error_metric:.2e} "
            f"(tolerance: {tolerance:.2e}). "
            f"Initial mass: {mass_initial:.6e}, Final mass: {mass_final:.6e}"
        )

    if not finite_initial or not finite_final:
        raise NumericalStabilityError(
            f"Non-finite values detected. "
            f"Initial finite: {finite_initial}, Final finite: {finite_final}"
        )

    return report


def validate_cg_convergence(
    residual_history: List[float],
    tolerance: float = 1e-6,
    max_stagnation_ratio: float = 0.99,
    min_progress_steps: int = 10,
) -> Dict[str, Any]:
    """Validate CG convergence and detect stagnation.

    Parameters
    ----------
    residual_history : list of float
        Sequence of residual norms throughout CG iterations
    tolerance : float
        Target convergence tolerance
    max_stagnation_ratio : float
        Maximum ratio for consecutive residuals (detect stagnation)
    min_progress_steps : int
        Minimum steps before checking for stagnation

    Returns
    -------
    convergence_report : dict
        CG convergence analysis with stagnation detection

    Raises
    ------
    NumericalStabilityError
        If CG exhibits problematic behavior (divergence, stagnation)
    """
    if len(residual_history) < 2:
        raise ValueError("Need at least 2 residual values for convergence analysis")

    residuals = np.array(residual_history)
    n_iterations = len(residuals)

    # Check for non-finite residuals
    if not np.all(np.isfinite(residuals)):
        raise NumericalStabilityError(f"Non-finite residuals detected: {residuals}")

    # Check for divergence
    if residuals[-1] > residuals[0]:
        warnings.warn(
            f"CG residual increased from {residuals[0]:.2e} to {residuals[-1]:.2e}",
            stacklevel=2,
        )

    # Check for convergence
    converged = residuals[-1] <= tolerance
    final_residual = residuals[-1]

    # Stagnation detection
    stagnation_detected = False
    stagnation_start = None

    if n_iterations >= min_progress_steps:
        # Look for consecutive steps with minimal progress
        for i in range(min_progress_steps, n_iterations):
            if residuals[i - 1] > 1e-15:  # Avoid division by zero
                ratio = residuals[i] / residuals[i - 1]
                if ratio >= max_stagnation_ratio:
                    stagnation_detected = True
                    stagnation_start = i - 1
                    break

    # Convergence rate analysis (if enough iterations)
    convergence_rate = None
    if n_iterations >= 3:
        # Estimate convergence rate from last few iterations
        recent_residuals = residuals[-3:]
        if np.all(recent_residuals > 1e-15):
            rates = recent_residuals[1:] / recent_residuals[:-1]
            convergence_rate = np.mean(rates)

    # Progress assessment
    if residuals[0] > 1e-15:
        progress_ratio = residuals[-1] / residuals[0]
    else:
        progress_ratio = 1.0

    report = {
        "converged": converged,
        "final_residual": float(final_residual),
        "initial_residual": float(residuals[0]),
        "progress_ratio": float(progress_ratio),
        "n_iterations": n_iterations,
        "stagnation_detected": stagnation_detected,
        "stagnation_start": stagnation_start,
        "convergence_rate": (
            float(convergence_rate) if convergence_rate is not None else None
        ),
        "tolerance": tolerance,
    }

    # Raise errors for problematic behavior
    if stagnation_detected:
        raise NumericalStabilityError(
            f"CG stagnation detected at iteration {stagnation_start}. "
            f"Residual progress ratio >= {max_stagnation_ratio}. "
            f"Consider using preconditioning or increasing regularization."
        )

    if final_residual > 1e3 * residuals[0]:
        raise NumericalStabilityError(
            f"CG divergence detected: residual increased by factor {final_residual/residuals[0]:.2e}"
        )

    return report


def check_eigenvalue_validity(
    eigenvalues: np.ndarray, matrix_type: str = "laplacian", tolerance: float = 1e-12
) -> Dict[str, Any]:
    """Validate eigenvalue properties for different matrix types.

    Parameters
    ----------
    eigenvalues : ndarray
        Computed eigenvalues
    matrix_type : str
        Type of matrix: "laplacian", "symmetric", "psd"
    tolerance : float
        Tolerance for numerical checks

    Returns
    -------
    validity_report : dict
        Eigenvalue validity analysis

    Raises
    ------
    NumericalStabilityError
        If eigenvalues violate expected properties
    """
    evals = np.asarray(eigenvalues)

    # Check for non-finite values
    finite_mask = np.isfinite(evals)
    n_finite = np.sum(finite_mask)
    n_total = len(evals)

    if n_finite < n_total:
        raise NumericalStabilityError(
            f"Non-finite eigenvalues detected: {n_total - n_finite} out of {n_total}"
        )

    # Check for complex eigenvalues (should be real for symmetric matrices)
    if np.any(np.abs(np.imag(evals)) > tolerance):
        raise NumericalStabilityError(
            f"Complex eigenvalues detected for {matrix_type} matrix. "
            f"Max imaginary part: {np.max(np.abs(np.imag(evals))):.2e}"
        )

    evals_real = np.real(evals)

    # Matrix-specific checks
    if matrix_type == "laplacian":
        # Laplacian eigenvalues should be non-negative
        negative_evals = evals_real[evals_real < -tolerance]
        if len(negative_evals) > 0:
            raise NumericalStabilityError(
                f"Negative Laplacian eigenvalues detected: {negative_evals[:5]}... "
                f"(showing first 5 of {len(negative_evals)})"
            )

        # Check for zero eigenvalue (connectivity)
        zero_evals = np.sum(np.abs(evals_real) <= tolerance)

        # For connected graphs, should have exactly one zero eigenvalue
        if zero_evals == 0:
            warnings.warn(
                "No zero eigenvalue found - unusual for Laplacian", stacklevel=2
            )
        elif zero_evals > 1:
            warnings.warn(
                f"Multiple zero eigenvalues ({zero_evals}) - disconnected graph",
                stacklevel=2,
            )

    elif matrix_type == "psd":
        # Positive semidefinite: all eigenvalues >= 0
        negative_evals = evals_real[evals_real < -tolerance]
        if len(negative_evals) > 0:
            raise NumericalStabilityError(
                f"Negative eigenvalues in PSD matrix: {negative_evals[:5]}..."
            )

    # Condition number assessment
    positive_evals = evals_real[evals_real > tolerance]
    if len(positive_evals) > 0:
        condition_number = np.max(positive_evals) / np.min(positive_evals)
        well_conditioned = condition_number < 1e12
    else:
        condition_number = float("inf")
        well_conditioned = False

    return {
        "valid": True,  # If we reach here without exceptions
        "n_eigenvalues": n_total,
        "n_finite": n_finite,
        "matrix_type": matrix_type,
        "min_eigenvalue": float(np.min(evals_real)),
        "max_eigenvalue": float(np.max(evals_real)),
        "condition_number": float(condition_number),
        "well_conditioned": well_conditioned,
        "spectral_gap": (
            float(np.min(evals_real[evals_real > tolerance]))
            if len(positive_evals) > 0
            else 0.0
        ),
    }


def validate_float64_precision(arrays: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Validate that critical arrays use float64 precision.

    Parameters
    ----------
    arrays : dict
        Dictionary mapping names to arrays to check

    Returns
    -------
    precision_report : dict
        Precision validation report

    Raises
    ------
    NumericalStabilityError
        If arrays use insufficient precision
    """
    precision_issues = []
    dtypes = {}

    for name, array in arrays.items():
        arr = np.asarray(array)
        dtypes[name] = str(arr.dtype)

        # Check for insufficient precision
        if arr.dtype not in [np.float64, np.complex128]:
            if arr.dtype in [np.float32, np.complex64]:
                precision_issues.append(f"{name}: {arr.dtype} (recommend float64)")
            else:
                precision_issues.append(f"{name}: {arr.dtype} (non-float type)")

    report = {
        "precision_adequate": len(precision_issues) == 0,
        "dtypes": dtypes,
        "issues": precision_issues,
    }

    if precision_issues:
        warnings.warn(
            f"Precision issues detected: {'; '.join(precision_issues)}. "
            f"Consider using float64 for critical computations.",
            stacklevel=2,
        )

    return report


def check_matrix_conditioning(
    matrix: Union[np.ndarray, "csr_matrix"], condition_threshold: float = 1e12
) -> Dict[str, Any]:
    """Check matrix conditioning and recommend fixes for ill-conditioning.

    Parameters
    ----------
    matrix : sparse matrix or ndarray
        Matrix to analyze
    condition_threshold : float
        Threshold for identifying ill-conditioning

    Returns
    -------
    conditioning_report : dict
        Matrix conditioning analysis with recommendations
    """
    if issparse(matrix):
        # For sparse matrices, estimate condition number via eigenvalues
        from solvers.lanczos import topk_eigs

        try:
            # Get largest and smallest eigenvalues
            max_evals, _ = topk_eigs(matrix, k=1, which="LM")
            min_evals, _ = topk_eigs(matrix, k=1, which="SM")

            max_eval = max_evals[0] if len(max_evals) > 0 else 1.0
            min_eval = (
                min_evals[0] if len(min_evals) > 0 and min_evals[0] > 1e-12 else 1e-12
            )

            condition_number = max_eval / min_eval

        except Exception:
            condition_number = float("inf")
    else:
        # Dense matrix condition number
        try:
            condition_number = np.linalg.cond(matrix)
        except np.linalg.LinAlgError:
            condition_number = float("inf")

    well_conditioned = condition_number < condition_threshold

    # Recommendations for ill-conditioned matrices
    recommendations = []
    if not well_conditioned:
        recommendations.extend(
            [
                "Add regularization (α parameter in solvers)",
                "Use preconditioning for iterative methods",
                "Check for rank deficiency or near-singular rows/columns",
                "Consider eigenvalue filtering for very small eigenvalues",
            ]
        )

    return {
        "well_conditioned": well_conditioned,
        "condition_number": float(condition_number),
        "threshold": condition_threshold,
        "recommendations": recommendations,
    }
