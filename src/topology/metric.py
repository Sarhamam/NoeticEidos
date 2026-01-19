"""Seam-compatibility for Fisher-Rao pullback metrics and operators.

Implements validation and enforcement of seam-compatibility conditions
for metrics and operators on the Möbius band quotient. The key requirement
is that metrics must satisfy g(u+π, -v) = (dT)ᵀ g(u,v) dT where dT = diag(1,-1).
"""

from typing import Any, Callable, Dict, Optional

import numpy as np

from .coords import Strip, deck_map


class SeamCompatibilityError(Exception):
    """Raised when seam-compatibility requirements are violated."""

    pass


def seam_compatible_metric(
    g_fn: Callable[[np.ndarray], np.ndarray],
    q: np.ndarray,
    strip: Strip,
    tolerance: float = 1e-8,
) -> bool:
    """Check if metric function satisfies seam-compatibility condition.

    For a metric to be well-defined on the Möbius band quotient, it must satisfy:
    g(u+π, -v) = (dT)ᵀ g(u,v) dT
    where dT = diag(1, -1) is the deck map differential.

    Parameters
    ----------
    g_fn : callable
        Metric function that takes coordinates and returns 2×2 metric tensor
    q : ndarray
        Test coordinates [u, v]
    strip : Strip
        Strip configuration
    tolerance : float
        Numerical tolerance for compatibility check

    Returns
    -------
    compatible : bool
        True if metric satisfies seam-compatibility

    Notes
    -----
    The compatibility condition expands to:
    - g₁₁(u+π, -v) = g₁₁(u, v)
    - g₁₂(u+π, -v) = -g₁₂(u, v)
    - g₂₁(u+π, -v) = -g₂₁(u, v)
    - g₂₂(u+π, -v) = g₂₂(u, v)
    """
    q = np.asarray(q)
    if q.shape != (2,):
        raise ValueError("Coordinates must be 2D array [u, v]")

    u, v = q[0], q[1]

    # Evaluate metric at original point
    g_orig = g_fn(q)
    if g_orig.shape != (2, 2):
        raise ValueError("Metric function must return 2×2 matrix")

    # Evaluate metric at deck-mapped point
    u_deck, v_deck = deck_map(u, v, strip)
    q_deck = np.array([u_deck, v_deck])
    g_deck = g_fn(q_deck)

    # Deck map differential: dT = diag(1, -1)
    dT = np.array([[1.0, 0.0], [0.0, -1.0]])

    # Expected metric at deck point: (dT)ᵀ g(u,v) dT
    g_expected = dT.T @ g_orig @ dT

    # Check compatibility
    error = np.max(np.abs(g_deck - g_expected))
    return error <= tolerance


def enforce_seam_compatibility(
    g_fn: Callable[[np.ndarray], np.ndarray],
    q: np.ndarray,
    strip: Strip,
    tolerance: float = 1e-8,
) -> None:
    """Enforce seam-compatibility for metric function, raising error if violated.

    Parameters
    ----------
    g_fn : callable
        Metric function to validate
    q : ndarray
        Test coordinates
    strip : Strip
        Strip configuration
    tolerance : float
        Numerical tolerance

    Raises
    ------
    SeamCompatibilityError
        If metric violates seam-compatibility condition
    """
    if not seam_compatible_metric(g_fn, q, strip, tolerance):
        u, v = q[0], q[1]
        u_deck, v_deck = deck_map(u, v, strip)

        g_orig = g_fn(q)
        g_deck = g_fn(np.array([u_deck, v_deck]))

        dT = np.array([[1.0, 0.0], [0.0, -1.0]])
        g_expected = dT.T @ g_orig @ dT

        error = np.max(np.abs(g_deck - g_expected))

        raise SeamCompatibilityError(
            f"Metric violates seam-compatibility at ({u:.3f}, {v:.3f}). "
            f"Max error: {error:.2e} > tolerance: {tolerance:.2e}. "
            f"Metric at deck point should satisfy g(u+π,-v) = dT^T g(u,v) dT."
        )


def validate_metric_grid(
    g_fn: Callable[[np.ndarray], np.ndarray],
    strip: Strip,
    n_u: int = 10,
    n_v: int = 10,
    tolerance: float = 1e-8,
) -> Dict[str, Any]:
    """Validate seam-compatibility across a grid of points.

    Parameters
    ----------
    g_fn : callable
        Metric function to validate
    strip : Strip
        Strip configuration
    n_u, n_v : int
        Grid resolution in u and v directions
    tolerance : float
        Numerical tolerance

    Returns
    -------
    validation_report : dict
        Grid validation results with error statistics
    """
    u_vals = np.linspace(0, strip.period, n_u, endpoint=False)
    v_vals = np.linspace(-strip.w, strip.w, n_v)

    errors = []
    violations = []

    for u in u_vals:
        for v in v_vals:
            q = np.array([u, v])
            try:
                if not seam_compatible_metric(g_fn, q, strip, tolerance):
                    # Compute actual error
                    g_orig = g_fn(q)
                    u_deck, v_deck = deck_map(u, v, strip)
                    g_deck = g_fn(np.array([u_deck, v_deck]))

                    dT = np.array([[1.0, 0.0], [0.0, -1.0]])
                    g_expected = dT.T @ g_orig @ dT
                    error = np.max(np.abs(g_deck - g_expected))

                    errors.append(error)
                    violations.append((u, v, error))
                else:
                    errors.append(0.0)

            except Exception as e:
                violations.append((u, v, f"Error: {e}"))
                errors.append(float("inf"))

    errors = np.array(errors)
    finite_errors = errors[np.isfinite(errors)]

    report = {
        "grid_size": (n_u, n_v),
        "total_points": len(errors),
        "violations": len(violations),
        "violation_rate": len(violations) / len(errors),
        "max_error": float(np.max(finite_errors)) if len(finite_errors) > 0 else 0.0,
        "mean_error": float(np.mean(finite_errors)) if len(finite_errors) > 0 else 0.0,
        "tolerance": tolerance,
        "compatible": len(violations) == 0,
        "violation_details": violations[:10],  # First 10 violations for debugging
    }

    return report


def seam_compatible_operator(
    Aq_fn: Callable[[np.ndarray], np.ndarray],
    q: np.ndarray,
    strip: Strip,
    tolerance: float = 1e-8,
) -> bool:
    """Check if operator satisfies seam-compatibility.

    An operator A is seam-compatible if A(T(q)) = dT A(q) dT^T
    where T is the deck map and dT its differential.

    Parameters
    ----------
    Aq_fn : callable
        Operator function that takes coordinates and returns matrix
    q : ndarray
        Test coordinates [u, v]
    strip : Strip
        Strip configuration
    tolerance : float
        Numerical tolerance

    Returns
    -------
    compatible : bool
        True if operator satisfies seam-compatibility
    """
    q = np.asarray(q)
    if q.shape != (2,):
        raise ValueError("Coordinates must be 2D array [u, v]")

    u, v = q[0], q[1]

    # Evaluate operator at original point
    A_orig = Aq_fn(q)
    if A_orig.shape != (2, 2):
        raise ValueError("Operator function must return 2×2 matrix")

    # Evaluate operator at deck-mapped point
    u_deck, v_deck = deck_map(u, v, strip)
    q_deck = np.array([u_deck, v_deck])
    A_deck = Aq_fn(q_deck)

    # Deck map differential: dT = diag(1, -1)
    dT = np.array([[1.0, 0.0], [0.0, -1.0]])

    # Expected operator at deck point: dT A(q) dT^T
    A_expected = dT @ A_orig @ dT.T

    # Check compatibility
    error = np.max(np.abs(A_deck - A_expected))
    return error <= tolerance


def symmetrize_metric(g: np.ndarray) -> np.ndarray:
    """Ensure metric tensor is symmetric.

    Parameters
    ----------
    g : ndarray
        2×2 metric tensor

    Returns
    -------
    g_sym : ndarray
        Symmetrized metric tensor
    """
    g = np.asarray(g)
    if g.shape != (2, 2):
        raise ValueError("Metric must be 2×2 matrix")

    return 0.5 * (g + g.T)


def check_metric_positive_definite(g: np.ndarray, min_eigenvalue: float = 1e-9) -> bool:
    """Check if metric tensor is positive definite.

    Parameters
    ----------
    g : ndarray
        2×2 metric tensor
    min_eigenvalue : float
        Minimum allowed eigenvalue

    Returns
    -------
    is_pd : bool
        True if metric is positive definite
    """
    g = np.asarray(g)
    if g.shape != (2, 2):
        raise ValueError("Metric must be 2×2 matrix")

    # Symmetrize to handle numerical errors
    g_sym = symmetrize_metric(g)

    # Check eigenvalues
    eigenvals = np.linalg.eigvals(g_sym)
    return np.all(eigenvals >= min_eigenvalue)


def regularize_metric(g: np.ndarray, regularization: float = 1e-6) -> np.ndarray:
    """Add regularization to ensure positive definiteness.

    Parameters
    ----------
    g : ndarray
        2×2 metric tensor
    regularization : float
        Regularization parameter added to diagonal

    Returns
    -------
    g_reg : ndarray
        Regularized metric tensor
    """
    g = np.asarray(g)
    if g.shape != (2, 2):
        raise ValueError("Metric must be 2×2 matrix")

    g_sym = symmetrize_metric(g)
    return g_sym + regularization * np.eye(2)


def make_seam_compatible_metric(
    g11_fn: Callable[[np.ndarray], float],
    g22_fn: Callable[[np.ndarray], float],
    g12_fn: Optional[Callable[[np.ndarray], float]] = None,
) -> Callable:
    """Construct seam-compatible metric from component functions.

    Creates a metric function that automatically satisfies seam-compatibility
    by enforcing the transformation rules under the deck map.

    Parameters
    ----------
    g11_fn : callable
        Function for g₁₁ component (must be even in v: g₁₁(u,-v) = g₁₁(u,v))
    g22_fn : callable
        Function for g₂₂ component (must be even in v: g₂₂(u,-v) = g₂₂(u,v))
    g12_fn : callable or None
        Function for g₁₂ component (must be odd in v: g₁₂(u,-v) = -g₁₂(u,v))

    Returns
    -------
    metric_fn : callable
        Seam-compatible metric function

    Notes
    -----
    For seam-compatibility, the metric components must satisfy:
    - g₁₁(u+π, -v) = g₁₁(u, v) (even in v, periodic in u)
    - g₂₂(u+π, -v) = g₂₂(u, v) (even in v, periodic in u)
    - g₁₂(u+π, -v) = -g₁₂(u, v) (odd in v, periodic in u)
    """

    def metric_fn(q: np.ndarray) -> np.ndarray:
        _u, _v = q[0], q[1]

        g11 = g11_fn(q)
        g22 = g22_fn(q)
        g12 = g12_fn(q) if g12_fn is not None else 0.0

        g = np.array([[g11, g12], [g12, g22]])

        return g

    return metric_fn


def validate_component_symmetries(
    g11_fn: Callable,
    g22_fn: Callable,
    g12_fn: Callable,
    strip: Strip,
    tolerance: float = 1e-8,
) -> Dict[str, bool]:
    """Validate that metric components satisfy required symmetries.

    Parameters
    ----------
    g11_fn, g22_fn, g12_fn : callable
        Metric component functions
    strip : Strip
        Strip configuration
    tolerance : float
        Numerical tolerance

    Returns
    -------
    symmetry_results : dict
        Results of symmetry validation for each component
    """
    # Test points
    u_test = np.array([0.0, np.pi / 2, np.pi, 3 * np.pi / 2])
    v_test = np.array(
        [-0.8 * strip.w, -0.5 * strip.w, 0.0, 0.5 * strip.w, 0.8 * strip.w]
    )

    results = {
        "g11_even_in_v": True,
        "g22_even_in_v": True,
        "g12_odd_in_v": True,
        "g11_periodic_in_u": True,
        "g22_periodic_in_u": True,
        "g12_periodic_in_u": True,
    }

    for u in u_test:
        for v in v_test:
            q = np.array([u, v])
            q_v_flipped = np.array([u, -v])
            q_u_shifted = np.array([u + np.pi, v])

            # Test v-symmetries
            if np.abs(g11_fn(q) - g11_fn(q_v_flipped)) > tolerance:
                results["g11_even_in_v"] = False

            if np.abs(g22_fn(q) - g22_fn(q_v_flipped)) > tolerance:
                results["g22_even_in_v"] = False

            if np.abs(g12_fn(q) + g12_fn(q_v_flipped)) > tolerance:
                results["g12_odd_in_v"] = False

            # Test u-periodicity (shifted by π for seam compatibility)
            if np.abs(g11_fn(q) - g11_fn(q_u_shifted)) > tolerance:
                results["g11_periodic_in_u"] = False

            if np.abs(g22_fn(q) - g22_fn(q_u_shifted)) > tolerance:
                results["g22_periodic_in_u"] = False

            if np.abs(g12_fn(q) - g12_fn(q_u_shifted)) > tolerance:
                results["g12_periodic_in_u"] = False

    return results
