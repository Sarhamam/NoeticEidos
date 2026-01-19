"""Topological invariance validation utilities.

Provides functions to validate that metrics, operators, and statistics
are properly invariant under the deck map transformations of the Möbius
band quotient topology.
"""

import numpy as np
import warnings
from typing import Callable, Dict, Any, List, Optional, Tuple, Union
from .coords import Strip, deck_map, seam_equivalent_points


class TopologicalValidationError(Exception):
    """Raised when topological invariance requirements are violated."""
    pass


def seam_invariance(phi: Callable[[np.ndarray], Union[float, np.ndarray]],
                   q: np.ndarray,
                   strip: Strip,
                   tolerance: float = 1e-8) -> bool:
    """Check if function phi is invariant under deck map transformation.

    For a function to be well-defined on the Möbius band quotient,
    it must satisfy: phi(T(q)) = phi(q) where T is the deck map.

    Parameters
    ----------
    phi : callable
        Function to test for invariance
    q : ndarray, shape (2,)
        Test point coordinates [u, v]
    strip : Strip
        Strip configuration
    tolerance : float
        Numerical tolerance for invariance check

    Returns
    -------
    invariant : bool
        True if function satisfies deck map invariance

    Notes
    -----
    The deck map is T(u,v) = (u+π, -v), so invariance means
    phi(u+π, -v) = phi(u, v) for all points in the fundamental domain.
    """
    q = np.asarray(q)
    if q.shape != (2,):
        raise ValueError("Coordinates must be 2D array [u, v]")

    u, v = q[0], q[1]

    # Evaluate function at original point
    try:
        value_orig = phi(q)
    except Exception as e:
        raise TopologicalValidationError(f"Function evaluation failed at ({u:.3f}, {v:.3f}): {e}")

    # Evaluate function at deck-mapped point
    u_deck, v_deck = deck_map(u, v, strip)
    q_deck = np.array([u_deck, v_deck])

    try:
        value_deck = phi(q_deck)
    except Exception as e:
        raise TopologicalValidationError(f"Function evaluation failed at deck point ({u_deck:.3f}, {v_deck:.3f}): {e}")

    # Check invariance
    if np.isscalar(value_orig) and np.isscalar(value_deck):
        error = abs(value_deck - value_orig)
    else:
        value_orig = np.asarray(value_orig)
        value_deck = np.asarray(value_deck)
        if value_orig.shape != value_deck.shape:
            raise TopologicalValidationError("Function returns different shapes at equivalent points")
        error = np.max(np.abs(value_deck - value_orig))

    return error <= tolerance


def seam_invariance_grid(phi: Callable[[np.ndarray], Union[float, np.ndarray]],
                        U: np.ndarray, V: np.ndarray,
                        strip: Strip,
                        tolerance: float = 1e-8) -> Dict[str, Any]:
    """Check seam invariance across a grid of points.

    Parameters
    ----------
    phi : callable
        Function to test for invariance
    U, V : ndarray
        2D coordinate grids (from meshgrid)
    strip : Strip
        Strip configuration
    tolerance : float
        Numerical tolerance

    Returns
    -------
    validation_report : dict
        Grid validation results with error statistics
    """
    U = np.asarray(U)
    V = np.asarray(V)

    if U.shape != V.shape:
        raise ValueError("U and V grids must have same shape")

    U_flat = U.flatten()
    V_flat = V.flatten()
    n_points = len(U_flat)

    errors = []
    violations = []
    failed_evaluations = []

    for i in range(n_points):
        u, v = U_flat[i], V_flat[i]
        q = np.array([u, v])

        try:
            if not seam_invariance(phi, q, strip, tolerance):
                # Compute actual error for reporting
                value_orig = phi(q)
                u_deck, v_deck = deck_map(u, v, strip)
                value_deck = phi(np.array([u_deck, v_deck]))

                if np.isscalar(value_orig):
                    error = abs(value_deck - value_orig)
                else:
                    error = np.max(np.abs(np.asarray(value_deck) - np.asarray(value_orig)))

                errors.append(error)
                violations.append((u, v, error))
            else:
                errors.append(0.0)

        except Exception as e:
            failed_evaluations.append((u, v, str(e)))
            errors.append(float('inf'))

    errors = np.array(errors)
    finite_errors = errors[np.isfinite(errors)]

    report = {
        "grid_shape": U.shape,
        "total_points": n_points,
        "violations": len(violations),
        "failed_evaluations": len(failed_evaluations),
        "violation_rate": len(violations) / n_points,
        "failure_rate": len(failed_evaluations) / n_points,
        "max_error": float(np.max(finite_errors)) if len(finite_errors) > 0 else 0.0,
        "mean_error": float(np.mean(finite_errors)) if len(finite_errors) > 0 else 0.0,
        "std_error": float(np.std(finite_errors)) if len(finite_errors) > 0 else 0.0,
        "tolerance": tolerance,
        "invariant": len(violations) == 0 and len(failed_evaluations) == 0,
        "violation_details": violations[:10],  # First 10 violations
        "failure_details": failed_evaluations[:5]  # First 5 failures
    }

    return report


def validate_metric_invariance(g_fn: Callable[[np.ndarray], np.ndarray],
                              strip: Strip,
                              n_test: int = 50,
                              tolerance: float = 1e-8) -> Dict[str, Any]:
    """Validate that metric function satisfies topological invariance.

    Tests the stronger condition that g(T(q)) = g(q), not just seam-compatibility.
    This is for metrics that are truly invariant (constant) rather than
    seam-compatible pullbacks.

    Parameters
    ----------
    g_fn : callable
        Metric function to validate
    strip : Strip
        Strip configuration
    n_test : int
        Number of random test points
    tolerance : float
        Numerical tolerance

    Returns
    -------
    validation_report : dict
        Invariance validation results
    """
    rng = np.random.default_rng(42)  # Reproducible tests

    # Generate random test points in fundamental domain
    u_test = rng.uniform(0, strip.period, n_test)
    v_test = rng.uniform(-strip.w, strip.w, n_test)

    violations = []
    errors = []

    for i in range(n_test):
        q = np.array([u_test[i], v_test[i]])

        try:
            # Test full invariance: g(T(q)) = g(q)
            g_orig = g_fn(q)
            u_deck, v_deck = deck_map(u_test[i], v_test[i], strip)
            g_deck = g_fn(np.array([u_deck, v_deck]))

            error = np.max(np.abs(g_deck - g_orig))
            errors.append(error)

            if error > tolerance:
                violations.append((u_test[i], v_test[i], error))

        except Exception as e:
            violations.append((u_test[i], v_test[i], f"Error: {e}"))
            errors.append(float('inf'))

    errors = np.array(errors)
    finite_errors = errors[np.isfinite(errors)]

    report = {
        "test_type": "metric_invariance",
        "n_test": n_test,
        "violations": len(violations),
        "violation_rate": len(violations) / n_test,
        "max_error": float(np.max(finite_errors)) if len(finite_errors) > 0 else 0.0,
        "mean_error": float(np.mean(finite_errors)) if len(finite_errors) > 0 else 0.0,
        "tolerance": tolerance,
        "invariant": len(violations) == 0,
        "violation_details": violations[:10]
    }

    return report


def validate_spectral_invariance(spectrum_fn: Callable[[np.ndarray], np.ndarray],
                                 strip: Strip,
                                 n_test: int = 20,
                                 tolerance: float = 1e-6) -> Dict[str, Any]:
    """Validate that spectral statistics are invariant under deck map.

    Parameters
    ----------
    spectrum_fn : callable
        Function that computes spectral statistics at given coordinates
    strip : Strip
        Strip configuration
    n_test : int
        Number of test points
    tolerance : float
        Numerical tolerance

    Returns
    -------
    validation_report : dict
        Spectral invariance validation results
    """
    def spectral_invariance_test(q):
        """Helper function for testing spectral invariance."""
        return seam_invariance(spectrum_fn, q, strip, tolerance)

    # Generate test grid
    u_test = np.linspace(0, strip.period, n_test, endpoint=False)
    v_test = np.linspace(-strip.w, strip.w, n_test)
    U, V = np.meshgrid(u_test, v_test)

    report = seam_invariance_grid(spectrum_fn, U, V, strip, tolerance)
    report["test_type"] = "spectral_invariance"

    return report


def validate_operator_invariance(operator_fn: Callable[[np.ndarray], np.ndarray],
                                strip: Strip,
                                n_test: int = 30,
                                tolerance: float = 1e-8) -> Dict[str, Any]:
    """Validate that operator satisfies proper transformation under deck map.

    For operators, the required transformation is A(T(q)) = dT A(q) dT^T,
    not simple invariance.

    Parameters
    ----------
    operator_fn : callable
        Operator function returning matrix at given coordinates
    strip : Strip
        Strip configuration
    n_test : int
        Number of test points
    tolerance : float
        Numerical tolerance

    Returns
    -------
    validation_report : dict
        Operator transformation validation results
    """
    from .metric import seam_compatible_operator

    rng = np.random.default_rng(42)
    u_test = rng.uniform(0, strip.period, n_test)
    v_test = rng.uniform(-strip.w, strip.w, n_test)

    violations = []
    errors = []

    for i in range(n_test):
        q = np.array([u_test[i], v_test[i]])

        try:
            compatible = seam_compatible_operator(operator_fn, q, strip, tolerance)
            if not compatible:
                # Compute error for reporting
                A_orig = operator_fn(q)
                u_deck, v_deck = deck_map(u_test[i], v_test[i], strip)
                A_deck = operator_fn(np.array([u_deck, v_deck]))

                dT = np.array([[1.0, 0.0], [0.0, -1.0]])
                A_expected = dT @ A_orig @ dT.T

                error = np.max(np.abs(A_deck - A_expected))
                errors.append(error)
                violations.append((u_test[i], v_test[i], error))
            else:
                errors.append(0.0)

        except Exception as e:
            violations.append((u_test[i], v_test[i], f"Error: {e}"))
            errors.append(float('inf'))

    errors = np.array(errors)
    finite_errors = errors[np.isfinite(errors)]

    report = {
        "test_type": "operator_seam_compatibility",
        "n_test": n_test,
        "violations": len(violations),
        "violation_rate": len(violations) / n_test,
        "max_error": float(np.max(finite_errors)) if len(finite_errors) > 0 else 0.0,
        "mean_error": float(np.mean(finite_errors)) if len(finite_errors) > 0 else 0.0,
        "tolerance": tolerance,
        "seam_compatible": len(violations) == 0,
        "violation_details": violations[:10]
    }

    return report


def validate_geodesic_invariance(geodesic_fn: Callable[[np.ndarray, np.ndarray, float], Tuple],
                                strip: Strip,
                                n_test: int = 10,
                                tolerance: float = 1e-6) -> Dict[str, Any]:
    """Validate that geodesic integration respects topological structure.

    Tests that geodesics starting from equivalent points (related by deck map)
    produce equivalent trajectories.

    Parameters
    ----------
    geodesic_fn : callable
        Function that integrates geodesics: (q0, v0, t) -> (traj_q, traj_v, info)
    strip : Strip
        Strip configuration
    n_test : int
        Number of test cases
    tolerance : float
        Tolerance for trajectory comparison

    Returns
    -------
    validation_report : dict
        Geodesic invariance validation results
    """
    rng = np.random.default_rng(42)

    violations = []
    trajectory_errors = []

    for i in range(n_test):
        # Random initial conditions
        u0 = rng.uniform(0, strip.period)
        v0 = rng.uniform(-strip.w, strip.w)
        du0 = rng.uniform(-1, 1)
        dv0 = rng.uniform(-1, 1)

        q0 = np.array([u0, v0])
        v0_vec = np.array([du0, dv0])

        # Equivalent initial conditions via deck map
        u0_deck, v0_deck = deck_map(u0, v0, strip)
        q0_deck = np.array([u0_deck, v0_deck])
        # Transform velocity: dT = diag(1, -1)
        v0_deck = np.array([du0, -dv0])

        try:
            # Integrate both geodesics
            traj_q1, traj_v1, info1 = geodesic_fn(q0, v0_vec, 2.0)  # Short integration
            traj_q2, traj_v2, info2 = geodesic_fn(q0_deck, v0_deck, 2.0)

            if not (info1["success"] and info2["success"]):
                violations.append((i, "Integration failed"))
                trajectory_errors.append(float('inf'))
                continue

            # Compare final points (should be equivalent under deck map)
            q_final1 = traj_q1[-1]
            q_final2 = traj_q2[-1]

            # Transform second trajectory endpoint back
            u_final2_mapped, v_final2_mapped = deck_map(q_final2[0], q_final2[1], strip)
            q_final2_mapped = np.array([u_final2_mapped, v_final2_mapped])

            # Compare endpoints
            endpoint_error = np.linalg.norm(q_final1 - q_final2_mapped)
            trajectory_errors.append(endpoint_error)

            if endpoint_error > tolerance:
                violations.append((i, endpoint_error))

        except Exception as e:
            violations.append((i, f"Error: {e}"))
            trajectory_errors.append(float('inf'))

    trajectory_errors = np.array(trajectory_errors)
    finite_errors = trajectory_errors[np.isfinite(trajectory_errors)]

    report = {
        "test_type": "geodesic_invariance",
        "n_test": n_test,
        "violations": len(violations),
        "violation_rate": len(violations) / n_test,
        "max_trajectory_error": float(np.max(finite_errors)) if len(finite_errors) > 0 else 0.0,
        "mean_trajectory_error": float(np.mean(finite_errors)) if len(finite_errors) > 0 else 0.0,
        "tolerance": tolerance,
        "geodesics_invariant": len(violations) == 0,
        "violation_details": violations[:5]
    }

    return report


def comprehensive_topology_validation(g_fn: Callable[[np.ndarray], np.ndarray],
                                     spectrum_fn: Optional[Callable] = None,
                                     operator_fn: Optional[Callable] = None,
                                     geodesic_fn: Optional[Callable] = None,
                                     strip: Optional[Strip] = None,
                                     tolerance: float = 1e-8) -> Dict[str, Any]:
    """Run comprehensive topological validation suite.

    Parameters
    ----------
    g_fn : callable
        Metric function
    spectrum_fn : callable, optional
        Spectral statistics function
    operator_fn : callable, optional
        Operator function
    geodesic_fn : callable, optional
        Geodesic integration function
    strip : Strip, optional
        Strip configuration (default: Strip(w=1.0))
    tolerance : float
        Numerical tolerance

    Returns
    -------
    validation_report : dict
        Comprehensive validation results
    """
    if strip is None:
        strip = Strip(w=1.0)

    report = {
        "strip_config": {"w": strip.w, "period": strip.period},
        "tolerance": tolerance,
        "tests_run": [],
        "all_passed": True
    }

    # Always test metric compatibility
    from .metric import validate_metric_grid
    metric_report = validate_metric_grid(g_fn, strip, tolerance=tolerance)
    report["metric_seam_compatibility"] = metric_report
    report["tests_run"].append("metric_seam_compatibility")
    if not metric_report["compatible"]:
        report["all_passed"] = False

    # Optional tests
    if spectrum_fn is not None:
        spectral_report = validate_spectral_invariance(spectrum_fn, strip, tolerance=tolerance)
        report["spectral_invariance"] = spectral_report
        report["tests_run"].append("spectral_invariance")
        if not spectral_report["invariant"]:
            report["all_passed"] = False

    if operator_fn is not None:
        operator_report = validate_operator_invariance(operator_fn, strip, tolerance=tolerance)
        report["operator_seam_compatibility"] = operator_report
        report["tests_run"].append("operator_seam_compatibility")
        if not operator_report["seam_compatible"]:
            report["all_passed"] = False

    if geodesic_fn is not None:
        geodesic_report = validate_geodesic_invariance(geodesic_fn, strip, tolerance=tolerance)
        report["geodesic_invariance"] = geodesic_report
        report["tests_run"].append("geodesic_invariance")
        if not geodesic_report["geodesics_invariant"]:
            report["all_passed"] = False

    return report