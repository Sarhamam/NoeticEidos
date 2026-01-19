"""Mellin balance analysis and s=0.5 optimization for dual transport coupling."""

import warnings
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
from scipy.sparse import csr_matrix

from graphs.knn import build_graph
from graphs.laplacian import laplacian


def mellin_coupled_stat(
    X: np.ndarray,
    stat_fn: Callable,
    s: float = 0.5,
    k: int = 16,
    sigma: Union[str, float] = "median",
    tau: float = 1.0,
    eps: float = 1e-6,
    seed: Optional[int] = None,
) -> float:
    """Compute statistic on Mellin-coupled graph.

    Parameters
    ----------
    X : ndarray, shape (n, d)
        Input dataset
    stat_fn : callable
        Function that takes Laplacian and returns scalar statistic
    s : float
        Mellin coupling parameter (s=0.5 is canonical balance point)
    k : int
        Number of k-NN neighbors
    sigma : str or float
        Bandwidth for additive (Gaussian) kernel
    tau : float
        Bandwidth for multiplicative (Poisson) kernel
    eps : float
        Regularization for log-map in multiplicative mode
    seed : int or None
        Random seed for reproducibility

    Returns
    -------
    statistic : float
        Computed statistic on coupled graph

    Notes
    -----
    Constructs coupled adjacency matrix:
    A_s = (A_add)^s ⊙ (A_mult)^(1-s)

    where ⊙ denotes Hadamard (element-wise) product.
    At s=0.5, this represents the canonical Mellin balance.
    """
    if not (0.0 <= s <= 1.0):
        raise ValueError("Mellin parameter s must be in [0, 1]")

    # Build additive graph
    A_add = build_graph(X, mode="additive", k=k, sigma=sigma, seed=seed)

    # Build multiplicative graph
    A_mult = build_graph(X, mode="multiplicative", k=k, tau=tau, eps=eps, seed=seed)

    # Ensure same sparsity pattern for proper coupling
    # Take union of edges (or intersection - depends on desired behavior)
    A_add_dense = A_add.toarray() if hasattr(A_add, "toarray") else A_add
    A_mult_dense = A_mult.toarray() if hasattr(A_mult, "toarray") else A_mult

    # Mellin coupling: A_s = (A_add)^s ⊙ (A_mult)^(1-s)
    # Handle zero entries carefully
    A_coupled = np.zeros_like(A_add_dense)

    # Only couple where both graphs have edges
    mask = (A_add_dense > 0) & (A_mult_dense > 0)

    if np.any(mask):
        if s == 0.0:
            A_coupled[mask] = A_mult_dense[mask]
        elif s == 1.0:
            A_coupled[mask] = A_add_dense[mask]
        else:
            # Use logarithmic coupling for numerical stability
            log_A_add = np.log(A_add_dense[mask] + eps)
            log_A_mult = np.log(A_mult_dense[mask] + eps)
            log_A_coupled = s * log_A_add + (1 - s) * log_A_mult
            A_coupled[mask] = np.exp(log_A_coupled)

    # Convert back to sparse format
    A_coupled_sparse = csr_matrix(A_coupled)

    # Build Laplacian
    L_coupled = laplacian(A_coupled_sparse, normalized=True)

    # Compute statistic
    try:
        return float(stat_fn(L_coupled))
    except Exception as e:
        warnings.warn(f"Statistic computation failed: {e}", stacklevel=2)
        return 0.0


def balance_curve(
    X: np.ndarray, stat_fn: Callable, s_range: np.ndarray = None, **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate statistic across Mellin coupling parameters.

    Parameters
    ----------
    X : ndarray, shape (n, d)
        Input dataset
    stat_fn : callable
        Statistic function
    s_range : ndarray or None
        Range of s values to evaluate (default: 21 points from 0 to 1)
    **kwargs
        Additional arguments for mellin_coupled_stat

    Returns
    -------
    s_values : ndarray
        Mellin parameter values
    stat_values : ndarray
        Statistic values at each s
    """
    if s_range is None:
        s_range = np.linspace(0.0, 1.0, 21)

    stat_values = []

    for s in s_range:
        try:
            stat_val = mellin_coupled_stat(X, stat_fn, s=s, **kwargs)
            stat_values.append(stat_val)
        except Exception as e:
            warnings.warn(f"Failed to compute statistic at s={s}: {e}", stacklevel=2)
            stat_values.append(0.0)

    return s_range, np.array(stat_values)


def balance_score(
    X: np.ndarray,
    stat_fn: Callable,
    s_range: np.ndarray = None,
    target_s: float = 0.5,
    **kwargs,
) -> Tuple[float, np.ndarray, Dict[str, Any]]:
    """Find Mellin parameter that maximizes statistic stability.

    Parameters
    ----------
    X : ndarray, shape (n, d)
        Input dataset
    stat_fn : callable
        Statistic function (higher values = more stable)
    s_range : ndarray or None
        Range of s values to search (default: focused around 0.5)
    target_s : float
        Expected optimal value (default: 0.5 for canonical balance)
    **kwargs
        Additional arguments for mellin_coupled_stat

    Returns
    -------
    best_s : float
        Mellin parameter that maximizes statistic
    curve : ndarray
        Full stability curve
    info : dict
        Additional information about optimization

    Notes
    -----
    Expectation: maximum should occur at s=0.5 (canonical Mellin balance)
    for properly balanced data and statistics.
    """
    if s_range is None:
        s_range = np.linspace(0.3, 0.7, 21)  # Focus around canonical point

    s_values, stat_values = balance_curve(X, stat_fn, s_range=s_range, **kwargs)

    if len(stat_values) == 0 or np.all(~np.isfinite(stat_values)):
        warnings.warn("No valid statistic values - returning default", stacklevel=2)
        return target_s, np.array([]), {"converged": False, "peak_quality": 0.0}

    # Find maximum
    max_idx = np.argmax(stat_values)
    best_s = s_values[max_idx]
    max_stat = stat_values[max_idx]

    # Assess peak quality
    peak_quality = _assess_peak_quality(s_values, stat_values, max_idx, target_s)

    info = {
        "converged": peak_quality > 0.5,
        "peak_quality": peak_quality,
        "max_statistic": max_stat,
        "deviation_from_target": abs(best_s - target_s),
        "s_range": s_values,
        "stat_values": stat_values,
    }

    return float(best_s), stat_values, info


def _assess_peak_quality(
    s_values: np.ndarray, stat_values: np.ndarray, max_idx: int, target_s: float
) -> float:
    """Assess quality of peak in balance curve.

    Returns a score in [0, 1] indicating how well-defined the peak is.
    """
    if len(stat_values) < 3:
        return 0.0

    n = len(stat_values)
    best_s = s_values[max_idx]

    # Factor 1: How close to target (s=0.5)
    target_score = 1.0 - min(1.0, 2 * abs(best_s - target_s))

    # Factor 2: Peak prominence (how much higher than neighbors)
    prominence_score = 0.0
    if 0 < max_idx < n - 1:
        left_val = stat_values[max_idx - 1]
        right_val = stat_values[max_idx + 1]
        max_val = stat_values[max_idx]

        if max_val > max(left_val, right_val):
            prominence_score = min(
                1.0, (max_val - max(left_val, right_val)) / max(1e-12, max_val)
            )

    # Factor 3: Curve smoothness (penalize noise)
    smoothness_score = 0.0
    if n >= 5:
        # Compute second differences to detect smoothness
        second_diffs = np.diff(stat_values, n=2)
        noise_level = np.std(second_diffs) if len(second_diffs) > 0 else 0.0
        signal_level = np.std(stat_values)

        if signal_level > 1e-12:
            smoothness_score = max(0.0, 1.0 - noise_level / signal_level)

    # Combine factors
    quality = 0.5 * target_score + 0.3 * prominence_score + 0.2 * smoothness_score
    return min(1.0, max(0.0, quality))


def mellin_stability_test(
    X: np.ndarray,
    stat_fn: Callable,
    trials: int = 10,
    noise_level: float = 0.1,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Test stability of Mellin balance under perturbations.

    Parameters
    ----------
    X : ndarray, shape (n, d)
        Input dataset
    stat_fn : callable
        Statistic function
    trials : int
        Number of perturbation trials
    noise_level : float
        Noise level for perturbations
    seed : int or None
        Random seed

    Returns
    -------
    results : dict
        Stability test results including optimal s distribution
    """
    rng = np.random.default_rng(seed)

    optimal_s_values = []
    peak_qualities = []

    for trial in range(trials):
        trial_seed = rng.integers(0, 2**31) if seed is not None else None

        # Add noise to data
        noise = rng.normal(0, noise_level * np.std(X, axis=0), size=X.shape)
        X_perturbed = X + noise

        try:
            best_s, _, info = balance_score(X_perturbed, stat_fn, seed=trial_seed)
            optimal_s_values.append(best_s)
            peak_qualities.append(info["peak_quality"])

        except Exception as e:
            warnings.warn(f"Trial {trial} failed: {e}", stacklevel=2)
            continue

    if len(optimal_s_values) == 0:
        return {
            "mean_optimal_s": 0.5,
            "std_optimal_s": float("inf"),
            "stability_score": 0.0,
            "fraction_converged": 0.0,
            "mean_peak_quality": 0.0,
        }

    optimal_s_values = np.array(optimal_s_values)
    peak_qualities = np.array(peak_qualities)

    # Compute stability metrics
    mean_s = np.mean(optimal_s_values)
    std_s = np.std(optimal_s_values, ddof=1) if len(optimal_s_values) > 1 else 0.0

    # Stability score: how consistently we find s ≈ 0.5
    stability_score = 1.0 - min(1.0, std_s / 0.1)  # Normalize by reasonable scale

    fraction_converged = np.mean(peak_qualities > 0.5)
    mean_peak_quality = np.mean(peak_qualities)

    return {
        "mean_optimal_s": float(mean_s),
        "std_optimal_s": float(std_s),
        "stability_score": float(stability_score),
        "fraction_converged": float(fraction_converged),
        "mean_peak_quality": float(mean_peak_quality),
        "optimal_s_values": optimal_s_values,
        "peak_qualities": peak_qualities,
    }


def cross_mode_balance_analysis(
    X: np.ndarray,
    stat_fns: Dict[str, Callable],
    s_resolution: int = 21,
    seed: Optional[int] = None,
) -> Dict[str, Dict[str, Any]]:
    """Comprehensive balance analysis across multiple statistics.

    Parameters
    ----------
    X : ndarray, shape (n, d)
        Input dataset
    stat_fns : dict
        Dictionary mapping names to statistic functions
    s_resolution : int
        Number of points in s_range
    seed : int or None
        Random seed

    Returns
    -------
    results : dict
        Analysis results for each statistic
    """
    s_range = np.linspace(0.0, 1.0, s_resolution)
    results = {}

    for stat_name, stat_fn in stat_fns.items():
        try:
            # Balance curve
            s_values, stat_values = balance_curve(
                X, stat_fn, s_range=s_range, seed=seed
            )

            # Optimal point
            best_s, _, info = balance_score(X, stat_fn, s_range=s_range, seed=seed)

            # Stability test
            stability_info = mellin_stability_test(X, stat_fn, trials=5, seed=seed)

            results[stat_name] = {
                "balance_curve": (s_values, stat_values),
                "optimal_s": best_s,
                "peak_info": info,
                "stability": stability_info,
            }

        except Exception as e:
            warnings.warn(f"Analysis failed for {stat_name}: {e}", stacklevel=2)
            results[stat_name] = {
                "balance_curve": (np.array([]), np.array([])),
                "optimal_s": 0.5,
                "peak_info": {"converged": False},
                "stability": {"stability_score": 0.0},
            }

    return results


def additive_multiplicative_interpolation(
    X: np.ndarray,
    stat_fn: Callable,
    s_fine: np.ndarray = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Detailed analysis of additive-multiplicative interpolation.

    Parameters
    ----------
    X : ndarray, shape (n, d)
        Input dataset
    stat_fn : callable
        Statistic function
    s_fine : ndarray or None
        Fine-grained s values for interpolation study
    seed : int or None
        Random seed

    Returns
    -------
    analysis : dict
        Detailed interpolation analysis including derivatives and curvature
    """
    if s_fine is None:
        s_fine = np.linspace(0.0, 1.0, 101)  # Fine resolution

    s_values, stat_values = balance_curve(X, stat_fn, s_range=s_fine, seed=seed)

    if len(stat_values) < 10:
        return {"error": "Insufficient data points for interpolation analysis"}

    # Compute derivatives (finite differences)
    ds = s_values[1] - s_values[0] if len(s_values) > 1 else 1.0
    first_deriv = np.gradient(stat_values, ds)
    second_deriv = np.gradient(first_deriv, ds)

    # Find critical points (where derivative ≈ 0)
    critical_indices = []
    for i in range(1, len(first_deriv) - 1):
        if first_deriv[i - 1] * first_deriv[i + 1] <= 0 and abs(
            first_deriv[i]
        ) < 0.1 * np.std(first_deriv):
            critical_indices.append(i)

    critical_points = [(s_values[i], stat_values[i]) for i in critical_indices]

    # Assess monotonicity in each half
    mid_idx = len(s_values) // 2
    left_monotonic = np.all(np.diff(stat_values[:mid_idx]) >= -1e-6)
    right_monotonic = np.all(np.diff(stat_values[mid_idx:]) <= 1e-6)

    # Overall curvature analysis
    mean_curvature = np.mean(np.abs(second_deriv))
    max_curvature_idx = np.argmax(np.abs(second_deriv))

    return {
        "s_values": s_values,
        "stat_values": stat_values,
        "first_derivative": first_deriv,
        "second_derivative": second_deriv,
        "critical_points": critical_points,
        "left_monotonic": left_monotonic,
        "right_monotonic": right_monotonic,
        "symmetric_about_half": left_monotonic and right_monotonic,
        "mean_curvature": float(mean_curvature),
        "max_curvature_point": (
            s_values[max_curvature_idx],
            stat_values[max_curvature_idx],
        ),
        "additive_value": stat_values[0] if len(stat_values) > 0 else 0.0,
        "multiplicative_value": stat_values[-1] if len(stat_values) > 0 else 0.0,
        "balanced_value": (
            stat_values[len(stat_values) // 2] if len(stat_values) > 0 else 0.0
        ),
    }
