"""Stability analysis for robustness testing under perturbations."""

import numpy as np
from typing import Callable, Tuple, Dict, Any, Optional, Union
import warnings


def stability_score(stat_fn: Callable, X: np.ndarray, perturb_fn: Callable,
                   trials: int = 10, seed: Optional[int] = None) -> Tuple[float, float, float]:
    """Measure robustness of statistic under perturbations.

    Parameters
    ----------
    stat_fn : callable
        Function that takes data and returns a scalar statistic
    X : ndarray, shape (n, d)
        Original dataset
    perturb_fn : callable
        Function(X, seed) -> perturbed_X that applies perturbations
    trials : int
        Number of perturbation trials
    seed : int or None
        Random seed for reproducibility

    Returns
    -------
    mean : float
        Mean statistic value across trials
    std : float
        Standard deviation across trials
    stability : float
        Stability score: S = 1 - Std(φ) / E[φ]

    Notes
    -----
    Stability score S ∈ [0, 1]:
    - S ≈ 1: statistic is robust to perturbations
    - S ≈ 0: statistic varies significantly under perturbations
    """
    rng = np.random.default_rng(seed)

    # Collect statistic values across trials
    values = []

    for trial in range(trials):
        trial_seed = rng.integers(0, 2**31) if seed is not None else None

        try:
            # Apply perturbation
            X_perturbed = perturb_fn(X, trial_seed)

            # Compute statistic
            value = stat_fn(X_perturbed)

            if np.isfinite(value):
                values.append(float(value))
            else:
                warnings.warn(f"Non-finite statistic value in trial {trial}")

        except Exception as e:
            warnings.warn(f"Error in trial {trial}: {e}")
            continue

    if len(values) == 0:
        warnings.warn("No valid trials - returning zero stability")
        return 0.0, 0.0, 0.0

    values = np.array(values)

    # Compute statistics
    mean_val = np.mean(values)
    std_val = np.std(values, ddof=1) if len(values) > 1 else 0.0

    # Stability score: 1 - coefficient of variation
    if abs(mean_val) > 1e-12:
        stability = 1.0 - std_val / abs(mean_val)
    else:
        stability = 1.0 if std_val < 1e-12 else 0.0

    # Clamp to [0, 1]
    stability = max(0.0, min(1.0, stability))

    return float(mean_val), float(std_val), float(stability)


def noise_perturbation(noise_level: float = 0.1) -> Callable:
    """Create a noise perturbation function.

    Parameters
    ----------
    noise_level : float
        Standard deviation of Gaussian noise relative to data scale

    Returns
    -------
    perturb_fn : callable
        Function that adds Gaussian noise to data
    """
    def perturb_fn(X: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
        rng = np.random.default_rng(seed)

        # Estimate data scale
        data_scale = np.std(X, axis=0)
        data_scale = np.where(data_scale > 1e-12, data_scale, 1.0)

        # Add proportional noise
        noise = rng.normal(0, noise_level, size=X.shape) * data_scale
        return X + noise

    return perturb_fn


def subsample_perturbation(subsample_ratio: float = 0.9) -> Callable:
    """Create a subsampling perturbation function.

    Parameters
    ----------
    subsample_ratio : float
        Fraction of points to keep (between 0 and 1)

    Returns
    -------
    perturb_fn : callable
        Function that randomly subsamples data points
    """
    def perturb_fn(X: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
        rng = np.random.default_rng(seed)

        n = X.shape[0]
        k = max(1, int(n * subsample_ratio))

        # Random subsample
        indices = rng.choice(n, size=k, replace=False)
        return X[indices]

    return perturb_fn


def bootstrap_perturbation() -> Callable:
    """Create a bootstrap perturbation function.

    Returns
    -------
    perturb_fn : callable
        Function that bootstrap resamples the data
    """
    def perturb_fn(X: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
        rng = np.random.default_rng(seed)

        n = X.shape[0]
        indices = rng.choice(n, size=n, replace=True)
        return X[indices]

    return perturb_fn


def coordinate_perturbation(coord_noise: float = 0.1) -> Callable:
    """Create a coordinate-wise perturbation function.

    Parameters
    ----------
    coord_noise : float
        Noise level applied to each coordinate independently

    Returns
    -------
    perturb_fn : callable
        Function that perturbs each coordinate independently
    """
    def perturb_fn(X: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
        rng = np.random.default_rng(seed)

        # Apply different noise to each coordinate
        noise_scale = np.std(X, axis=0) * coord_noise
        noise = rng.normal(0, 1, size=X.shape) * noise_scale
        return X + noise

    return perturb_fn


def stability_curve(stat_fn: Callable, X: np.ndarray, perturb_fn: Callable,
                   noise_levels: np.ndarray, trials: int = 10,
                   seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute stability as a function of perturbation strength.

    Parameters
    ----------
    stat_fn : callable
        Statistic function
    X : ndarray
        Dataset
    perturb_fn : callable
        Base perturbation function (will be scaled by noise levels)
    noise_levels : ndarray
        Array of perturbation strengths
    trials : int
        Number of trials per noise level
    seed : int or None
        Random seed

    Returns
    -------
    noise_levels : ndarray
        Perturbation strengths
    stabilities : ndarray
        Stability scores at each noise level
    std_errors : ndarray
        Standard errors of stability estimates
    """
    stabilities = []
    std_errors = []

    rng = np.random.default_rng(seed)

    for noise_level in noise_levels:
        # Create scaled perturbation function
        def scaled_perturb_fn(X_in, seed_in):
            return perturb_fn(X_in, seed_in, noise_level)

        try:
            _, _, stability = stability_score(stat_fn, X, scaled_perturb_fn,
                                            trials=trials, seed=rng.integers(0, 2**31))
            stabilities.append(stability)

            # Estimate standard error (crude approximation)
            std_error = np.sqrt(stability * (1 - stability) / trials)
            std_errors.append(std_error)

        except Exception as e:
            warnings.warn(f"Error at noise level {noise_level}: {e}")
            stabilities.append(0.0)
            std_errors.append(0.0)

    return noise_levels, np.array(stabilities), np.array(std_errors)


def multi_statistic_stability(stat_fns: Dict[str, Callable], X: np.ndarray,
                             perturb_fn: Callable, trials: int = 10,
                             seed: Optional[int] = None) -> Dict[str, Tuple[float, float, float]]:
    """Compute stability for multiple statistics simultaneously.

    Parameters
    ----------
    stat_fns : dict
        Dictionary mapping names to statistic functions
    X : ndarray
        Dataset
    perturb_fn : callable
        Perturbation function
    trials : int
        Number of trials
    seed : int or None
        Random seed

    Returns
    -------
    results : dict
        Dictionary mapping names to (mean, std, stability) tuples
    """
    results = {}
    rng = np.random.default_rng(seed)

    for name, stat_fn in stat_fns.items():
        try:
            mean_val, std_val, stability = stability_score(
                stat_fn, X, perturb_fn, trials=trials,
                seed=rng.integers(0, 2**31) if seed is not None else None
            )
            results[name] = (mean_val, std_val, stability)
        except Exception as e:
            warnings.warn(f"Error computing stability for {name}: {e}")
            results[name] = (0.0, 0.0, 0.0)

    return results


def relative_stability(stat_fn: Callable, X: np.ndarray, perturb_fn: Callable,
                      baseline_X: Optional[np.ndarray] = None,
                      trials: int = 10, seed: Optional[int] = None) -> float:
    """Compute stability relative to a baseline dataset.

    Parameters
    ----------
    stat_fn : callable
        Statistic function
    X : ndarray
        Test dataset
    perturb_fn : callable
        Perturbation function
    baseline_X : ndarray or None
        Baseline dataset (if None, use X)
    trials : int
        Number of trials
    seed : int or None
        Random seed

    Returns
    -------
    relative_stability : float
        Stability of X relative to baseline
    """
    if baseline_X is None:
        baseline_X = X

    # Compute stability for test dataset
    _, _, stability_test = stability_score(stat_fn, X, perturb_fn,
                                         trials=trials, seed=seed)

    # Compute stability for baseline
    rng = np.random.default_rng(seed)
    baseline_seed = rng.integers(0, 2**31) if seed is not None else None
    _, _, stability_baseline = stability_score(stat_fn, baseline_X, perturb_fn,
                                             trials=trials, seed=baseline_seed)

    # Return ratio (with protection against division by zero)
    if stability_baseline > 1e-12:
        return stability_test / stability_baseline
    else:
        return 1.0 if stability_test < 1e-12 else float('inf')