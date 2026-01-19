"""Separability testing for additive vs multiplicative transport modes."""

import warnings
from typing import Any, Dict, Optional

import numpy as np
from scipy import stats


def separability_test(
    phi_add: np.ndarray,
    phi_mult: np.ndarray,
    method: str = "bootstrap",
    trials: int = 1000,
    alpha: float = 0.05,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Test if additive vs multiplicative statistics differ significantly.

    Parameters
    ----------
    phi_add : ndarray, shape (n_add,)
        Statistic values from additive mode
    phi_mult : ndarray, shape (n_mult,)
        Statistic values from multiplicative mode
    method : str
        Statistical test method:
        - "bootstrap": Bootstrap confidence interval for difference
        - "ttest": Two-sample t-test
        - "mannwhitney": Mann-Whitney U test (non-parametric)
        - "permutation": Permutation test
    trials : int
        Number of bootstrap/permutation trials
    alpha : float
        Significance level (Type I error rate)
    seed : int or None
        Random seed for reproducibility

    Returns
    -------
    result : dict
        - "p_value": float, p-value of test
        - "ci": tuple (low, high), confidence interval for difference
        - "effect_size": float, standardized effect size
        - "separable": bool, whether modes are significantly different
        - "method": str, test method used
        - "statistic": float, test statistic value
    """
    phi_add = np.asarray(phi_add).flatten()
    phi_mult = np.asarray(phi_mult).flatten()

    if len(phi_add) == 0 or len(phi_mult) == 0:
        raise ValueError("Both arrays must have at least one element")

    # Compute basic statistics
    mean_add = np.mean(phi_add)
    mean_mult = np.mean(phi_mult)
    diff_mean = mean_add - mean_mult

    # Effect size (Cohen's d)
    pooled_std = np.sqrt(
        (
            (len(phi_add) - 1) * np.var(phi_add, ddof=1)
            + (len(phi_mult) - 1) * np.var(phi_mult, ddof=1)
        )
        / (len(phi_add) + len(phi_mult) - 2)
    )

    if pooled_std > 1e-12:
        effect_size = abs(diff_mean) / pooled_std
    else:
        effect_size = 0.0

    rng = np.random.default_rng(seed)

    if method == "bootstrap":
        return _bootstrap_test(
            phi_add, phi_mult, trials, alpha, rng, diff_mean, effect_size
        )

    elif method == "ttest":
        return _ttest(phi_add, phi_mult, alpha, diff_mean, effect_size)

    elif method == "mannwhitney":
        return _mannwhitney_test(phi_add, phi_mult, alpha, diff_mean, effect_size)

    elif method == "permutation":
        return _permutation_test(
            phi_add, phi_mult, trials, alpha, rng, diff_mean, effect_size
        )

    else:
        raise ValueError(f"Unknown method: {method}")


def _bootstrap_test(
    phi_add: np.ndarray,
    phi_mult: np.ndarray,
    trials: int,
    alpha: float,
    rng: np.random.Generator,
    diff_mean: float,
    effect_size: float,
) -> Dict[str, Any]:
    """Bootstrap confidence interval test."""
    # Bootstrap resampling
    diffs = []

    for _ in range(trials):
        # Resample with replacement
        add_sample = rng.choice(phi_add, size=len(phi_add), replace=True)
        mult_sample = rng.choice(phi_mult, size=len(phi_mult), replace=True)

        diff = np.mean(add_sample) - np.mean(mult_sample)
        diffs.append(diff)

    diffs = np.array(diffs)

    # Confidence interval
    ci_low = np.percentile(diffs, 100 * alpha / 2)
    ci_high = np.percentile(diffs, 100 * (1 - alpha / 2))

    # Test: reject H₀ if CI excludes 0
    separable = not (ci_low <= 0 <= ci_high)

    # Approximate p-value: fraction of bootstrap samples with |diff| > |observed|
    p_value = np.mean(np.abs(diffs) >= abs(diff_mean))

    return {
        "p_value": float(p_value),
        "ci": (float(ci_low), float(ci_high)),
        "effect_size": float(effect_size),
        "separable": bool(separable),
        "method": "bootstrap",
        "statistic": float(diff_mean),
    }


def _ttest(
    phi_add: np.ndarray,
    phi_mult: np.ndarray,
    alpha: float,
    diff_mean: float,
    effect_size: float,
) -> Dict[str, Any]:
    """Two-sample t-test."""
    try:
        statistic, p_value = stats.ttest_ind(phi_add, phi_mult, equal_var=False)

        # Confidence interval for difference (Welch's t-test)
        n1, n2 = len(phi_add), len(phi_mult)
        var1, var2 = np.var(phi_add, ddof=1), np.var(phi_mult, ddof=1)

        # Welch's degrees of freedom
        se_diff = np.sqrt(var1 / n1 + var2 / n2)
        dof = (var1 / n1 + var2 / n2) ** 2 / (
            (var1 / n1) ** 2 / (n1 - 1) + (var2 / n2) ** 2 / (n2 - 1)
        )

        t_crit = stats.t.ppf(1 - alpha / 2, dof)
        margin = t_crit * se_diff

        ci_low = diff_mean - margin
        ci_high = diff_mean + margin

        separable = p_value < alpha

        return {
            "p_value": float(p_value),
            "ci": (float(ci_low), float(ci_high)),
            "effect_size": float(effect_size),
            "separable": bool(separable),
            "method": "ttest",
            "statistic": float(statistic),
        }

    except Exception as e:
        warnings.warn(f"t-test failed: {e}", stacklevel=2)
        return {
            "p_value": 1.0,
            "ci": (-float("inf"), float("inf")),
            "effect_size": float(effect_size),
            "separable": False,
            "method": "ttest",
            "statistic": 0.0,
        }


def _mannwhitney_test(
    phi_add: np.ndarray,
    phi_mult: np.ndarray,
    alpha: float,
    diff_mean: float,
    effect_size: float,
) -> Dict[str, Any]:
    """Mann-Whitney U test (non-parametric)."""
    try:
        statistic, p_value = stats.mannwhitneyu(
            phi_add, phi_mult, alternative="two-sided"
        )

        separable = p_value < alpha

        # Crude CI estimate based on rank statistics
        # This is approximate - exact CI for Mann-Whitney is complex
        combined = np.concatenate([phi_add, phi_mult])
        np.sort(combined)

        # Use difference in medians as point estimate
        median_diff = np.median(phi_add) - np.median(phi_mult)

        # Rough CI bounds (not exact)
        n_total = len(combined)
        ci_width = 1.96 * np.std(combined) / np.sqrt(n_total)  # Approximate
        ci_low = median_diff - ci_width
        ci_high = median_diff + ci_width

        return {
            "p_value": float(p_value),
            "ci": (float(ci_low), float(ci_high)),
            "effect_size": float(effect_size),
            "separable": bool(separable),
            "method": "mannwhitney",
            "statistic": float(statistic),
        }

    except Exception as e:
        warnings.warn(f"Mann-Whitney test failed: {e}", stacklevel=2)
        return {
            "p_value": 1.0,
            "ci": (-float("inf"), float("inf")),
            "effect_size": float(effect_size),
            "separable": False,
            "method": "mannwhitney",
            "statistic": 0.0,
        }


def _permutation_test(
    phi_add: np.ndarray,
    phi_mult: np.ndarray,
    trials: int,
    alpha: float,
    rng: np.random.Generator,
    diff_mean: float,
    effect_size: float,
) -> Dict[str, Any]:
    """Permutation test."""
    # Combine samples
    combined = np.concatenate([phi_add, phi_mult])
    n1 = len(phi_add)
    n_total = len(combined)

    # Observed test statistic
    observed_diff = abs(diff_mean)

    # Permutation distribution
    perm_diffs = []

    for _ in range(trials):
        # Random permutation
        perm_indices = rng.permutation(n_total)
        perm_add = combined[perm_indices[:n1]]
        perm_mult = combined[perm_indices[n1:]]

        perm_diff = abs(np.mean(perm_add) - np.mean(perm_mult))
        perm_diffs.append(perm_diff)

    perm_diffs = np.array(perm_diffs)

    # P-value: fraction of permutations with |diff| >= observed
    p_value = np.mean(perm_diffs >= observed_diff)

    # Confidence interval based on permutation distribution
    ci_low = np.percentile(perm_diffs, 100 * alpha / 2)
    ci_high = np.percentile(perm_diffs, 100 * (1 - alpha / 2))

    # Adjust for two-sided test
    ci_low = -ci_high  # Symmetric around 0 for permutation null
    ci_high = np.percentile(perm_diffs, 100 * (1 - alpha / 2))

    separable = p_value < alpha

    return {
        "p_value": float(p_value),
        "ci": (float(ci_low), float(ci_high)),
        "effect_size": float(effect_size),
        "separable": bool(separable),
        "method": "permutation",
        "statistic": float(observed_diff),
    }


def mode_comparison_matrix(
    datasets: Dict[str, np.ndarray],
    modes: Dict[str, str] = None,
    method: str = "bootstrap",
    alpha: float = 0.05,
    seed: Optional[int] = None,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Compare multiple datasets across transport modes.

    Parameters
    ----------
    datasets : dict
        Dictionary mapping dataset names to arrays of statistic values
    modes : dict or None
        Dictionary mapping dataset names to mode labels ("additive" or "multiplicative")
        If None, assumes names contain mode information
    method : str
        Statistical test method
    alpha : float
        Significance level
    seed : int or None
        Random seed

    Returns
    -------
    comparison_matrix : dict
        Nested dictionary with pairwise comparison results
    """
    if modes is None:
        # Try to infer modes from dataset names
        modes = {}
        for name in datasets.keys():
            if "add" in name.lower() or "gaussian" in name.lower():
                modes[name] = "additive"
            elif "mult" in name.lower() or "poisson" in name.lower():
                modes[name] = "multiplicative"
            else:
                modes[name] = "unknown"

    results = {}
    dataset_names = list(datasets.keys())

    for i, name1 in enumerate(dataset_names):
        results[name1] = {}

        for j, name2 in enumerate(dataset_names):
            if i >= j:
                continue  # Skip self-comparison and duplicates

            # Determine if this is a cross-mode comparison
            mode1 = modes.get(name1, "unknown")
            mode2 = modes.get(name2, "unknown")
            is_cross_mode = (mode1 != mode2) and ("unknown" not in [mode1, mode2])

            try:
                result = separability_test(
                    datasets[name1],
                    datasets[name2],
                    method=method,
                    alpha=alpha,
                    seed=seed,
                )
                result["cross_mode"] = is_cross_mode
                result["modes"] = (mode1, mode2)

            except Exception as e:
                warnings.warn(
                    f"Comparison {name1} vs {name2} failed: {e}", stacklevel=2
                )
                result = {
                    "p_value": 1.0,
                    "ci": (-float("inf"), float("inf")),
                    "effect_size": 0.0,
                    "separable": False,
                    "method": method,
                    "statistic": 0.0,
                    "cross_mode": is_cross_mode,
                    "modes": (mode1, mode2),
                }

            results[name1][name2] = result

    return results


def effect_size_interpretation(effect_size: float) -> str:
    """Interpret Cohen's d effect size.

    Parameters
    ----------
    effect_size : float
        Standardized effect size

    Returns
    -------
    interpretation : str
        Qualitative interpretation
    """
    if effect_size < 0.2:
        return "negligible"
    elif effect_size < 0.5:
        return "small"
    elif effect_size < 0.8:
        return "medium"
    else:
        return "large"


def power_analysis(
    phi_add: np.ndarray,
    phi_mult: np.ndarray,
    alpha: float = 0.05,
    n_range: np.ndarray = None,
) -> Dict[str, np.ndarray]:
    """Estimate statistical power for different sample sizes.

    Parameters
    ----------
    phi_add : ndarray
        Additive mode values (for effect size estimation)
    phi_mult : ndarray
        Multiplicative mode values (for effect size estimation)
    alpha : float
        Significance level
    n_range : ndarray or None
        Sample sizes to evaluate

    Returns
    -------
    power_results : dict
        - "n_values": sample sizes
        - "power": estimated power at each sample size
    """
    # Estimate effect size from current data
    mean_diff = np.mean(phi_add) - np.mean(phi_mult)
    pooled_std = np.sqrt((np.var(phi_add, ddof=1) + np.var(phi_mult, ddof=1)) / 2)

    if pooled_std > 1e-12:
        effect_size = abs(mean_diff) / pooled_std
    else:
        effect_size = 0.0

    if n_range is None:
        n_range = np.array([5, 10, 20, 50, 100, 200, 500])

    powers = []

    for n in n_range:
        # Two-sample t-test power calculation (approximate)
        # Standard error of difference for equal sample sizes
        pooled_std * np.sqrt(2 / n)

        # Non-centrality parameter
        ncp = effect_size * np.sqrt(n / 2)

        # Degrees of freedom for two-sample t-test
        dof = 2 * n - 2

        # Critical value
        t_crit = stats.t.ppf(1 - alpha / 2, dof)

        # Power = P(|T| > t_crit | ncp)
        # Approximate using normal distribution for large n
        if n >= 30:
            power = 1 - stats.norm.cdf(t_crit - ncp) + stats.norm.cdf(-t_crit - ncp)
        else:
            # Use non-central t-distribution (approximate)
            power = 1 - stats.t.cdf(t_crit, dof, ncp) + stats.t.cdf(-t_crit, dof, ncp)

        powers.append(max(0.0, min(1.0, power)))

    return {"n_values": n_range, "power": np.array(powers), "effect_size": effect_size}
