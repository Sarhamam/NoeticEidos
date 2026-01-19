"""Statistical rigor guards for hypothesis testing and inference."""

import numpy as np
from typing import Dict, Any, List, Optional, Union
import warnings
from scipy import stats


class StatisticalValidityError(Exception):
    """Raised when statistical validity requirements are violated."""
    pass


def validate_bootstrap_size(n_bootstrap: int,
                          min_recommended: int = 1000,
                          min_required: int = 100) -> Dict[str, Any]:
    """Validate bootstrap sample size for reliable confidence intervals.

    Parameters
    ----------
    n_bootstrap : int
        Number of bootstrap resamples
    min_recommended : int
        Recommended minimum for reliable CIs
    min_required : int
        Absolute minimum for basic validity

    Returns
    -------
    validation_report : dict
        Bootstrap size validation with recommendations

    Raises
    ------
    StatisticalValidityError
        If bootstrap size is insufficient for valid inference

    Notes
    -----
    Bootstrap CI reliability increases with sample size. For percentile CIs:
    - n=100: Very rough estimates only
    - n=1000: Adequate for most purposes
    - n=10000: High precision
    """
    if n_bootstrap < min_required:
        raise StatisticalValidityError(
            f"Bootstrap sample size {n_bootstrap} is too small for valid inference. "
            f"Minimum required: {min_required}, recommended: {min_recommended}"
        )

    reliability_level = "inadequate"
    if n_bootstrap >= min_recommended:
        reliability_level = "good"
    elif n_bootstrap >= min_required:
        reliability_level = "minimal"

    # Estimate CI standard error
    # For percentile CIs, SE ≈ sqrt(p(1-p)/n) where p is the percentile
    ci_se_5pct = np.sqrt(0.05 * 0.95 / n_bootstrap)  # 5th percentile
    ci_se_95pct = np.sqrt(0.05 * 0.95 / n_bootstrap)  # 95th percentile (same)

    report = {
        "valid": True,
        "n_bootstrap": n_bootstrap,
        "reliability_level": reliability_level,
        "min_recommended": min_recommended,
        "estimated_ci_se": float(ci_se_5pct),
        "recommendations": []
    }

    if n_bootstrap < min_recommended:
        report["recommendations"].append(
            f"Increase bootstrap samples to ≥{min_recommended} for reliable CIs"
        )

    if reliability_level == "minimal":
        warnings.warn(
            f"Bootstrap sample size {n_bootstrap} is minimal. "
            f"Consider increasing to ≥{min_recommended} for more reliable results."
        )

    return report


def check_separability_null(identical_samples1: np.ndarray,
                          identical_samples2: np.ndarray,
                          test_function: callable,
                          alpha: float = 0.05,
                          n_trials: int = 100,
                          seed: Optional[int] = None) -> Dict[str, Any]:
    """Validate that separability test correctly handles null case (identical distributions).

    Parameters
    ----------
    identical_samples1, identical_samples2 : ndarray
        Two samples from the same distribution
    test_function : callable
        Separability test function to validate
    alpha : float
        Nominal significance level
    n_trials : int
        Number of null tests to perform
    seed : int or None
        Random seed for reproducibility

    Returns
    -------
    null_validation : dict
        Null hypothesis validation results

    Raises
    ------
    StatisticalValidityError
        If test shows excessive Type I error rate
    """
    rng = np.random.default_rng(seed)

    # Ensure we have identical distributions by creating two samples from same data
    combined_data = np.concatenate([identical_samples1, identical_samples2])
    n1, n2 = len(identical_samples1), len(identical_samples2)

    false_positive_count = 0
    p_values = []

    for trial in range(n_trials):
        # Create two samples from same distribution
        shuffled = rng.permutation(combined_data)
        sample1 = shuffled[:n1]
        sample2 = shuffled[n1:n1+n2]

        try:
            # Apply separability test
            result = test_function(sample1, sample2)
            p_value = result.get("p_value", 1.0)
            p_values.append(p_value)

            # Check for false positive
            if p_value < alpha:
                false_positive_count += 1

        except Exception as e:
            warnings.warn(f"Test failed on trial {trial}: {e}")
            continue

    if len(p_values) == 0:
        raise StatisticalValidityError("All null validation trials failed")

    observed_type1_rate = false_positive_count / len(p_values)
    p_values = np.array(p_values)

    # Expected Type I error rate should be ≈ α
    # Use binomial test to check if observed rate is significantly different
    try:
        # Try new scipy interface first
        binom_result = stats.binomtest(false_positive_count, len(p_values), alpha)
        binom_p = binom_result.pvalue
    except AttributeError:
        # Fallback to older scipy interface
        binom_p = stats.binom_test(false_positive_count, len(p_values), alpha)

    # Check p-value distribution uniformity (should be uniform under null)
    ks_stat, ks_p = stats.kstest(p_values, 'uniform')

    report = {
        "valid_null_behavior": observed_type1_rate <= 2 * alpha and binom_p > 0.01,
        "observed_type1_rate": float(observed_type1_rate),
        "expected_type1_rate": alpha,
        "n_trials": len(p_values),
        "false_positives": false_positive_count,
        "binomial_test_p": float(binom_p),
        "ks_uniformity_p": float(ks_p),
        "mean_p_value": float(np.mean(p_values)),
        "p_value_distribution": {
            "min": float(np.min(p_values)),
            "max": float(np.max(p_values)),
            "median": float(np.median(p_values))
        }
    }

    # Raise error if test is miscalibrated
    if observed_type1_rate > 2 * alpha:
        raise StatisticalValidityError(
            f"Excessive Type I error rate: {observed_type1_rate:.3f} > {2*alpha:.3f}. "
            f"Test may be miscalibrated or overly liberal."
        )

    if binom_p < 0.01:
        warnings.warn(
            f"Type I error rate significantly different from nominal level "
            f"(p={binom_p:.4f}). Test calibration may be poor."
        )

    return report


def apply_multiple_testing_correction(p_values: np.ndarray,
                                    method: str = "holm",
                                    alpha: float = 0.05) -> Dict[str, Any]:
    """Apply multiple testing correction and validate results.

    Parameters
    ----------
    p_values : ndarray
        Raw p-values from multiple tests
    method : str
        Correction method: "bonferroni", "holm", "benjamini_hochberg"
    alpha : float
        Family-wise error rate or false discovery rate

    Returns
    -------
    correction_results : dict
        Multiple testing correction results and validation
    """
    p_values = np.asarray(p_values)
    n_tests = len(p_values)

    if n_tests == 1:
        # No correction needed for single test
        return {
            "corrected_p_values": p_values.copy(),
            "significant": p_values < alpha,
            "method": "none",
            "n_tests": 1,
            "n_significant_raw": int(np.sum(p_values < alpha)),
            "n_significant_corrected": int(np.sum(p_values < alpha))
        }

    # Apply correction
    if method == "bonferroni":
        corrected_p = np.minimum(p_values * n_tests, 1.0)

    elif method == "holm":
        # Holm step-down procedure
        sorted_indices = np.argsort(p_values)
        sorted_p = p_values[sorted_indices]

        corrected_sorted = np.zeros_like(sorted_p)
        for i in range(n_tests):
            corrected_sorted[i] = min(1.0, sorted_p[i] * (n_tests - i))

        # Ensure monotonicity
        for i in range(1, n_tests):
            corrected_sorted[i] = max(corrected_sorted[i], corrected_sorted[i-1])

        # Restore original order
        corrected_p = np.zeros_like(p_values)
        corrected_p[sorted_indices] = corrected_sorted

    elif method == "benjamini_hochberg":
        # Benjamini-Hochberg FDR control
        sorted_indices = np.argsort(p_values)
        sorted_p = p_values[sorted_indices]

        corrected_sorted = np.zeros_like(sorted_p)
        for i in range(n_tests):
            corrected_sorted[i] = min(1.0, sorted_p[i] * n_tests / (i + 1))

        # Ensure monotonicity (reverse)
        for i in range(n_tests - 2, -1, -1):
            corrected_sorted[i] = min(corrected_sorted[i], corrected_sorted[i + 1])

        # Restore original order
        corrected_p = np.zeros_like(p_values)
        corrected_p[sorted_indices] = corrected_sorted

    else:
        raise ValueError(f"Unknown correction method: {method}")

    # Determine significance
    significant_corrected = corrected_p < alpha
    significant_raw = p_values < alpha

    # Validation checks
    if np.any(corrected_p < p_values - 1e-10):  # Allow small numerical errors
        warnings.warn("Corrected p-values smaller than raw p-values detected")

    if np.any(corrected_p > 1.0 + 1e-10):
        warnings.warn("Corrected p-values > 1.0 detected")

    return {
        "corrected_p_values": corrected_p,
        "significant": significant_corrected,
        "method": method,
        "alpha": alpha,
        "n_tests": n_tests,
        "n_significant_raw": int(np.sum(significant_raw)),
        "n_significant_corrected": int(np.sum(significant_corrected)),
        "raw_p_values": p_values.copy(),
        "correction_severity": float(np.mean(corrected_p / np.maximum(p_values, 1e-10)))
    }


def validate_sample_size_adequacy(sample1: np.ndarray,
                                sample2: np.ndarray,
                                effect_size: float,
                                alpha: float = 0.05,
                                power: float = 0.8) -> Dict[str, Any]:
    """Validate that sample sizes are adequate for detecting given effect size.

    Parameters
    ----------
    sample1, sample2 : ndarray
        The two samples being compared
    effect_size : float
        Standardized effect size (Cohen's d)
    alpha : float
        Significance level
    power : float
        Desired statistical power

    Returns
    -------
    sample_size_report : dict
        Sample size adequacy analysis
    """
    n1, n2 = len(sample1), len(sample2)
    n_harmonic = 2 * n1 * n2 / (n1 + n2)  # Harmonic mean for unequal sizes

    # Estimate required sample size for two-sample t-test
    # Using approximate formula: n ≈ 2(z_α/2 + z_β)² / δ²
    z_alpha = stats.norm.ppf(1 - alpha/2)
    z_beta = stats.norm.ppf(power)

    if effect_size > 0:
        n_required_per_group = 2 * (z_alpha + z_beta)**2 / (effect_size**2)
        n_required_total = 2 * n_required_per_group
    else:
        n_required_per_group = float('inf')
        n_required_total = float('inf')

    # Current power estimate
    if effect_size > 0 and n_harmonic > 0:
        ncp = effect_size * np.sqrt(n_harmonic / 2)  # Non-centrality parameter
        current_power = 1 - stats.t.cdf(z_alpha, n1 + n2 - 2, ncp) + \
                       stats.t.cdf(-z_alpha, n1 + n2 - 2, ncp)
        current_power = max(0.0, min(1.0, current_power))
    else:
        current_power = 0.0

    adequate = (n1 >= n_required_per_group * 0.8 and
               n2 >= n_required_per_group * 0.8 and
               current_power >= power * 0.9)

    return {
        "adequate": adequate,
        "n1": n1,
        "n2": n2,
        "n_harmonic": float(n_harmonic),
        "effect_size": effect_size,
        "required_per_group": float(n_required_per_group),
        "required_total": float(n_required_total),
        "current_power": float(current_power),
        "target_power": power,
        "recommendations": [
            f"Increase sample sizes to ≥{n_required_per_group:.0f} per group"
        ] if not adequate else []
    }


def check_statistical_assumptions(sample1: np.ndarray,
                                sample2: np.ndarray,
                                test_type: str = "ttest") -> Dict[str, Any]:
    """Check statistical assumptions for common tests.

    Parameters
    ----------
    sample1, sample2 : ndarray
        Samples to check
    test_type : str
        Type of test: "ttest", "mannwhitney", "bootstrap"

    Returns
    -------
    assumption_report : dict
        Assumption checking results with recommendations
    """
    s1, s2 = np.asarray(sample1), np.asarray(sample2)

    # Basic checks
    finite1 = np.all(np.isfinite(s1))
    finite2 = np.all(np.isfinite(s2))

    if not (finite1 and finite2):
        return {
            "valid": False,
            "error": "Non-finite values in samples",
            "finite_sample1": finite1,
            "finite_sample2": finite2
        }

    # Remove any infinite or NaN values for further analysis
    s1_clean = s1[np.isfinite(s1)]
    s2_clean = s2[np.isfinite(s2)]

    if len(s1_clean) < 3 or len(s2_clean) < 3:
        return {
            "valid": False,
            "error": "Insufficient sample sizes after cleaning",
            "n1_clean": len(s1_clean),
            "n2_clean": len(s2_clean)
        }

    results = {
        "valid": True,
        "test_type": test_type,
        "n1": len(s1_clean),
        "n2": len(s2_clean)
    }

    if test_type == "ttest":
        # Check normality (Shapiro-Wilk for small samples, Anderson-Darling for larger)
        if len(s1_clean) <= 50:
            _, norm_p1 = stats.shapiro(s1_clean)
        else:
            norm_stat1 = stats.anderson(s1_clean, dist='norm')
            norm_p1 = 0.01 if norm_stat1.statistic > norm_stat1.critical_values[2] else 0.1

        if len(s2_clean) <= 50:
            _, norm_p2 = stats.shapiro(s2_clean)
        else:
            norm_stat2 = stats.anderson(s2_clean, dist='norm')
            norm_p2 = 0.01 if norm_stat2.statistic > norm_stat2.critical_values[2] else 0.1

        # Check equal variances (Levene's test)
        _, equal_var_p = stats.levene(s1_clean, s2_clean)

        results.update({
            "normality_p1": float(norm_p1),
            "normality_p2": float(norm_p2),
            "equal_variances_p": float(equal_var_p),
            "assumptions_met": norm_p1 > 0.05 and norm_p2 > 0.05 and equal_var_p > 0.05,
            "recommendations": []
        })

        if norm_p1 <= 0.05 or norm_p2 <= 0.05:
            results["recommendations"].append("Consider Mann-Whitney U test (non-parametric)")

        if equal_var_p <= 0.05:
            results["recommendations"].append("Use Welch's t-test (unequal variances)")

    elif test_type == "mannwhitney":
        # Mann-Whitney has fewer assumptions
        results.update({
            "assumptions_met": True,
            "note": "Mann-Whitney U test has minimal distributional assumptions"
        })

    elif test_type == "bootstrap":
        # Bootstrap assumptions
        results.update({
            "assumptions_met": True,
            "note": "Bootstrap methods are distribution-free but require adequate sample sizes"
        })

    return results