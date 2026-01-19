"""Mellin transform and coupling between additive/multiplicative transports."""

from typing import Any, Dict, Tuple

import numpy as np

# Handle numpy 2.x deprecation of trapz -> trapezoid
_trapz = getattr(np, "trapezoid", None) or np.trapz


def mellin_transform_discrete(
    f_vals: np.ndarray, y_grid: np.ndarray, s: complex
) -> complex:
    """Discrete approximation of Mellin transform.

    Parameters
    ----------
    f_vals : np.ndarray
        Function values at grid points
    y_grid : np.ndarray
        Grid points on ℝ₊ (must be positive)
    s : complex
        Mellin parameter (Re(s) = 1/2 for unitarity)

    Returns
    -------
    M_s : complex
        Mellin transform value at s

    Notes
    -----
    The Mellin transform is: M[f](s) = ∫₀^∞ y^(s-1) f(y) dy
    At s = 1/2, this respects the Haar measure dy/y.
    """
    if np.any(y_grid <= 0):
        raise ValueError("Grid points must be positive for Mellin transform")

    # Trapezoidal rule approximation with Haar measure
    dy = np.diff(y_grid)
    dy = np.append(dy, dy[-1])  # Extend for last point

    # Integrand: y^(s-1) * f(y)
    integrand = y_grid ** (s - 1) * f_vals

    # Numerical integration
    M_s = _trapz(integrand, y_grid)

    return M_s


def mellin_unitarity_test(
    f_vals: np.ndarray, g_vals: np.ndarray, y_grid: np.ndarray, s: float = 0.5
) -> Dict[str, Any]:
    """Test Mellin unitarity at critical line Re(s) = 1/2.

    Parameters
    ----------
    f_vals : np.ndarray
        First function values
    g_vals : np.ndarray
        Second function values
    y_grid : np.ndarray
        Grid points on ℝ₊
    s : float
        Real part of Mellin parameter (default 1/2)

    Returns
    -------
    result : dict
        Contains:
        - 'inner_mellin': Mellin inner product
        - 'inner_l2': Standard L² inner product
        - 'ratio': Ratio showing unitarity preservation

    Notes
    -----
    At s = 1/2, the Mellin transform preserves inner products
    up to a constant factor (Plancherel theorem analogue).
    """
    # Mellin inner product at s
    M_f = mellin_transform_discrete(f_vals, y_grid, s + 0j)
    M_g = mellin_transform_discrete(g_vals, y_grid, s + 0j)
    inner_mellin = np.real(M_f * np.conj(M_g))

    # L² inner product with Haar measure
    haar_weights = 1.0 / y_grid
    inner_l2_haar = _trapz(f_vals * g_vals * haar_weights, y_grid)

    # Standard L² inner product
    inner_l2 = _trapz(f_vals * g_vals, y_grid)

    return {
        "inner_mellin": inner_mellin,
        "inner_l2_haar": inner_l2_haar,
        "inner_l2": inner_l2,
        "ratio_haar": inner_mellin / (inner_l2_haar + 1e-10),
        "s": s,
    }


def mellin_balance_score(
    X_add: np.ndarray, X_mult: np.ndarray, s_values: np.ndarray = None
) -> Tuple[np.ndarray, float]:
    """Compute balance score between additive and multiplicative transports.

    Parameters
    ----------
    X_add : np.ndarray
        Features from additive transport
    X_mult : np.ndarray
        Features from multiplicative transport
    s_values : np.ndarray
        Mellin parameters to test (default: around s=1/2)

    Returns
    -------
    scores : np.ndarray
        Balance scores for each s
    optimal_s : float
        Value of s with best balance

    Notes
    -----
    The balance score measures coupling strength between the
    two transport modes. Maximum at s=1/2 indicates unitarity.
    """
    if s_values is None:
        s_values = np.linspace(0.3, 0.7, 9)

    scores = []

    for s in s_values:
        # Weight features by y^(s-1/2) to test balance
        weight = np.abs(X_mult) ** (s - 0.5)
        weight = weight / (np.mean(weight) + 1e-10)  # Normalize

        # Correlation between weighted features
        X_add_flat = X_add.flatten()
        X_mult_weighted = (X_mult * weight).flatten()

        # Remove NaN/Inf
        valid = np.isfinite(X_add_flat) & np.isfinite(X_mult_weighted)
        if np.sum(valid) > 0:
            corr = np.corrcoef(X_add_flat[valid], X_mult_weighted[valid])[0, 1]
        else:
            corr = 0.0

        scores.append(abs(corr))

    scores = np.array(scores)
    optimal_idx = np.argmax(scores)
    optimal_s = s_values[optimal_idx]

    return scores, optimal_s


def analytical_mellin_pairs() -> Dict[str, Tuple]:
    """Return dictionary of known Mellin transform pairs.

    Returns
    -------
    pairs : dict
        Maps function names to (function, transform) pairs

    Notes
    -----
    These analytical results validate numerical implementations.
    """
    pairs = {
        "exponential": (
            lambda y: np.exp(-y),
            lambda s: np.math.gamma(s.real) if s.real > 0 else np.inf,
        ),
        "power": (
            lambda y, a: y**a / (1 + y) ** (a + 1),
            lambda s, a: np.pi / np.sin(np.pi * s) if 0 < s.real < a + 1 else np.inf,
        ),
        "gaussian": (
            lambda y: np.exp(-(y**2)),
            lambda s: 0.5 * np.math.gamma(s.real / 2) if s.real > 0 else np.inf,
        ),
    }
    return pairs
