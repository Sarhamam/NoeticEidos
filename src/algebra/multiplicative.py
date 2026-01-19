"""Multiplicative transport operations (Poisson kernel via log/Haar)."""

from typing import Union

import numpy as np


def log_map(X: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Apply elementwise logarithm with regularization.

    Parameters
    ----------
    X : np.ndarray
        Input data
    eps : float
        Regularization to avoid log(0)

    Returns
    -------
    Z : np.ndarray
        Log-transformed data: log(|X| + eps)

    Notes
    -----
    This is the fundamental map taking multiplicative structure
    on ℝ₊ to additive structure on ℝ.
    """
    return np.log(np.abs(X) + eps)


def poisson_kernel_log(
    delta: Union[float, np.ndarray], t: float
) -> Union[float, np.ndarray]:
    """Poisson kernel in log-domain for multiplicative transport.

    Parameters
    ----------
    delta : float or np.ndarray
        Log-domain distance: log(x) - log(y)
    t : float
        Scale parameter (> 0)

    Returns
    -------
    P_t : float or np.ndarray
        Poisson kernel value: t / (π(δ² + t²))

    Notes
    -----
    This kernel respects the Haar measure dy/y on ℝ₊ and
    provides the multiplicative analogue of the Gaussian.
    The formula is the Cauchy distribution in log-space.
    """
    if t <= 0:
        raise ValueError("Scale t must be positive")

    return t / (np.pi * (delta**2 + t**2))


def haar_measure_weight(y: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Compute Haar measure weights dy/y for integration.

    Parameters
    ----------
    y : np.ndarray
        Points on ℝ₊
    eps : float
        Regularization for near-zero values

    Returns
    -------
    weights : np.ndarray
        Haar measure weights 1/|y|

    Notes
    -----
    The Haar measure dy/y is the unique (up to scale)
    translation-invariant measure on the multiplicative group ℝ₊.
    """
    return 1.0 / (np.abs(y) + eps)


def multiplicative_distance(
    X: np.ndarray, Y: np.ndarray, eps: float = 1e-6
) -> np.ndarray:
    """Compute distance in multiplicative (log) space.

    Parameters
    ----------
    X : np.ndarray, shape (n, d)
        First dataset
    Y : np.ndarray, shape (m, d)
        Second dataset (or same as X for pairwise)
    eps : float
        Regularization for log transform

    Returns
    -------
    D : np.ndarray, shape (n, m)
        Distance matrix in log-space

    Notes
    -----
    Distance is computed as ||log(|X|+eps) - log(|Y|+eps)||
    which respects the multiplicative group structure.
    """
    # Transform to log space
    log_X = log_map(X, eps)
    log_Y = log_map(Y, eps)

    # Compute Euclidean distances in log space
    # ||log(x) - log(y)||² = ||log(x)||² + ||log(y)||² - 2⟨log(x), log(y)⟩
    X_norm2 = np.sum(log_X**2, axis=1, keepdims=True)
    Y_norm2 = np.sum(log_Y**2, axis=1, keepdims=True)
    dist2 = X_norm2 + Y_norm2.T - 2 * log_X @ log_Y.T

    # Ensure non-negative and take sqrt
    dist2 = np.maximum(dist2, 0)
    return np.sqrt(dist2)


def poisson_affinity_matrix(X: np.ndarray, tau: float, eps: float = 1e-6) -> np.ndarray:
    """Build Poisson affinity matrix for multiplicative transport.

    Parameters
    ----------
    X : np.ndarray, shape (n, d)
        Data points on ℝ₊ᵈ
    tau : float
        Scale parameter for Poisson kernel
    eps : float
        Regularization for log transform

    Returns
    -------
    W : np.ndarray, shape (n, n)
        Dense affinity matrix with Poisson kernel weights

    Notes
    -----
    Creates DENSE matrix. For large n, use sparse approximations.
    The kernel is applied to pairwise log-distances.
    """
    n = X.shape[0]

    # Transform to log space
    log_X = log_map(X, eps)

    # Compute pairwise differences in log space
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                # Log-distance for each dimension
                delta = log_X[i] - log_X[j]
                # Average Poisson kernel across dimensions
                W[i, j] = np.mean([poisson_kernel_log(d, tau) for d in delta])

    return W


def multiplicative_heat_kernel(
    X: np.ndarray, t: float, eps: float = 1e-6
) -> np.ndarray:
    """Heat kernel for multiplicative transport via log-space diffusion.

    Parameters
    ----------
    X : np.ndarray, shape (n, d)
        Data points
    t : float
        Diffusion time
    eps : float
        Regularization

    Returns
    -------
    H_t : np.ndarray, shape (n, n)
        Heat kernel matrix

    Notes
    -----
    This performs diffusion in log-space, which corresponds to
    multiplicative diffusion in the original space with Haar measure.
    """
    # Transform to log space
    log_X = log_map(X, eps)

    # Compute Gaussian heat kernel in log space
    n = X.shape[0]
    H_t = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            # Squared distance in log space
            dist2 = np.sum((log_X[i] - log_X[j]) ** 2)
            # Gaussian kernel with variance 2t
            H_t[i, j] = np.exp(-dist2 / (4 * t)) / (4 * np.pi * t) ** (X.shape[1] / 2)

    # Normalize diagonal
    np.fill_diagonal(H_t, 0)
    row_sums = H_t.sum(axis=1)
    H_t = H_t + np.diag(1 - row_sums)

    return H_t


def log_ratio_distance(x: np.ndarray, y: np.ndarray, eps: float = 1e-10) -> float:
    """Compute log-ratio distance between positive vectors.

    Parameters
    ----------
    x : np.ndarray
        First positive vector
    y : np.ndarray
        Second positive vector
    eps : float
        Regularization

    Returns
    -------
    d : float
        Log-ratio distance: sqrt(Σ(log(x_i/y_i))²)

    Notes
    -----
    This distance is invariant under global scaling:
    d(cx, cy) = d(x, y) for any c > 0.
    """
    ratio = (np.abs(x) + eps) / (np.abs(y) + eps)
    log_ratio = np.log(ratio)
    return np.linalg.norm(log_ratio)
