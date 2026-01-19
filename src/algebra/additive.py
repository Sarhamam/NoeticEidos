"""Additive transport operations (Gaussian kernel)."""

from typing import Union

import numpy as np


def gaussian_kernel(
    dist2: Union[float, np.ndarray], sigma: float
) -> Union[float, np.ndarray]:
    """Compute Gaussian kernel exp(-d²/σ²).

    Parameters
    ----------
    dist2 : float or np.ndarray
        Squared distances
    sigma : float
        Bandwidth parameter (scale)

    Returns
    -------
    weights : float or np.ndarray
        Gaussian kernel values

    Notes
    -----
    The Gaussian kernel satisfies the semigroup property:
    G_σ₁ * G_σ₂ = G_{√(σ₁²+σ₂²)}
    where * denotes convolution.
    """
    if sigma <= 0:
        raise ValueError("Sigma must be positive")

    return np.exp(-dist2 / (sigma**2))


def gaussian_affinity_matrix(X: np.ndarray, sigma: float) -> np.ndarray:
    """Build Gaussian affinity matrix for dataset.

    Parameters
    ----------
    X : np.ndarray, shape (n, d)
        Data points
    sigma : float
        Gaussian bandwidth

    Returns
    -------
    W : np.ndarray, shape (n, n)
        Dense affinity matrix W_ij = exp(-||x_i - x_j||² / σ²)

    Notes
    -----
    This creates a DENSE matrix. For large n, use sparse k-NN
    approximation via graphs.knn.build_graph instead.
    """
    X.shape[0]

    # Compute pairwise squared distances
    # ||x_i - x_j||² = ||x_i||² + ||x_j||² - 2⟨x_i, x_j⟩
    X_norm2 = np.sum(X**2, axis=1, keepdims=True)
    dist2 = X_norm2 + X_norm2.T - 2 * X @ X.T

    # Ensure non-negative (numerical issues can make it slightly negative)
    dist2 = np.maximum(dist2, 0)

    # Apply Gaussian kernel
    W = gaussian_kernel(dist2, sigma)

    # Zero diagonal (no self-loops)
    np.fill_diagonal(W, 0)

    return W


def heat_kernel(L: np.ndarray, t: float, k: int = 50) -> np.ndarray:
    """Compute heat kernel exp(-tL) via eigendecomposition.

    Parameters
    ----------
    L : np.ndarray, shape (n, n)
        Graph Laplacian (should be PSD)
    t : float
        Diffusion time
    k : int
        Number of eigenvectors to use (truncation)

    Returns
    -------
    H_t : np.ndarray, shape (n, n)
        Heat kernel matrix

    Notes
    -----
    The heat kernel solves ∂u/∂t = -Lu with u(0) = δ.
    For large graphs, use sparse approximations.
    """
    if t < 0:
        raise ValueError("Time t must be non-negative")

    n = L.shape[0]

    if k >= n:
        # Full eigendecomposition
        evals, evecs = np.linalg.eigh(L)
    else:
        # Use only k smallest eigenvalues
        from scipy.sparse.linalg import eigsh

        evals, evecs = eigsh(L, k=k, which="SM")

    # Heat kernel: H_t = V exp(-t Λ) V^T
    H_t = evecs @ np.diag(np.exp(-t * evals)) @ evecs.T

    return H_t


def diffusion_distance(H_t: np.ndarray) -> np.ndarray:
    """Compute diffusion distances from heat kernel.

    Parameters
    ----------
    H_t : np.ndarray, shape (n, n)
        Heat kernel at time t

    Returns
    -------
    D : np.ndarray, shape (n, n)
        Diffusion distance matrix

    Notes
    -----
    Diffusion distance: D²_t(i,j) = ||p_t(i,·) - p_t(j,·)||²
    where p_t is the transition probability.
    """
    H_t.shape[0]

    # Normalize rows to get transition probabilities
    row_sums = H_t.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)  # Avoid division by zero
    P = H_t / row_sums

    # Diffusion distance squared
    # D²_ij = Σ_k (P_ik - P_jk)² = P_ii + P_jj - 2P_ij (for normalized P)
    P_diag = np.diag(P).reshape(-1, 1)
    D2 = P_diag + P_diag.T - 2 * P

    # Ensure non-negative and take sqrt
    D2 = np.maximum(D2, 0)
    D = np.sqrt(D2)

    return D
