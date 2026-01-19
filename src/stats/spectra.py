"""Spectral measures for both additive and multiplicative transport modes."""

import numpy as np
from scipy.sparse import csr_matrix, issparse
from typing import Union, Optional
import warnings

from graphs.knn import build_graph
from graphs.laplacian import laplacian
from solvers.lanczos import topk_eigs


def spectral_gap(L: Union[np.ndarray, csr_matrix], k: int = 2, tol: float = 1e-9) -> float:
    """Compute smallest nonzero eigenvalue (spectral gap) of Laplacian.

    Parameters
    ----------
    L : sparse matrix or ndarray, shape (n, n)
        Graph Laplacian matrix
    k : int
        Number of smallest eigenvalues to compute
    tol : float
        Tolerance for zero eigenvalues

    Returns
    -------
    gap : float
        Smallest nonzero eigenvalue

    Notes
    -----
    For connected graphs, the smallest eigenvalue should be ≈ 0.
    The spectral gap is the second smallest eigenvalue λ₁.
    """
    if issparse(L):
        # Use Lanczos for sparse matrices
        try:
            evals, _, _ = topk_eigs(L, k=max(k, 3), which="SM")
        except:
            # Fallback to dense computation for small matrices
            evals, _ = np.linalg.eigh(L.toarray())
    else:
        evals, _ = np.linalg.eigh(L)

    # Sort eigenvalues
    evals = np.sort(np.real(evals))

    # Find first nonzero eigenvalue
    nonzero_evals = evals[evals > tol]

    if len(nonzero_evals) == 0:
        warnings.warn("No nonzero eigenvalues found - disconnected graph?")
        return 0.0

    return float(nonzero_evals[0])


def spectral_entropy(L: Union[np.ndarray, csr_matrix], k: int = 10, tol: float = 1e-9) -> float:
    """Compute Shannon entropy of first k nonzero eigenvalues.

    Parameters
    ----------
    L : sparse matrix or ndarray, shape (n, n)
        Graph Laplacian matrix
    k : int
        Number of eigenvalues to include in entropy calculation
    tol : float
        Tolerance for zero eigenvalues

    Returns
    -------
    entropy : float
        Shannon entropy: H = -∑ pᵢ log pᵢ where pᵢ = λᵢ / ∑λⱼ

    Notes
    -----
    Provides a measure of eigenvalue distribution:
    - Low entropy: few dominant eigenvalues (regular structure)
    - High entropy: uniform eigenvalue distribution (irregular structure)
    """
    if issparse(L):
        try:
            evals, _, _ = topk_eigs(L, k=k+2, which="SM")
        except:
            evals, _ = np.linalg.eigh(L.toarray())
    else:
        evals, _ = np.linalg.eigh(L)

    # Get k largest nonzero eigenvalues
    evals = np.sort(np.real(evals))
    nonzero_evals = evals[evals > tol]

    if len(nonzero_evals) == 0:
        return 0.0

    # Take first k nonzero eigenvalues
    k_evals = nonzero_evals[:min(k, len(nonzero_evals))]

    # Normalize to probabilities
    total = np.sum(k_evals)
    if total <= tol:
        return 0.0

    probs = k_evals / total

    # Compute Shannon entropy
    # Use log(p + eps) to handle numerical issues
    eps = 1e-16
    entropy = -np.sum(probs * np.log(probs + eps))

    return float(entropy)


def spectral_gap_additive(X: np.ndarray, k: int = 2, sigma: Union[float, str] = "median",
                         neighbors: int = 16, seed: Optional[int] = None) -> float:
    """Compute spectral gap using additive (Gaussian) transport.

    Parameters
    ----------
    X : ndarray, shape (n, d)
        Data points
    k : int
        Number of eigenvalues to compute
    sigma : float or "median"
        Bandwidth for Gaussian kernel
    neighbors : int
        Number of k-NN neighbors
    seed : int or None
        Random seed for reproducibility

    Returns
    -------
    gap : float
        Spectral gap of additive graph
    """
    A = build_graph(X, mode="additive", k=neighbors, sigma=sigma, seed=seed)
    L = laplacian(A, normalized=True)
    return spectral_gap(L, k=k)


def spectral_entropy_additive(X: np.ndarray, k: int = 10, sigma: Union[float, str] = "median",
                             neighbors: int = 16, seed: Optional[int] = None) -> float:
    """Compute spectral entropy using additive (Gaussian) transport.

    Parameters
    ----------
    X : ndarray, shape (n, d)
        Data points
    k : int
        Number of eigenvalues for entropy
    sigma : float or "median"
        Bandwidth for Gaussian kernel
    neighbors : int
        Number of k-NN neighbors
    seed : int or None
        Random seed for reproducibility

    Returns
    -------
    entropy : float
        Spectral entropy of additive graph
    """
    A = build_graph(X, mode="additive", k=neighbors, sigma=sigma, seed=seed)
    L = laplacian(A, normalized=True)
    return spectral_entropy(L, k=k)


def spectral_gap_multiplicative(X: np.ndarray, k: int = 2, tau: float = 1.0,
                               eps: float = 1e-6, neighbors: int = 16,
                               seed: Optional[int] = None) -> float:
    """Compute spectral gap using multiplicative (Poisson) transport.

    Parameters
    ----------
    X : ndarray, shape (n, d)
        Data points
    k : int
        Number of eigenvalues to compute
    tau : float
        Bandwidth for Poisson kernel
    eps : float
        Regularization for log-map
    neighbors : int
        Number of k-NN neighbors
    seed : int or None
        Random seed for reproducibility

    Returns
    -------
    gap : float
        Spectral gap of multiplicative graph
    """
    A = build_graph(X, mode="multiplicative", k=neighbors, tau=tau, eps=eps, seed=seed)
    L = laplacian(A, normalized=True)
    return spectral_gap(L, k=k)


def spectral_entropy_multiplicative(X: np.ndarray, k: int = 10, tau: float = 1.0,
                                   eps: float = 1e-6, neighbors: int = 16,
                                   seed: Optional[int] = None) -> float:
    """Compute spectral entropy using multiplicative (Poisson) transport.

    Parameters
    ----------
    X : ndarray, shape (n, d)
        Data points
    k : int
        Number of eigenvalues for entropy
    tau : float
        Bandwidth for Poisson kernel
    eps : float
        Regularization for log-map
    neighbors : int
        Number of k-NN neighbors
    seed : int or None
        Random seed for reproducibility

    Returns
    -------
    entropy : float
        Spectral entropy of multiplicative graph
    """
    A = build_graph(X, mode="multiplicative", k=neighbors, tau=tau, eps=eps, seed=seed)
    L = laplacian(A, normalized=True)
    return spectral_entropy(L, k=k)


def eigenvalue_distribution(L: Union[np.ndarray, csr_matrix], k: int = 20) -> np.ndarray:
    """Get eigenvalue distribution for analysis.

    Parameters
    ----------
    L : sparse matrix or ndarray, shape (n, n)
        Graph Laplacian matrix
    k : int
        Number of smallest eigenvalues to return

    Returns
    -------
    evals : ndarray, shape (k,)
        k smallest eigenvalues in ascending order
    """
    if issparse(L):
        try:
            evals, _ = topk_eigs(L, k=k, which="SM")
        except:
            evals, _ = np.linalg.eigh(L.toarray())
    else:
        evals, _ = np.linalg.eigh(L)

    evals = np.sort(np.real(evals))
    return evals[:k]


def effective_resistance_sum(L: Union[np.ndarray, csr_matrix],
                           sample_size: int = 100, seed: Optional[int] = None) -> float:
    """Estimate sum of effective resistances via sampling.

    Parameters
    ----------
    L : sparse matrix or ndarray, shape (n, n)
        Graph Laplacian matrix
    sample_size : int
        Number of node pairs to sample
    seed : int or None
        Random seed for reproducibility

    Returns
    -------
    resistance_sum : float
        Estimated sum of effective resistances

    Notes
    -----
    Effective resistance between nodes i,j is (eᵢ - eⱼ)ᵀ L⁺ (eᵢ - eⱼ)
    where L⁺ is the Moore-Penrose pseudoinverse.
    """
    rng = np.random.default_rng(seed)
    n = L.shape[0]

    if n <= 1:
        return 0.0

    # Sample pairs of nodes
    num_pairs = min(sample_size, n * (n - 1) // 2)

    if issparse(L):
        try:
            # Use eigendecomposition for pseudoinverse
            evals, evecs = topk_eigs(L, k=min(n-1, 50), which="SM")
            # Remove zero eigenvalue
            nonzero_mask = evals > 1e-12
            evals_nz = evals[nonzero_mask]
            evecs_nz = evecs[:, nonzero_mask]

            # Pseudoinverse via eigendecomposition
            L_pinv_diag = 1.0 / evals_nz

            total_resistance = 0.0
            for _ in range(num_pairs):
                i, j = rng.choice(n, size=2, replace=False)

                # Effective resistance: (eᵢ - eⱼ)ᵀ L⁺ (eᵢ - eⱼ)
                diff_i = evecs_nz[i, :] - evecs_nz[j, :]
                resistance = np.sum(L_pinv_diag * diff_i**2)
                total_resistance += resistance

            return total_resistance / num_pairs * (n * (n - 1) // 2)

        except:
            # Fallback to dense computation
            L_dense = L.toarray()
    else:
        L_dense = L

    # Dense pseudoinverse
    L_pinv = np.linalg.pinv(L_dense)

    total_resistance = 0.0
    for _ in range(num_pairs):
        i, j = rng.choice(n, size=2, replace=False)
        diff = np.zeros(n)
        diff[i] = 1.0
        diff[j] = -1.0
        resistance = diff.T @ L_pinv @ diff
        total_resistance += resistance

    return total_resistance / num_pairs * (n * (n - 1) // 2)