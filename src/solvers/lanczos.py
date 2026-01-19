"""Lanczos algorithm for eigenvalue computation."""

import time
from typing import Literal, Optional, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

from .utils import LanczosInfo


def topk_eigs(
    L: csr_matrix,
    k: int = 16,
    which: Literal["SM", "LM", "SA", "LA"] = "SM",
    maxiter: int = 5000,
    tol: float = 1e-6,
    seed: Optional[int] = 0,
) -> Tuple[np.ndarray, np.ndarray, LanczosInfo]:
    """Compute top-k eigenvalues and eigenvectors using Lanczos.

    Parameters
    ----------
    L : scipy.sparse.csr_matrix, shape (n, n)
        Sparse symmetric matrix (typically normalized Laplacian)
    k : int
        Number of eigenvalues/vectors to compute
    which : {"SM", "LM", "SA", "LA"}
        Which eigenvalues to find:
        - "SM": smallest magnitude
        - "LM": largest magnitude
        - "SA": smallest algebraic (most negative)
        - "LA": largest algebraic (most positive)
    maxiter : int
        Maximum Lanczos iterations
    tol : float
        Tolerance for eigenvalue convergence
    seed : int or None
        Random seed for initial vector

    Returns
    -------
    evals : np.ndarray, shape (k,)
        Eigenvalues in ascending order
    evecs : np.ndarray, shape (n, k)
        Corresponding eigenvectors
    info : LanczosInfo
        Convergence information

    Notes
    -----
    This is a wrapper around scipy.sparse.linalg.eigsh with
    additional logging and convergence tracking.
    For graph Laplacians, typically use which="SM" to get
    the low-frequency eigenmodes.
    """
    if seed is not None:
        np.random.seed(seed)

    start_time = time.time()
    n = L.shape[0]

    # Ensure k is valid
    k = min(k, n - 1)

    # Initial vector for Lanczos
    v0 = np.random.randn(n)
    v0 = v0 / np.linalg.norm(v0)

    try:
        # Call scipy's eigsh (uses ARPACK's Lanczos)
        evals, evecs = eigsh(
            L,
            k=k,
            which=which,
            tol=tol,
            maxiter=maxiter,
            v0=v0,
            return_eigenvectors=True,
        )

        # Sort by eigenvalue (ascending)
        idx = np.argsort(evals)
        evals = evals[idx]
        evecs = evecs[:, idx]

        # Compute Ritz residuals for convergence check
        ritz_residuals = np.zeros(k)
        for i in range(k):
            # Residual: ||L v_i - λ_i v_i||
            residual = L @ evecs[:, i] - evals[i] * evecs[:, i]
            ritz_residuals[i] = np.linalg.norm(residual)

        converged = np.all(ritz_residuals < tol)
        n_converged = np.sum(ritz_residuals < tol)

    except Exception as e:
        # Handle convergence failures gracefully
        print(f"Warning: eigsh did not fully converge: {e}")
        evals = np.array([])
        evecs = np.array([]).reshape(n, 0)
        ritz_residuals = None
        converged = False
        n_converged = 0

    # Build info
    info = LanczosInfo(
        converged=converged,
        iterations=min(maxiter, k * 10),  # Estimate
        n_converged=n_converged,
        ritz_residuals=ritz_residuals,
        matvecs=k * 10,  # Rough estimate
        wall_time=time.time() - start_time,
        tol=tol,
        maxiter=maxiter,
    )

    return evals, evecs, info


def spectral_gap(evals: np.ndarray) -> float:
    """Compute spectral gap λ_2 - λ_1.

    Parameters
    ----------
    evals : np.ndarray
        Eigenvalues in ascending order

    Returns
    -------
    gap : float
        Spectral gap (0 if less than 2 eigenvalues)
    """
    if len(evals) < 2:
        return 0.0
    return evals[1] - evals[0]


def fiedler_vector(L: csr_matrix, **kwargs) -> Tuple[np.ndarray, float]:
    """Compute Fiedler vector (second smallest eigenvector).

    Parameters
    ----------
    L : scipy.sparse.csr_matrix
        Graph Laplacian
    **kwargs
        Arguments passed to topk_eigs

    Returns
    -------
    fiedler : np.ndarray, shape (n,)
        Fiedler vector (second eigenvector)
    fiedler_value : float
        Fiedler value (second eigenvalue)

    Notes
    -----
    The Fiedler vector provides a spectral embedding that
    often reveals graph structure and clusters.
    """
    evals, evecs, _ = topk_eigs(L, k=2, which="SM", **kwargs)

    if len(evals) < 2:
        raise ValueError("Graph has fewer than 2 eigenvalues")

    return evecs[:, 1], evals[1]
