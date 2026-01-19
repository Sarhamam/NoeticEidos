"""k-NN graph construction with dual transport modes."""

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.neighbors import NearestNeighbors
from typing import Union, Literal, Optional


def build_graph(
    X: np.ndarray,
    mode: Literal["additive", "multiplicative"] = "additive",
    k: int = 16,
    sigma: Union[str, float] = "median",
    tau: Union[str, float] = "median",
    eps: float = 1e-6,
    seed: Optional[int] = 0
) -> csr_matrix:
    """Build k-NN graph with additive or multiplicative transport.

    Parameters
    ----------
    X : np.ndarray, shape (n, d)
        Input dataset
    mode : {"additive", "multiplicative"}
        Transport mode:
        - additive: standard Euclidean distances with Gaussian weights
        - multiplicative: log-transformed distances with Haar measure
    k : int
        Number of nearest neighbors
    sigma : float or "median"
        Scale for Gaussian kernel (additive mode).
        If "median", uses median neighbor distance.
    tau : float or "median"
        Scale for multiplicative kernel.
        If "median", uses median neighbor distance.
    eps : float
        Regularization for log transform in multiplicative mode
    seed : int or None
        Random seed for reproducibility

    Returns
    -------
    A : scipy.sparse.csr_matrix, shape (n, n)
        Sparse symmetric adjacency matrix with Gaussian weights

    Notes
    -----
    Multiplicative mode applies Z = log(|X| + eps) before k-NN search,
    respecting the Haar measure dy/y on ℝ₊.
    """
    if seed is not None:
        np.random.seed(seed)

    n = X.shape[0]

    # Transform data for multiplicative mode
    if mode == "multiplicative":
        # Apply log transform with regularization
        Z = np.log(np.abs(X) + eps)
    else:
        Z = X.copy()

    # Build k-NN graph
    nbrs = NearestNeighbors(n_neighbors=min(k+1, n), algorithm='auto')
    nbrs.fit(Z)
    distances, indices = nbrs.kneighbors(Z)

    # Remove self-connections (first neighbor is always self)
    distances = distances[:, 1:]
    indices = indices[:, 1:]

    # Determine bandwidth parameter
    if mode == "additive":
        if sigma == "median":
            # Use median of all neighbor distances
            sigma_val = np.median(distances.flatten())
            if sigma_val == 0:
                sigma_val = 1.0  # Fallback for degenerate cases
        else:
            sigma_val = sigma
        bandwidth = sigma_val
    else:  # multiplicative
        if tau == "median":
            tau_val = np.median(distances.flatten())
            if tau_val == 0:
                tau_val = 1.0
        else:
            tau_val = tau
        bandwidth = tau_val

    # Compute Gaussian weights
    weights = np.exp(-(distances ** 2) / (bandwidth ** 2))

    # Build sparse adjacency matrix
    A = lil_matrix((n, n))

    for i in range(n):
        for j, w in zip(indices[i], weights[i]):
            if j < n:  # Safety check
                A[i, j] = w
                A[j, i] = w  # Ensure symmetry

    # Convert to CSR for efficient operations
    A = A.tocsr()

    # Remove self-loops
    A.setdiag(0)

    return A