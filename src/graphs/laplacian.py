"""Laplacian matrix computation for graph analysis."""

import numpy as np
from scipy.sparse import csr_matrix, diags, eye


def laplacian(A: csr_matrix, normalized: bool = True) -> csr_matrix:
    """Compute graph Laplacian (normalized or unnormalized).

    Parameters
    ----------
    A : scipy.sparse.csr_matrix, shape (n, n)
        Sparse symmetric adjacency matrix
    normalized : bool
        If True, returns normalized Laplacian: L = I - D^{-1/2} A D^{-1/2}
        If False, returns unnormalized Laplacian: L = D - A

    Returns
    -------
    L : scipy.sparse.csr_matrix, shape (n, n)
        Sparse Laplacian matrix

    Notes
    -----
    The normalized Laplacian has eigenvalues in [0, 2] and is better
    conditioned for spectral methods. The unnormalized version preserves
    the natural scale of the problem.

    For disconnected nodes (degree = 0), we handle gracefully:
    - Unnormalized: L[i,i] = 0 (no self-loop)
    - Normalized: L[i,i] = 1 (isolated node has full self-weight)
    """
    n = A.shape[0]

    # Compute degree vector
    degrees = np.array(A.sum(axis=1)).flatten()

    if normalized:
        # Handle zero degrees (disconnected nodes)
        with np.errstate(divide="ignore", invalid="ignore"):
            # D^{-1/2}
            d_inv_sqrt = np.where(degrees > 0, 1.0 / np.sqrt(degrees), 0)

        # Create D^{-1/2} as diagonal matrix
        D_inv_sqrt = diags(d_inv_sqrt, format="csr")

        # Normalized Laplacian: L = I - D^{-1/2} A D^{-1/2}
        L = eye(n, format="csr") - D_inv_sqrt @ A @ D_inv_sqrt

        # For isolated nodes, the diagonal should be 1
        # This is already handled by the formula above
    else:
        # Unnormalized Laplacian: L = D - A
        D = diags(degrees, format="csr")
        L = D - A

    return L.tocsr()
