"""Preconditioners for iterative solvers."""

import numpy as np
from scipy.sparse import diags, csr_matrix, issparse
from typing import Union, Callable, Optional


def jacobi_preconditioner(A: csr_matrix, eps: float = 1e-10) -> Callable:
    """Build Jacobi (diagonal) preconditioner for matrix A.

    Parameters
    ----------
    A : scipy.sparse.csr_matrix
        Matrix to precondition
    eps : float
        Regularization to prevent division by zero

    Returns
    -------
    M : callable
        Preconditioner function M(v) that applies M^{-1} to vector v
    """
    # Extract diagonal
    diag_A = A.diagonal()

    # Safe-guard against zero diagonal entries
    diag_inv = np.where(np.abs(diag_A) > eps, 1.0 / diag_A, 1.0)

    def M_apply(v):
        """Apply M^{-1} = D^{-1} to vector v."""
        return diag_inv * v

    return M_apply


def identity_preconditioner() -> Callable:
    """No preconditioning (identity)."""
    return lambda v: v


def build_preconditioner(
    A: csr_matrix,
    method: str = "jacobi",
    **kwargs
) -> Callable:
    """Build a preconditioner for matrix A.

    Parameters
    ----------
    A : scipy.sparse.csr_matrix
        Matrix to precondition
    method : str
        Preconditioner type:
        - "jacobi": Diagonal (Jacobi) preconditioner
        - "none": No preconditioning (identity)
        - "ilu": Incomplete LU (placeholder for future)
    **kwargs
        Additional parameters for specific preconditioners

    Returns
    -------
    M : callable
        Preconditioner function M(v) = M^{-1} v

    Notes
    -----
    For CG, the preconditioner M should be symmetric positive definite.
    Jacobi is always safe for SPD matrices.
    """
    if method == "jacobi":
        return jacobi_preconditioner(A, eps=kwargs.get('eps', 1e-10))
    elif method == "none" or method is None:
        return identity_preconditioner()
    elif method == "ilu":
        # Placeholder for ILU implementation
        raise NotImplementedError("ILU preconditioner not yet implemented. Use 'jacobi' for now.")
    else:
        raise ValueError(f"Unknown preconditioner method: {method}")


class DiagonalPreconditioner:
    """Diagonal preconditioner as a class (alternative interface)."""

    def __init__(self, diag_values: np.ndarray, eps: float = 1e-10):
        """Initialize with diagonal values.

        Parameters
        ----------
        diag_values : np.ndarray
            Diagonal entries of the matrix to precondition
        eps : float
            Regularization parameter
        """
        self.diag_inv = np.where(np.abs(diag_values) > eps,
                                 1.0 / diag_values, 1.0)

    def __call__(self, v: np.ndarray) -> np.ndarray:
        """Apply preconditioner M^{-1} to vector v."""
        return self.diag_inv * v

    def matvec(self, v: np.ndarray) -> np.ndarray:
        """Alternative name for compatibility with scipy."""
        return self.__call__(v)