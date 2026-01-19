"""Diffusion processes on graphs via matrix exponentials."""

import numpy as np
from scipy.sparse import csr_matrix, eye, issparse
from scipy.sparse.linalg import expm_multiply, LinearOperator
from typing import Union, Literal, Optional
import time

from solvers.cg import cg_solve
from solvers.lanczos import topk_eigs


def simulate_diffusion(
    L: Union[np.ndarray, csr_matrix],
    u0: np.ndarray,
    t: float,
    method: Literal["krylov", "cg_step", "eigendecomp"] = "krylov",
    alpha: float = 1e-3,
    tol: float = 1e-6,
    maxiter: int = 200,
    k_eigs: Optional[int] = None
) -> np.ndarray:
    """Simulate diffusion u(t) = exp(-tL) u0.

    Parameters
    ----------
    L : sparse matrix, shape (n, n)
        Graph Laplacian (should be PSD)
    u0 : np.ndarray, shape (n,)
        Initial state
    t : float
        Diffusion time
    method : {"krylov", "cg_step", "eigendecomp"}
        Method for matrix exponential:
        - "krylov": Krylov subspace approximation (recommended)
        - "cg_step": Backward Euler steps
        - "eigendecomp": Full eigendecomposition (small graphs only)
    alpha : float
        Regularization for CG method
    tol : float
        Tolerance for iterative methods
    maxiter : int
        Maximum iterations
    k_eigs : int or None
        Number of eigenvalues for partial eigendecomposition

    Returns
    -------
    u_t : np.ndarray, shape (n,)
        Solution at time t

    Notes
    -----
    Solves the heat equation du/dt = -L u with u(0) = u0.
    Mass is conserved: sum(u_t) = sum(u0) for connected graphs.
    """
    if t < 0:
        raise ValueError("Time t must be non-negative")

    if t == 0:
        return u0.copy()

    n = L.shape[0]

    if method == "krylov":
        # Use scipy's Krylov-based matrix exponential
        u_t = expm_multiply(-t * L, u0, start=0, stop=t, num=2)[-1]

    elif method == "cg_step":
        # Backward Euler approximation: (I + dt*L) u_{k+1} = u_k
        # with multiple steps to approximate exp(-tL)
        num_steps = max(1, int(np.ceil(t / 0.1)))  # Adaptive step size
        dt = t / num_steps

        u_current = u0.copy()
        I = eye(n, format='csr')

        for _ in range(num_steps):
            # Solve (I + dt*L + alpha*I) u = u_current
            A = I + dt * L + alpha * I
            u_current, info = cg_solve(A, u_current, alpha=0, rtol=tol, maxiter=maxiter)

            if not info.converged:
                print(f"Warning: CG did not converge in step, residual={info.residual_norm:.2e}")

        u_t = u_current

    elif method == "eigendecomp":
        # Full or partial eigendecomposition
        if k_eigs is None or k_eigs >= n - 1:
            # Full eigendecomposition
            if issparse(L):
                L_dense = L.toarray()
            else:
                L_dense = L
            evals, evecs = np.linalg.eigh(L_dense)
        else:
            # Partial eigendecomposition
            evals, evecs = topk_eigs(L, k=k_eigs, which="SM")

        # u(t) = V exp(-t Lambda) V^T u0
        coeffs = evecs.T @ u0
        u_t = evecs @ (np.exp(-t * evals) * coeffs)

    else:
        raise ValueError(f"Unknown method: {method}")

    return u_t


def simulate_poisson(
    L: Union[np.ndarray, csr_matrix],
    u0: np.ndarray,
    t: float,
    method: Literal["eigendecomp", "rational"] = "eigendecomp",
    tol: float = 1e-6,
    k_eigs: Optional[int] = None
) -> np.ndarray:
    """Simulate Poisson process u(t) = exp(-t sqrt(L)) u0.

    Parameters
    ----------
    L : sparse matrix, shape (n, n)
        Graph Laplacian
    u0 : np.ndarray, shape (n,)
        Initial state
    t : float
        Process time
    method : {"eigendecomp", "rational"}
        Method for fractional matrix exponential
    tol : float
        Tolerance for approximations
    k_eigs : int or None
        Number of eigenvalues for partial method

    Returns
    -------
    u_t : np.ndarray, shape (n,)
        Solution at time t

    Notes
    -----
    Implements the Poisson semigroup exp(-t sqrt(L)).
    This corresponds to subordinated Brownian motion.
    """
    if t < 0:
        raise ValueError("Time t must be non-negative")

    if t == 0:
        return u0.copy()

    n = L.shape[0]

    if method == "eigendecomp":
        # Eigendecomposition approach
        if k_eigs is None or k_eigs >= n - 1:
            # Full eigendecomposition
            if issparse(L):
                L_dense = L.toarray()
            else:
                L_dense = L
            evals, evecs = np.linalg.eigh(L_dense)
        else:
            # Partial eigendecomposition
            evals, evecs = topk_eigs(L, k=k_eigs, which="SM")

        # u(t) = V exp(-t sqrt(Lambda)) V^T u0
        coeffs = evecs.T @ u0
        sqrt_evals = np.sqrt(np.maximum(evals, 0))  # Ensure non-negative
        u_t = evecs @ (np.exp(-t * sqrt_evals) * coeffs)

    elif method == "rational":
        # Rational approximation (placeholder - could implement Padé)
        # For now, fall back to eigendecomposition
        return simulate_poisson(L, u0, t, method="eigendecomp", k_eigs=k_eigs)

    else:
        raise ValueError(f"Unknown method: {method}")

    return u_t


def diffusion_distance(L: Union[np.ndarray, csr_matrix], t: float,
                      method: str = "krylov") -> np.ndarray:
    """Compute diffusion distance matrix at time t.

    Parameters
    ----------
    L : sparse matrix, shape (n, n)
        Graph Laplacian
    t : float
        Diffusion time
    method : str
        Method for matrix exponential

    Returns
    -------
    D : np.ndarray, shape (n, n)
        Diffusion distance matrix

    Notes
    -----
    Diffusion distance: D_t(i,j)² = ||p_t(i,·) - p_t(j,·)||²
    where p_t(i,·) is the i-th row of exp(-tL).
    """
    n = L.shape[0]
    I = eye(n, format='csr')

    # Compute heat kernel matrix
    if method == "krylov":
        # Compute each column of exp(-tL)
        H_t = np.zeros((n, n))
        for i in range(n):
            e_i = np.zeros(n)
            e_i[i] = 1.0
            H_t[:, i] = simulate_diffusion(L, e_i, t, method="krylov")
    else:
        # Use eigendecomposition
        H_t = np.zeros((n, n))
        for i in range(n):
            e_i = np.zeros(n)
            e_i[i] = 1.0
            H_t[:, i] = simulate_diffusion(L, e_i, t, method=method)

    # Compute diffusion distances
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            diff = H_t[i, :] - H_t[j, :]
            D[i, j] = D[j, i] = np.linalg.norm(diff)

    return D


def heat_kernel_signature(L: Union[np.ndarray, csr_matrix],
                         times: np.ndarray,
                         method: str = "eigendecomp") -> np.ndarray:
    """Compute Heat Kernel Signature (HKS) for each node.

    Parameters
    ----------
    L : sparse matrix, shape (n, n)
        Graph Laplacian
    times : np.ndarray, shape (T,)
        Time scales for HKS
    method : str
        Method for matrix exponential

    Returns
    -------
    HKS : np.ndarray, shape (n, T)
        Heat kernel signature matrix

    Notes
    -----
    HKS(i,t) = [exp(-tL)]_{ii} is the diagonal of the heat kernel.
    Provides multiscale geometric information about each node.
    """
    n = L.shape[0]
    T = len(times)
    HKS = np.zeros((n, T))

    for t_idx, t in enumerate(times):
        # Get diagonal of exp(-tL)
        for i in range(n):
            e_i = np.zeros(n)
            e_i[i] = 1.0
            u_t = simulate_diffusion(L, e_i, t, method=method)
            HKS[i, t_idx] = u_t[i]

    return HKS


def multiscale_diffusion(L: Union[np.ndarray, csr_matrix],
                        u0: np.ndarray,
                        times: np.ndarray,
                        method: str = "krylov") -> np.ndarray:
    """Simulate diffusion at multiple time scales.

    Parameters
    ----------
    L : sparse matrix, shape (n, n)
        Graph Laplacian
    u0 : np.ndarray, shape (n,)
        Initial state
    times : np.ndarray, shape (T,)
        Time points
    method : str
        Method for simulation

    Returns
    -------
    U : np.ndarray, shape (n, T)
        Solutions at all time points

    Notes
    -----
    More efficient than calling simulate_diffusion repeatedly
    when eigendecomposition is available.
    """
    n = L.shape[0]
    T = len(times)
    U = np.zeros((n, T))

    if method == "eigendecomp":
        # Compute once, evaluate at all times
        if issparse(L):
            L_dense = L.toarray()
        else:
            L_dense = L
        evals, evecs = np.linalg.eigh(L_dense)

        coeffs = evecs.T @ u0
        for t_idx, t in enumerate(times):
            U[:, t_idx] = evecs @ (np.exp(-t * evals) * coeffs)
    else:
        # Compute each time separately
        for t_idx, t in enumerate(times):
            U[:, t_idx] = simulate_diffusion(L, u0, t, method=method)

    return U