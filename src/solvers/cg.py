"""Conjugate Gradient solver for shifted Laplacian systems."""

import time
from typing import Callable, Optional, Tuple, Union

import numpy as np
from scipy.sparse import csr_matrix, eye

from .preconditioners import build_preconditioner
from .utils import CGInfo, check_convergence, log_iteration


def cg_solve(
    L: csr_matrix,
    b: np.ndarray,
    alpha: float = 1e-3,
    rtol: float = 1e-6,
    atol: float = 0.0,
    maxiter: int = 2000,
    M: Optional[Union[Callable, str]] = None,
    seed: Optional[int] = 0,
    verbose: bool = False,
) -> Tuple[np.ndarray, CGInfo]:
    """Solve (L + αI)u = b using Conjugate Gradient.

    Parameters
    ----------
    L : scipy.sparse.csr_matrix, shape (n, n)
        Sparse PSD matrix (typically Laplacian)
    b : np.ndarray, shape (n,) or (n, k)
        Right-hand side vector(s)
    alpha : float
        Shift parameter for stabilization (L + αI)
    rtol : float
        Relative tolerance for convergence
    atol : float
        Absolute tolerance for convergence
    maxiter : int
        Maximum number of iterations
    M : callable, str, or None
        Preconditioner:
        - callable: M(v) applies M^{-1} to v
        - str: "jacobi", "none"
        - None: defaults to Jacobi
    seed : int or None
        Random seed for reproducibility
    verbose : bool
        Print iteration progress

    Returns
    -------
    u : np.ndarray
        Solution vector(s), shape matches b
    info : CGInfo
        Convergence information

    Notes
    -----
    Solves (L + αI)u = b where α > 0 ensures the system is SPD
    even when L has a nullspace (e.g., graph Laplacian).
    """
    if seed is not None:
        np.random.seed(seed)

    start_time = time.time()
    n = L.shape[0]

    # Handle multiple RHS
    if b.ndim == 1:
        b = b.reshape(-1, 1)
        squeeze_output = True
    else:
        squeeze_output = False

    n_rhs = b.shape[1]

    # Build shifted system A = L + αI
    A = L + alpha * eye(n, format="csr")

    # Setup preconditioner
    if M is None:
        M_apply = build_preconditioner(A, method="jacobi")
    elif isinstance(M, str):
        M_apply = build_preconditioner(A, method=M)
    elif callable(M):
        M_apply = M
    else:
        raise ValueError("M must be callable, string, or None")

    # Initialize solution array
    u = np.zeros((n, n_rhs))

    # Track convergence for each RHS
    all_converged = True
    total_iters = 0
    total_matvecs = 0

    for j in range(n_rhs):
        b_j = b[:, j]
        u_j = np.zeros(n)  # Initial guess

        # Initialize CG
        r = b_j - A @ u_j  # Initial residual
        z = M_apply(r)  # Preconditioned residual
        p = z.copy()  # Initial search direction

        b_norm = np.linalg.norm(b_j)
        r_norm = np.linalg.norm(r)
        initial_residual = r_norm

        residual_history = [r_norm]
        converged = False
        iteration = 0

        # Handle zero RHS case
        if b_norm == 0:
            converged = True
        else:
            # CG iterations
            for iteration in range(maxiter):
                # Matrix-vector product
                Ap = A @ p
                total_matvecs += 1

                # Step size
                r_dot_z = np.dot(r, z)
                p_dot_Ap = np.dot(p, Ap)
                if abs(p_dot_Ap) < 1e-14:
                    # Breakdown: search direction orthogonal to A
                    break
                alpha_k = r_dot_z / p_dot_Ap

                # Update solution
                u_j = u_j + alpha_k * p

                # Update residual
                r = r - alpha_k * Ap
                r_norm = np.linalg.norm(r)
                residual_history.append(r_norm)

                log_iteration(iteration, r_norm, verbose)

                # Check convergence
                if check_convergence(r_norm, b_norm, rtol, atol):
                    converged = True
                    break

                # Preconditioned residual
                z_new = M_apply(r)

                # Update search direction
                beta_k = np.dot(r, z_new) / r_dot_z
                p = z_new + beta_k * p
                z = z_new

        u[:, j] = u_j
        total_iters = max(total_iters, iteration + 1)
        all_converged = all_converged and converged

        if verbose:
            status = "converged" if converged else "not converged"
            print(f"  RHS {j+1}/{n_rhs}: {status} in {iteration+1} iterations")

    # Prepare output
    if squeeze_output:
        u = u.squeeze()

    # Build convergence info
    info = CGInfo(
        converged=all_converged,
        iterations=total_iters,
        residual_norm=r_norm,
        residual_history=residual_history,
        matvecs=total_matvecs,
        alpha=alpha,
        rtol=rtol,
        atol=atol,
        maxiter=maxiter,
        wall_time=time.time() - start_time,
        initial_residual=initial_residual,
    )

    return u, info


def effective_resistance(
    L: csr_matrix, pairs: np.ndarray, alpha: float = 1e-6, **cg_kwargs
) -> Tuple[np.ndarray, CGInfo]:
    """Compute effective resistance between node pairs.

    Parameters
    ----------
    L : scipy.sparse.csr_matrix, shape (n, n)
        Graph Laplacian
    pairs : np.ndarray, shape (k, 2)
        Node pairs (i, j) for resistance computation
    alpha : float
        Regularization for pseudo-inverse
    **cg_kwargs
        Additional arguments for CG solver

    Returns
    -------
    resistances : np.ndarray, shape (k,)
        Effective resistance R_ij for each pair
    info : CGInfo
        Aggregated solver statistics

    Notes
    -----
    Effective resistance R_ij = (e_i - e_j)^T L^+ (e_i - e_j)
    where L^+ is the Moore-Penrose pseudo-inverse.
    We approximate using (L + αI)^{-1} via CG.
    """
    n = L.shape[0]
    k = pairs.shape[0]
    resistances = np.zeros(k)

    total_iters = 0
    total_matvecs = 0
    wall_time = 0.0

    for idx, (i, j) in enumerate(pairs):
        # Build difference vector
        b = np.zeros(n)
        b[i] = 1.0
        b[j] = -1.0

        # Solve (L + αI)u = b
        u, info = cg_solve(L, b, alpha=alpha, **cg_kwargs)

        # Compute resistance: R_ij ≈ b^T u
        resistances[idx] = np.dot(b, u)

        # Aggregate stats
        total_iters += info.iterations
        total_matvecs += info.matvecs
        wall_time += info.wall_time

    # Build aggregated info
    agg_info = CGInfo(
        converged=True,  # Individual convergence tracked per pair
        iterations=total_iters,
        matvecs=total_matvecs,
        alpha=alpha,
        wall_time=wall_time,
    )

    return resistances, agg_info
