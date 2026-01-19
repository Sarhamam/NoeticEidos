"""Tangent space projections for constrained dynamics."""

import numpy as np
from scipy.linalg import pinv, svd


def project_to_tangent(J_f: np.ndarray, method: str = "svd") -> np.ndarray:
    """Project onto tangent space T_x Z = ker(J_f).

    Parameters
    ----------
    J_f : np.ndarray, shape (2, d)
        Jacobian matrix at point x
    method : str
        Method for computing projection:
        - "svd": Use SVD-based pseudoinverse
        - "normal": Use normal equations J_f^T (J_f J_f^T)^{-1} J_f

    Returns
    -------
    P : np.ndarray, shape (d, d)
        Projection matrix P = I - J_f^T (J_f J_f^T)^{-1} J_f

    Notes
    -----
    The tangent space is T_x Z = {v : J_f(x) v = 0}.
    The projector satisfies P^2 = P, P^T = P.
    """
    if J_f.shape[0] != 2:
        raise ValueError("Jacobian must have 2 rows (codimension 2)")

    d = J_f.shape[1]
    I = np.eye(d)

    if method == "svd":
        # SVD-based computation (more stable)
        U, s, Vt = svd(J_f, full_matrices=False)

        # Check rank
        rank = np.sum(s > 1e-12)
        if rank < 2:
            raise ValueError(f"Jacobian is rank deficient (rank={rank}, expected 2)")

        # P = I - V S^{-1} U^T where J_f = U S V^T
        # P = I - J_f^+ J_f where J_f^+ is pseudoinverse
        J_f_pinv = Vt.T @ np.diag(1.0 / s) @ U.T
        P = I - J_f_pinv @ J_f

    elif method == "normal":
        # Normal equations approach
        JJT = J_f @ J_f.T
        try:
            JJT_inv = np.linalg.inv(JJT)
        except np.linalg.LinAlgError:
            # Fallback to pseudoinverse
            JJT_inv = pinv(JJT)

        P = I - J_f.T @ JJT_inv @ J_f

    else:
        raise ValueError(f"Unknown method: {method}")

    return P


def project_vector(v: np.ndarray, J_f: np.ndarray) -> np.ndarray:
    """Project vector onto tangent space.

    Parameters
    ----------
    v : np.ndarray, shape (d,)
        Vector to project
    J_f : np.ndarray, shape (2, d)
        Jacobian matrix

    Returns
    -------
    v_proj : np.ndarray, shape (d,)
        Projected vector v_proj = P v where P is tangent projector
    """
    P = project_to_tangent(J_f)
    return P @ v


def tangent_basis(J_f: np.ndarray) -> np.ndarray:
    """Compute orthonormal basis for tangent space.

    Parameters
    ----------
    J_f : np.ndarray, shape (2, d)
        Jacobian matrix

    Returns
    -------
    basis : np.ndarray, shape (d, d-2)
        Orthonormal basis vectors for T_x Z

    Notes
    -----
    The tangent space has dimension d-2 (codimension 2).
    Basis vectors satisfy J_f @ basis = 0.
    """
    J_f.shape[1]

    # SVD of J_f
    U, s, Vt = svd(J_f, full_matrices=True)

    # Check rank
    rank = np.sum(s > 1e-12)
    if rank < 2:
        raise ValueError(f"Jacobian is rank deficient (rank={rank})")

    # Tangent space basis is null space of J_f
    # These are the last (d-2) columns of V
    basis = Vt[rank:, :].T

    return basis


def check_projection_properties(P: np.ndarray, tol: float = 1e-10) -> dict:
    """Verify that P is a valid projection matrix.

    Parameters
    ----------
    P : np.ndarray, shape (d, d)
        Candidate projection matrix
    tol : float
        Numerical tolerance

    Returns
    -------
    props : dict
        Properties:
        - is_symmetric: P = P^T
        - is_idempotent: P^2 = P
        - rank: numerical rank
        - eigenvalues: sorted eigenvalues

    Notes
    -----
    A projection matrix must satisfy:
    1. P^T = P (symmetric)
    2. P^2 = P (idempotent)
    3. Eigenvalues ∈ {0, 1}
    """
    # Check symmetry
    is_symmetric = np.allclose(P, P.T, atol=tol)

    # Check idempotency
    is_idempotent = np.allclose(P @ P, P, atol=tol)

    # Compute eigenvalues
    evals = np.linalg.eigvals(P)
    evals_sorted = np.sort(evals)

    # Numerical rank
    rank = np.sum(evals > tol)

    # Check eigenvalues are 0 or 1
    binary_evals = np.allclose(evals, np.round(evals), atol=tol)

    return {
        "is_symmetric": is_symmetric,
        "is_idempotent": is_idempotent,
        "rank": rank,
        "eigenvalues": evals_sorted,
        "binary_eigenvalues": binary_evals,
        "trace": np.trace(P),
    }


def constraint_violation(f_vals: np.ndarray) -> np.ndarray:
    """Measure constraint violation ||f(x)||.

    Parameters
    ----------
    f_vals : np.ndarray, shape (n, 2)
        Submersion values f(x)

    Returns
    -------
    violations : np.ndarray, shape (n,)
        Constraint violations ||f(x)||
    """
    return np.linalg.norm(f_vals, axis=1)


def project_to_manifold(
    X: np.ndarray,
    f: callable,
    jacobian: callable,
    max_iter: int = 10,
    tol: float = 1e-8,
) -> np.ndarray:
    """Project points onto constraint manifold Z = {x : f(x) = 0}.

    Parameters
    ----------
    X : np.ndarray, shape (n, d)
        Initial points
    f : callable
        Submersion function
    jacobian : callable
        Jacobian function
    max_iter : int
        Maximum Newton iterations
    tol : float
        Convergence tolerance

    Returns
    -------
    X_proj : np.ndarray, shape (n, d)
        Projected points on manifold

    Notes
    -----
    Uses Newton's method: x_{k+1} = x_k - J_f^+ f(x_k)
    where J_f^+ is the pseudoinverse.
    """
    X_proj = X.copy()
    n, d = X.shape

    for i in range(n):
        x = X_proj[i]

        for _iteration in range(max_iter):
            # Evaluate submersion and Jacobian
            f_val = f(x.reshape(1, -1)).flatten()
            J_val = jacobian(x.reshape(1, -1))[0]  # Shape (2, d)

            # Check convergence
            if np.linalg.norm(f_val) < tol:
                break

            # Newton step: x_{k+1} = x_k - J_f^+ f(x_k)
            try:
                J_pinv = pinv(J_val)
                x = x - J_pinv @ f_val
            except np.linalg.LinAlgError:
                break

        X_proj[i] = x

    return X_proj
