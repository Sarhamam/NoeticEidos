"""Tangent space projection for constrained dynamics on manifolds."""

import numpy as np
from scipy.linalg import svd


def projected_velocity(v: np.ndarray, J_f: np.ndarray) -> np.ndarray:
    """Project velocity vector onto tangent space T_x Z = ker(J_f).

    Parameters
    ----------
    v : np.ndarray, shape (d,)
        Velocity vector in ambient space
    J_f : np.ndarray, shape (2, d)
        Jacobian matrix of submersion f at point x

    Returns
    -------
    v_proj : np.ndarray, shape (d,)
        Projected velocity P_{T_x Z} v

    Notes
    -----
    Uses the formula: P_{T_x Z} = I - J_f^T (J_f J_f^T)^{-1} J_f
    The projected velocity satisfies J_f @ v_proj = 0 (tangent to constraints).

    Raises
    ------
    ValueError
        If J_f is not rank 2 (transversality failure)
    """
    if J_f.shape[0] < 1:
        raise ValueError("Jacobian must have at least 1 row")
    if J_f.shape[0] > J_f.shape[1]:
        raise ValueError(
            "Jacobian cannot have more rows than columns (overconstrained)"
        )

    d = J_f.shape[1]

    # Check rank using SVD
    U, s, Vt = svd(J_f, full_matrices=False)
    rank = np.sum(s > 1e-12)
    expected_rank = J_f.shape[0]

    if rank < expected_rank:
        raise ValueError(
            f"Jacobian is rank deficient (rank={rank}, expected {expected_rank}). "
            "Transversality condition violated."
        )

    # Ensure v is 1D vector
    v_flat = np.asarray(v).flatten()
    if len(v_flat) != d:
        raise ValueError(
            f"Velocity vector has wrong shape: expected ({d},), got {v.shape}"
        )

    # Compute projection: P = I - J_f^T (J_f J_f^T)^{-1} J_f
    I = np.eye(d)

    # Use SVD for numerical stability
    # Only use non-zero singular values
    s_inv = np.zeros_like(s)
    s_inv[:rank] = 1.0 / s[:rank]
    J_f_pinv = Vt.T @ np.diag(s_inv) @ U.T
    P = I - J_f_pinv @ J_f

    return P @ v_flat


def projected_gradient_step(
    x: np.ndarray, grad: np.ndarray, J_f: np.ndarray, step_size: float = 0.01
) -> np.ndarray:
    """Take projected gradient step constrained to manifold.

    Parameters
    ----------
    x : np.ndarray, shape (d,)
        Current point
    grad : np.ndarray, shape (d,)
        Gradient vector
    J_f : np.ndarray, shape (2, d)
        Jacobian of constraints at x
    step_size : float
        Step size for gradient descent

    Returns
    -------
    x_new : np.ndarray, shape (d,)
        New point after projected gradient step
    """
    # Project gradient to tangent space
    grad_proj = projected_velocity(-grad, J_f)  # Negative for descent

    # Take step
    x_new = x + step_size * grad_proj

    return x_new


def check_tangency(v: np.ndarray, J_f: np.ndarray, tol: float = 1e-10) -> bool:
    """Check if vector is tangent to constraint manifold.

    Parameters
    ----------
    v : np.ndarray, shape (d,)
        Vector to check
    J_f : np.ndarray, shape (2, d)
        Jacobian of constraints
    tol : float
        Numerical tolerance

    Returns
    -------
    is_tangent : bool
        True if J_f @ v ≈ 0
    """
    constraint_violation = J_f @ v
    return np.linalg.norm(constraint_violation) < tol


def parallel_transport_approximation(
    v: np.ndarray, x0: np.ndarray, x1: np.ndarray, J_f_func: callable
) -> np.ndarray:
    """Approximate parallel transport of tangent vector.

    Parameters
    ----------
    v : np.ndarray, shape (d,)
        Tangent vector at x0
    x0 : np.ndarray, shape (d,)
        Starting point
    x1 : np.ndarray, shape (d,)
        Ending point
    J_f_func : callable
        Function that computes Jacobian J_f(x)

    Returns
    -------
    v_transported : np.ndarray, shape (d,)
        Approximated parallel transport of v from x0 to x1

    Notes
    -----
    Simple approximation: project v onto tangent space at x1.
    For true parallel transport, would need connection/metric information.
    """
    J_f_1 = J_f_func(x1)
    return projected_velocity(v, J_f_1)


def constraint_force(
    f_vals: np.ndarray, J_f: np.ndarray, stiffness: float = 1.0
) -> np.ndarray:
    """Compute constraint force to restore points to manifold.

    Parameters
    ----------
    f_vals : np.ndarray, shape (2,)
        Constraint violation: f(x) should be 0
    J_f : np.ndarray, shape (2, d)
        Jacobian of constraints
    stiffness : float
        Stiffness parameter for constraint restoration

    Returns
    -------
    force : np.ndarray, shape (d,)
        Force vector F = -stiffness * J_f^T f(x)

    Notes
    -----
    This force pulls points back toward the constraint manifold.
    Used in penalty methods for constraint enforcement.
    """
    return -stiffness * J_f.T @ f_vals


def tangent_space_basis(J_f: np.ndarray) -> np.ndarray:
    """Compute orthonormal basis for tangent space.

    Parameters
    ----------
    J_f : np.ndarray, shape (2, d)
        Jacobian matrix

    Returns
    -------
    basis : np.ndarray, shape (d, d-2)
        Orthonormal basis vectors for tangent space

    Notes
    -----
    Tangent space T_x Z = ker(J_f) has dimension d-2.
    Uses SVD to find null space basis.
    """
    J_f.shape[1]

    # SVD of J_f
    U, s, Vt = svd(J_f, full_matrices=True)

    # Check rank
    rank = np.sum(s > 1e-12)
    if rank < 2:
        raise ValueError(f"Jacobian is rank deficient (rank={rank})")

    # Null space basis: last (d-2) rows of V^T
    basis = Vt[rank:, :].T

    return basis


def project_matrix_to_tangent(A: np.ndarray, J_f: np.ndarray) -> np.ndarray:
    """Project matrix to act only in tangent directions.

    Parameters
    ----------
    A : np.ndarray, shape (d, d)
        Matrix to project
    J_f : np.ndarray, shape (2, d)
        Jacobian of constraints

    Returns
    -------
    A_proj : np.ndarray, shape (d, d)
        Projected matrix A_proj = P A P where P is tangent projector

    Notes
    -----
    Useful for projecting Hessians or metric tensors to tangent space.
    """
    # Get projector
    d = J_f.shape[1]
    I = np.eye(d)

    # Use SVD for stability
    U, s, Vt = svd(J_f, full_matrices=False)
    J_f_pinv = Vt.T @ np.diag(1.0 / s) @ U.T
    P = I - J_f_pinv @ J_f

    # Project: A_proj = P A P
    A_proj = P @ A @ P

    return A_proj
