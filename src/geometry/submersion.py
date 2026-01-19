"""Submersion maps f: M → ℝ² with transversality checking."""

import numpy as np
from typing import Tuple, Callable, Dict, Any, Optional, Literal
from scipy.linalg import svd


def build_submersion(
    X: np.ndarray,
    method: Literal["linear", "least_squares"] = "linear",
    seed: Optional[int] = 0
) -> Tuple[Callable, Callable]:
    """Build submersion map f = (τ, σ): M → ℝ².

    Parameters
    ----------
    X : np.ndarray, shape (n, d)
        Reference dataset for determining submersion
    method : {"linear", "least_squares"}
        Construction method:
        - "linear": Two random linear forms f(x) = (a^T x, b^T x)
        - "least_squares": Fit to target if provided
    seed : int or None
        Random seed for reproducible submersion

    Returns
    -------
    f : callable
        Submersion function f(X) -> (τ, σ) values, shape (n, 2)
    jacobian : callable
        Jacobian function J_f(X) -> array shape (n, 2, d)

    Notes
    -----
    The submersion defines the constraint manifold Z = {x : f(x) = 0}.
    Transversality requires rank(J_f) = 2 everywhere on Z.
    """
    if seed is not None:
        np.random.seed(seed)

    n, d = X.shape

    if method == "linear":
        # Sample two independent random vectors
        a = np.random.randn(d)
        b = np.random.randn(d)

        # Ensure independence via Gram-Schmidt if needed
        a = a / np.linalg.norm(a)
        b_perp = b - np.dot(b, a) * a
        if np.linalg.norm(b_perp) < 1e-10:
            # Retry with different random vector
            b = np.random.randn(d)
            b_perp = b - np.dot(b, a) * a
        b = b_perp / np.linalg.norm(b_perp)

        def f(Y):
            """Submersion function f(x) = (a^T x, b^T x)."""
            if Y.ndim == 1:
                Y = Y.reshape(1, -1)
            tau = Y @ a
            sigma = Y @ b
            return np.column_stack([tau, sigma])

        def jacobian(Y):
            """Jacobian J_f(x) = [a; b] (constant 2×d matrix)."""
            if Y.ndim == 1:
                Y = Y.reshape(1, -1)
            m = Y.shape[0]
            J = np.zeros((m, 2, d))
            J[:, 0, :] = a[None, :]
            J[:, 1, :] = b[None, :]
            return J

    elif method == "least_squares":
        # For now, implement as random linear (can extend later)
        # In future: fit to minimize some target objective
        return build_submersion(X, method="linear", seed=seed)

    else:
        raise ValueError(f"Unknown method: {method}")

    return f, jacobian


def check_transversal(
    submersion: Tuple[Callable, Callable],
    X: np.ndarray,
    tol_rank: float = 1e-6,
    kappa_max: float = 1e6,
    zero_threshold: float = 1e-3
) -> Tuple[bool, Dict[str, Any]]:
    """Check transversality condition for submersion.

    Parameters
    ----------
    submersion : tuple of (f, jacobian)
        Submersion functions
    X : np.ndarray, shape (n, d)
        Points to check
    tol_rank : float
        Tolerance for rank determination (min singular value)
    kappa_max : float
        Maximum allowed condition number
    zero_threshold : float
        Threshold for considering f(x) ≈ 0

    Returns
    -------
    ok : bool
        True if transversality holds
    cert : dict
        Certificate with diagnostics:
        - min_singular: smallest singular value
        - max_condition: largest condition number
        - n_zero_points: number of near-zero points
        - rank_deficient: number of rank-deficient Jacobians

    Notes
    -----
    Transversality requires:
    1. On Z = {x : f(x) = 0}, rank(J_f(x)) = 2
    2. Condition number κ(J_f^T J_f) ≤ κ_max
    """
    f, jacobian = submersion
    n, d = X.shape

    # Evaluate submersion at all points
    f_vals = f(X)
    J_vals = jacobian(X)

    # Find points near zero set
    f_norms = np.linalg.norm(f_vals, axis=1)
    near_zero = f_norms < zero_threshold
    n_zero_points = np.sum(near_zero)

    if n_zero_points == 0:
        # Generate some points closer to zero set using Newton steps
        # Simple approach: take points with smallest f_norms
        n_candidates = min(10, n)
        candidates_idx = np.argsort(f_norms)[:n_candidates]
        near_zero = np.zeros(n, dtype=bool)
        near_zero[candidates_idx] = True
        n_zero_points = n_candidates

    # Check rank and condition at near-zero points
    min_singular = float('inf')
    max_condition = 0.0
    rank_deficient = 0

    for i in range(n):
        if not near_zero[i]:
            continue

        J_i = J_vals[i]  # Shape (2, d)

        # Compute SVD of J_i
        try:
            U, s, Vt = svd(J_i, full_matrices=False)
            min_singular = min(min_singular, s.min())

            # Condition number of J_i^T J_i
            if s.min() > 1e-14:
                cond = (s.max() / s.min()) ** 2
                max_condition = max(max_condition, cond)
            else:
                max_condition = float('inf')

            # Check rank
            if s.min() < tol_rank:
                rank_deficient += 1

        except np.linalg.LinAlgError:
            rank_deficient += 1
            max_condition = float('inf')

    # Determine if transversal
    ok = (rank_deficient == 0) and (max_condition <= kappa_max) and (min_singular >= tol_rank)

    cert = {
        'min_singular': min_singular if min_singular != float('inf') else 0.0,
        'max_condition': max_condition,
        'n_zero_points': n_zero_points,
        'rank_deficient': rank_deficient,
        'total_checked': np.sum(near_zero),
        'tol_rank': tol_rank,
        'kappa_max': kappa_max
    }

    return ok, cert


def find_zero_set(
    f: Callable,
    X: np.ndarray,
    method: str = "newton",
    max_iter: int = 50,
    tol: float = 1e-8
) -> np.ndarray:
    """Find points on zero set Z = {x : f(x) = 0}.

    Parameters
    ----------
    f : callable
        Submersion function
    X : np.ndarray, shape (n, d)
        Initial points
    method : str
        Root finding method ("newton" or "minimize")
    max_iter : int
        Maximum iterations
    tol : float
        Convergence tolerance

    Returns
    -------
    Z_points : np.ndarray, shape (m, d)
        Points on zero set

    Notes
    -----
    This uses numerical root finding to project points onto Z.
    For Newton method, requires Jacobian information.
    """
    from scipy.optimize import fsolve

    Z_points = []

    for i, x0 in enumerate(X):
        try:
            if method == "newton":
                # Use fsolve (hybrid Newton method)
                def objective(x):
                    return f(x.reshape(1, -1)).flatten()

                sol = fsolve(objective, x0, xtol=tol, maxfev=max_iter)
                f_val = objective(sol)

                if np.linalg.norm(f_val) < tol:
                    Z_points.append(sol)

        except Exception:
            continue

    if len(Z_points) == 0:
        # Return empty array with correct shape
        return np.array([]).reshape(0, X.shape[1])

    return np.array(Z_points)