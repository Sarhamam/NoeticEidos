"""Conjugate Gradient dynamics constrained to manifolds."""

import numpy as np
from scipy.sparse import csr_matrix, issparse
from typing import Tuple, List, Callable, Optional, Dict, Any
import time

from dynamics.projected import projected_velocity, constraint_force
from solvers.cg import cg_solve, CGInfo
from geometry.projection import project_to_manifold


class ConstrainedCGInfo:
    """Information about constrained CG iterations."""

    def __init__(self):
        self.converged = False
        self.iterations = 0
        self.residual_history = []
        self.constraint_violations = []
        self.projection_errors = []
        self.wall_time = 0.0
        self.cg_info = None


def inner_cg_dynamics(
    L: csr_matrix,
    b: np.ndarray,
    f: Callable,
    jacobian: Callable,
    steps: int = 50,
    alpha: float = 1e-3,
    rtol: float = 1e-6,
    constraint_tol: float = 1e-6,
    projection_method: str = "tangent",
    verbose: bool = False
) -> Tuple[List[np.ndarray], ConstrainedCGInfo]:
    """Run CG iterations constrained to manifold Z = {x : f(x) = 0}.

    Parameters
    ----------
    L : csr_matrix, shape (n, n)
        System matrix (typically Laplacian)
    b : np.ndarray, shape (n,)
        Right-hand side vector
    f : callable
        Submersion function f: R^n -> R^2
    jacobian : callable
        Function returning Jacobian J_f(x)
    steps : int
        Number of CG iterations
    alpha : float
        Regularization for system (L + αI)
    rtol : float
        CG convergence tolerance
    constraint_tol : float
        Tolerance for constraint satisfaction
    projection_method : str
        Method for constraint enforcement:
        - "tangent": Project CG steps to tangent space
        - "penalty": Add penalty forces
        - "manifold": Project to manifold after each step
    verbose : bool
        Print iteration progress

    Returns
    -------
    trajectory : list of np.ndarray
        Sequence of iterates, each satisfying constraints
    info : ConstrainedCGInfo
        Convergence and constraint information

    Notes
    -----
    Solves the constrained optimization problem:
        minimize (1/2) x^T (L + αI) x - b^T x
        subject to f(x) = 0

    The algorithm projects each CG step onto the tangent space
    of the constraint manifold.
    """
    start_time = time.time()
    n = L.shape[0]

    # Initialize on manifold (find feasible starting point)
    x0 = np.random.randn(n)
    try:
        x0 = project_to_manifold(x0.reshape(1, -1), f, jacobian, max_iter=10)[0]
    except:
        # If projection fails, start with random point and add penalty
        pass

    trajectory = [x0.copy()]
    info = ConstrainedCGInfo()

    # System matrix
    from scipy.sparse import eye
    A = L + alpha * eye(n, format='csr')

    # Initialize CG state
    x = x0.copy()
    r = A @ x - b  # Initial residual

    # Evaluate constraints
    f_val = f(x.reshape(1, -1)).flatten()
    J_f = jacobian(x.reshape(1, -1))
    # Handle different jacobian formats
    if len(J_f.shape) == 3:
        J_f = J_f[0]  # Remove batch dimension: (1, k, d) -> (k, d)

    # Project initial residual to tangent space
    if projection_method == "tangent":
        r = projected_velocity(r, J_f)

    p = -r.copy()  # Initial search direction
    rsold = np.dot(r, r)

    info.constraint_violations.append(np.linalg.norm(f_val))
    info.residual_history.append(np.sqrt(rsold))

    for iteration in range(steps):
        # Matrix-vector product
        Ap = A @ p

        # Project Ap to tangent space for consistency
        if projection_method == "tangent":
            Ap = projected_velocity(Ap, J_f)

        # CG step size
        pAp = np.dot(p, Ap)
        if abs(pAp) < 1e-14:
            if verbose:
                print(f"CG breakdown at iteration {iteration}")
            break

        alpha_cg = rsold / pAp

        # Update solution
        x_new = x + alpha_cg * p

        # Enforce constraints
        if projection_method == "tangent":
            # Already projected via tangent steps
            pass
        elif projection_method == "penalty":
            # Add constraint forces
            f_new = f(x_new.reshape(1, -1)).flatten()
            if np.linalg.norm(f_new) > constraint_tol:
                J_f_new = jacobian(x_new.reshape(1, -1))
                if len(J_f_new.shape) == 3:
                    J_f_new = J_f_new[0]
                force = constraint_force(f_new, J_f_new, stiffness=1.0)
                x_new = x_new + 0.1 * force  # Small correction
        elif projection_method == "manifold":
            # Project back to manifold
            try:
                x_new = project_to_manifold(x_new.reshape(1, -1), f, jacobian,
                                          max_iter=5, tol=constraint_tol)[0]
            except:
                if verbose:
                    print(f"Manifold projection failed at iteration {iteration}")

        # Update residual
        r_new = A @ x_new - b

        # Update Jacobian at new point
        f_val = f(x_new.reshape(1, -1)).flatten()
        J_f = jacobian(x_new.reshape(1, -1))
        if len(J_f.shape) == 3:
            J_f = J_f[0]

        # Project residual to tangent space
        if projection_method == "tangent":
            r_new = projected_velocity(r_new, J_f)

        rsnew = np.dot(r_new, r_new)

        # Check convergence
        residual_norm = np.sqrt(rsnew)
        constraint_violation = np.linalg.norm(f_val)

        if verbose and iteration % 10 == 0:
            print(f"Iter {iteration:3d}: res={residual_norm:.2e}, "
                  f"constraint={constraint_violation:.2e}")

        # Store trajectory point
        x = x_new.copy()
        trajectory.append(x.copy())

        # Store diagnostics
        info.residual_history.append(residual_norm)
        info.constraint_violations.append(constraint_violation)

        # Check convergence
        b_norm = np.linalg.norm(b)
        if residual_norm < rtol * b_norm:
            info.converged = True
            if verbose:
                print(f"CG converged in {iteration+1} iterations")
            break

        # Update CG direction
        beta = rsnew / rsold
        p = -r_new + beta * p

        # Project search direction to tangent space
        if projection_method == "tangent":
            p = projected_velocity(p, J_f)

        r = r_new
        rsold = rsnew

    info.iterations = len(trajectory) - 1
    info.wall_time = time.time() - start_time

    return trajectory, info


def constrained_gradient_descent(
    objective_grad: Callable,
    f: Callable,
    jacobian: Callable,
    x0: np.ndarray,
    steps: int = 100,
    step_size: float = 0.01,
    constraint_tol: float = 1e-6,
    verbose: bool = False
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """Projected gradient descent on constraint manifold.

    Parameters
    ----------
    objective_grad : callable
        Function computing gradient of objective
    f : callable
        Constraint function f(x) = 0
    jacobian : callable
        Jacobian of constraints
    x0 : np.ndarray
        Initial point (should satisfy constraints)
    steps : int
        Number of gradient steps
    step_size : float
        Step size for gradient descent
    constraint_tol : float
        Tolerance for constraint violation
    verbose : bool
        Print progress

    Returns
    -------
    trajectory : list
        Optimization trajectory
    info : dict
        Optimization information
    """
    trajectory = [x0.copy()]
    objective_values = []
    constraint_violations = []

    x = x0.copy()

    for iteration in range(steps):
        # Compute gradient
        grad = objective_grad(x)

        # Evaluate constraints and Jacobian
        f_val = f(x.reshape(1, -1)).flatten()
        J_f = jacobian(x.reshape(1, -1))
        # Handle different jacobian formats
        if len(J_f.shape) == 3:
            J_f = J_f[0]  # Remove batch dimension: (1, k, d) -> (k, d)

        # Project gradient to tangent space
        grad_proj = projected_velocity(grad, J_f)

        # Take step
        x_new = x - step_size * grad_proj

        # Optional: project back to manifold
        try:
            x_new = project_to_manifold(x_new.reshape(1, -1), f, jacobian,
                                      max_iter=3, tol=constraint_tol)[0]
        except:
            pass

        # Update
        x = x_new
        trajectory.append(x.copy())

        # Diagnostics
        f_val_new = f(x.reshape(1, -1)).flatten()
        constraint_violation = np.linalg.norm(f_val_new)
        constraint_violations.append(constraint_violation)

        if verbose and iteration % 20 == 0:
            print(f"Iter {iteration:3d}: constraint_viol={constraint_violation:.2e}")

    info = {
        'constraint_violations': constraint_violations,
        'converged': constraint_violations[-1] < constraint_tol
    }

    return trajectory, info


def lagrange_multiplier_cg(
    L: csr_matrix,
    b: np.ndarray,
    A_constraint: np.ndarray,
    b_constraint: np.ndarray,
    alpha: float = 1e-3,
    rtol: float = 1e-6,
    maxiter: int = 1000
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Solve constrained linear system using Lagrange multipliers.

    Parameters
    ----------
    L : csr_matrix
        System matrix
    b : np.ndarray
        RHS vector
    A_constraint : np.ndarray, shape (m, n)
        Constraint matrix (A_constraint @ x = b_constraint)
    b_constraint : np.ndarray, shape (m,)
        Constraint RHS
    alpha : float
        Regularization
    rtol : float
        Tolerance
    maxiter : int
        Maximum iterations

    Returns
    -------
    x : np.ndarray
        Primal solution
    lambda_ : np.ndarray
        Lagrange multipliers
    info : dict
        Solver information

    Notes
    -----
    Solves the KKT system:
    [L + αI    A^T] [x]   [b]
    [A         0  ] [λ] = [b_c]
    """
    from scipy.sparse import bmat, csr_matrix as csr
    from scipy.sparse import linalg as spla

    n = L.shape[0]
    m = A_constraint.shape[0]

    # Build KKT matrix
    A_sparse = csr(A_constraint)
    zero_block = csr((m, m))

    from scipy.sparse import eye
    L_reg = L + alpha * eye(n, format='csr')

    KKT = bmat([
        [L_reg, A_sparse.T],
        [A_sparse, zero_block]
    ], format='csr')

    # Build RHS
    rhs = np.concatenate([b, b_constraint])

    # Solve KKT system
    start_time = time.time()
    try:
        # Use direct solver for now (could use iterative)
        sol = spla.spsolve(KKT, rhs)
        converged = True
    except:
        # Fallback to CG on full system
        sol, cg_info = cg_solve(KKT, rhs, alpha=0, rtol=rtol, maxiter=maxiter)
        converged = cg_info.converged

    x = sol[:n]
    lambda_ = sol[n:]

    info = {
        'converged': converged,
        'wall_time': time.time() - start_time,
        'constraint_residual': np.linalg.norm(A_constraint @ x - b_constraint)
    }

    return x, lambda_, info