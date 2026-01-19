"""Utilities for iterative solvers."""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class CGInfo:
    """Convergence information for CG solver."""

    converged: bool = False
    iterations: int = 0
    residual_norm: float = float("inf")
    residual_history: List[float] = field(default_factory=list)
    matvecs: int = 0
    alpha: float = 0.0
    rtol: float = 1e-6
    atol: float = 0.0
    maxiter: int = 2000
    wall_time: float = 0.0
    initial_residual: float = 0.0

    def __repr__(self):
        status = "converged" if self.converged else "not converged"
        return (
            f"CGInfo({status}, "
            f"iter={self.iterations}, "
            f"res={self.residual_norm:.2e}, "
            f"time={self.wall_time:.3f}s)"
        )


@dataclass
class LanczosInfo:
    """Convergence information for Lanczos solver."""

    converged: bool = False
    iterations: int = 0
    n_converged: int = 0
    ritz_residuals: Optional[np.ndarray] = None
    matvecs: int = 0
    wall_time: float = 0.0
    tol: float = 1e-6
    maxiter: int = 5000

    def __repr__(self):
        status = "converged" if self.converged else "not converged"
        return (
            f"LanczosInfo({status}, "
            f"n_conv={self.n_converged}, "
            f"iter={self.iterations}, "
            f"time={self.wall_time:.3f}s)"
        )


def log_iteration(iteration: int, residual: float, verbose: bool = False):
    """Log CG iteration progress."""
    if verbose and iteration % 10 == 0:
        print(f"  CG iter {iteration:4d}: residual = {residual:.2e}")


def check_convergence(
    residual_norm: float, b_norm: float, rtol: float = 1e-6, atol: float = 0.0
) -> bool:
    """Check CG convergence criteria.

    Converged if: ||r|| <= rtol * ||b|| + atol
    """
    return residual_norm <= rtol * b_norm + atol
