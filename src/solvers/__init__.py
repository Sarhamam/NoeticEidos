"""Iterative solvers for sparse linear systems and eigenproblems."""

from .cg import cg_solve, effective_resistance
from .lanczos import topk_eigs, spectral_gap, fiedler_vector
from .preconditioners import (
    build_preconditioner,
    jacobi_preconditioner,
    identity_preconditioner,
    DiagonalPreconditioner
)
from .utils import CGInfo, LanczosInfo

__all__ = [
    # CG solver
    "cg_solve",
    "effective_resistance",
    # Lanczos eigensolvers
    "topk_eigs",
    "spectral_gap",
    "fiedler_vector",
    # Preconditioners
    "build_preconditioner",
    "jacobi_preconditioner",
    "identity_preconditioner",
    "DiagonalPreconditioner",
    # Info classes
    "CGInfo",
    "LanczosInfo",
]