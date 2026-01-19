"""Iterative solvers for sparse linear systems and eigenproblems."""

from .cg import cg_solve, effective_resistance
from .lanczos import fiedler_vector, spectral_gap, topk_eigs
from .preconditioners import (
    DiagonalPreconditioner,
    build_preconditioner,
    identity_preconditioner,
    jacobi_preconditioner,
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
