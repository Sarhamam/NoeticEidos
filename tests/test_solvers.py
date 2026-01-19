"""Tests for iterative solvers."""

import sys

import numpy as np
import pytest
from scipy.sparse import csr_matrix, diags, eye

sys.path.insert(0, "src")

from graphs.knn import build_graph
from graphs.laplacian import laplacian
from solvers.cg import cg_solve, effective_resistance
from solvers.lanczos import fiedler_vector, spectral_gap, topk_eigs
from solvers.preconditioners import build_preconditioner, jacobi_preconditioner


def tiny_laplacian():
    """Create a simple path graph Laplacian for testing."""
    # Path graph on 4 nodes: 0--1--2--3
    offdiag = np.array([-1, -1, -1])
    L = diags([offdiag, [1, 2, 2, 1], offdiag], [-1, 0, 1], shape=(4, 4))
    return L.tocsr()


class TestCG:
    """Tests for Conjugate Gradient solver."""

    def test_cg_converges_on_shifted_laplacian(self):
        """Test CG convergence on shifted Laplacian system."""
        L = tiny_laplacian()
        n = L.shape[0]
        b = np.ones(n)
        alpha = 1e-3

        u, info = cg_solve(L, b, alpha=alpha, rtol=1e-8, maxiter=2000)

        assert info.converged
        assert info.iterations < 100  # Should converge quickly

        # Check residual manually
        A = L + alpha * eye(n, format="csr")
        r = A.dot(u) - b
        rel_residual = np.linalg.norm(r) / np.linalg.norm(b)
        assert rel_residual <= 1e-6

    def test_cg_with_jacobi_preconditioner(self):
        """Test CG with Jacobi preconditioning."""
        # Build a slightly larger test problem
        X = np.random.randn(20, 3)
        A = build_graph(X, mode="additive", k=5, seed=42)
        L = laplacian(A, normalized=False)
        b = np.random.randn(20)

        # Without preconditioner
        u1, info1 = cg_solve(L, b, alpha=1e-3, M=None, seed=0)

        # With explicit Jacobi
        u2, info2 = cg_solve(L, b, alpha=1e-3, M="jacobi", seed=0)

        # Both should converge
        assert info1.converged
        assert info2.converged

        # Solutions should be close
        assert np.allclose(u1, u2, rtol=1e-5)

    def test_cg_multiple_rhs(self):
        """Test CG with multiple right-hand sides."""
        L = tiny_laplacian()
        n = L.shape[0]
        k = 3
        B = np.random.randn(n, k)

        U, info = cg_solve(L, B, alpha=1e-2, rtol=1e-6)

        assert U.shape == (n, k)
        assert info.converged

        # Check each solution
        A = L + 1e-2 * eye(n, format="csr")
        for j in range(k):
            r = A.dot(U[:, j]) - B[:, j]
            assert np.linalg.norm(r) / np.linalg.norm(B[:, j]) <= 1e-5

    def test_cg_zero_rhs(self):
        """Test CG with zero RHS (should give zero solution)."""
        L = tiny_laplacian()
        n = L.shape[0]
        b = np.zeros(n)

        u, info = cg_solve(L, b, alpha=1e-3)

        # Solution should be zero
        assert np.allclose(u, 0)
        assert info.converged

    def test_effective_resistance_symmetry(self):
        """Test that effective resistance is symmetric."""
        X = np.array([[0.0], [1.0], [2.0], [3.0]])
        A = build_graph(X, mode="additive", k=2, seed=0)
        L = laplacian(A, normalized=False)

        pairs = np.array([[0, 1], [1, 0], [0, 2], [2, 0]])
        resistances, _ = effective_resistance(L, pairs, alpha=1e-6)

        # R_ij should equal R_ji
        assert np.allclose(resistances[0], resistances[1])
        assert np.allclose(resistances[2], resistances[3])

        # Resistances should be positive
        assert np.all(resistances > 0)


class TestLanczos:
    """Tests for Lanczos eigenvalue computation."""

    def test_lanczos_low_spectrum_bounds(self):
        """Test Lanczos returns correct eigenvalue bounds."""
        X = np.array([[0.0], [1.0], [2.0], [3.0]])
        A = build_graph(X, mode="additive", k=2, seed=0)
        L = laplacian(A, normalized=True)

        evals, evecs, info = topk_eigs(L, k=3, which="SM")

        # Check dimensions
        assert len(evals) == 3
        assert evecs.shape == (4, 3)

        # PSD property: all eigenvalues >= 0
        assert np.all(evals >= -1e-9)

        # Normalized Laplacian: eigenvalues <= 2
        assert np.all(evals <= 2 + 1e-9)

        # First eigenvalue should be close to 0 (connected graph)
        assert evals[0] < 1e-6

    def test_lanczos_eigenvector_orthogonality(self):
        """Test that Lanczos eigenvectors are orthonormal."""
        X = np.random.randn(30, 5)
        A = build_graph(X, mode="additive", k=8, seed=123)
        L = laplacian(A, normalized=True)

        k = 10
        evals, evecs, info = topk_eigs(L, k=k, which="SM")

        # Check orthogonality: V^T V = I
        gram = evecs.T @ evecs
        assert np.allclose(gram, np.eye(k), atol=1e-8)

    def test_spectral_gap_computation(self):
        """Test spectral gap calculation."""
        evals = np.array([0.0, 0.1, 0.3, 0.5])
        gap = spectral_gap(evals)
        assert np.isclose(gap, 0.1)

        # Single eigenvalue case
        gap_single = spectral_gap(np.array([0.5]))
        assert gap_single == 0.0

    def test_fiedler_vector(self):
        """Test Fiedler vector computation."""
        # Create a simple graph with clear structure
        X = np.array([[0.0], [0.1], [2.0], [2.1]])  # Two clusters
        A = build_graph(X, mode="additive", k=2, sigma=0.5, seed=0)
        L = laplacian(A, normalized=False)

        fiedler, fiedler_val = fiedler_vector(L, seed=0)

        # Fiedler vector should separate clusters
        # Points 0,1 should have similar sign, points 2,3 opposite
        cluster1_sign = np.sign(fiedler[:2].mean())
        cluster2_sign = np.sign(fiedler[2:].mean())
        assert cluster1_sign != cluster2_sign

    def test_lanczos_convergence_info(self):
        """Test that Lanczos returns meaningful convergence info."""
        X = np.random.randn(20, 3)
        A = build_graph(X, mode="additive", k=5, seed=0)
        L = laplacian(A, normalized=True)

        evals, evecs, info = topk_eigs(L, k=5, tol=1e-8, maxiter=5000)

        assert info.n_converged > 0
        assert info.n_converged <= 5
        assert info.wall_time > 0
        if info.ritz_residuals is not None:
            assert len(info.ritz_residuals) == 5


class TestPreconditioners:
    """Tests for preconditioner utilities."""

    def test_jacobi_preconditioner_diagonal(self):
        """Test Jacobi preconditioner extracts correct diagonal."""
        # Create a matrix with known diagonal
        diag_vals = np.array([2.0, 3.0, 1.0, 4.0])
        A = diags(diag_vals, format="csr")

        M = jacobi_preconditioner(A)
        v = np.ones(4)
        Mv = M(v)

        # M^{-1} should be diagonal inverse
        expected = 1.0 / diag_vals
        assert np.allclose(Mv, expected)

    def test_jacobi_zero_diagonal_handling(self):
        """Test Jacobi handles zero diagonal entries."""
        # Matrix with zero diagonal entry
        A = csr_matrix(np.array([[0, 1, 0], [1, 2, 1], [0, 1, 3]]))

        M = jacobi_preconditioner(A, eps=1e-10)
        v = np.ones(3)
        Mv = M(v)

        # First entry should be 1 (no scaling due to zero diagonal)
        assert Mv[0] == 1.0
        assert np.isclose(Mv[1], 1.0 / 2.0)
        assert np.isclose(Mv[2], 1.0 / 3.0)

    def test_build_preconditioner_interface(self):
        """Test preconditioner factory function."""
        L = tiny_laplacian()

        # Test different methods
        M_none = build_preconditioner(L, method="none")
        M_jacobi = build_preconditioner(L, method="jacobi")

        v = np.array([1.0, 2.0, 3.0, 4.0])

        # Identity should not change vector
        assert np.allclose(M_none(v), v)

        # Jacobi should scale by diagonal
        diag_L = L.diagonal()
        expected = v / np.where(diag_L != 0, diag_L, 1.0)
        assert np.allclose(M_jacobi(v), expected)

        # ILU should raise NotImplementedError
        with pytest.raises(NotImplementedError):
            build_preconditioner(L, method="ilu")
