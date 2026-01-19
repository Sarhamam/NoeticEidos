"""Tests for graph construction and Laplacian computation."""

import numpy as np
import pytest
from scipy.sparse import issparse
from scipy.sparse.linalg import eigsh
import sys
sys.path.insert(0, 'src')

from graphs.knn import build_graph
from graphs.laplacian import laplacian


class TestBuildGraph:
    """Tests for k-NN graph construction."""

    def test_build_graph_additive_basic(self):
        """Test basic additive mode graph construction."""
        X = np.array([[0.0], [1.0], [2.0]])
        A = build_graph(X, mode="additive", k=2, seed=42)

        assert A.shape == (3, 3)
        assert issparse(A)
        assert (A.toarray() == A.toarray().T).all()  # Symmetry
        assert np.allclose(A.diagonal(), 0)  # No self-loops
        assert A.nnz > 0  # Has edges

    def test_build_graph_multiplicative_basic(self):
        """Test basic multiplicative mode graph construction."""
        X = np.array([[1.0], [2.0], [4.0]])
        A = build_graph(X, mode="multiplicative", k=2, eps=1e-6, seed=42)

        assert A.shape == (3, 3)
        assert issparse(A)
        assert (A.toarray() == A.toarray().T).all()  # Symmetry
        assert np.allclose(A.diagonal(), 0)  # No self-loops

    def test_different_modes_produce_different_graphs(self):
        """Verify additive and multiplicative modes give different results."""
        np.random.seed(123)
        X = np.random.exponential(2.0, size=(10, 3))

        A_add = build_graph(X, mode="additive", k=3, seed=0)
        A_mult = build_graph(X, mode="multiplicative", k=3, seed=0)

        # The graphs should be different
        diff = np.abs(A_add.toarray() - A_mult.toarray()).sum()
        assert diff > 0.1  # Substantially different

    def test_median_bandwidth_estimation(self):
        """Test median bandwidth parameter estimation."""
        X = np.array([[0.0], [1.0], [2.0], [10.0]])  # Last point is outlier
        A = build_graph(X, mode="additive", k=2, sigma="median", seed=0)

        # Should have non-zero weights despite outlier
        assert A.nnz > 0

    def test_fixed_bandwidth(self):
        """Test using fixed bandwidth parameters."""
        X = np.array([[0.0], [1.0], [2.0]])

        A1 = build_graph(X, mode="additive", k=2, sigma=0.5, seed=0)
        A2 = build_graph(X, mode="additive", k=2, sigma=2.0, seed=0)

        # Larger bandwidth should give larger weights
        assert A2.data.mean() > A1.data.mean()

    def test_k_neighbors_limit(self):
        """Test k is properly bounded by dataset size."""
        X = np.array([[0.0], [1.0], [2.0]])
        A = build_graph(X, mode="additive", k=10, seed=0)  # k > n

        # Should not crash and produce valid graph
        assert A.shape == (3, 3)
        assert A.nnz > 0

    def test_reproducibility_with_seed(self):
        """Test that same seed produces same graph."""
        X = np.random.randn(10, 3)

        A1 = build_graph(X, mode="additive", k=3, seed=42)
        A2 = build_graph(X, mode="additive", k=3, seed=42)

        assert np.allclose(A1.toarray(), A2.toarray())


class TestLaplacian:
    """Tests for Laplacian matrix computation."""

    def test_laplacian_unnormalized_basic(self):
        """Test unnormalized Laplacian properties."""
        X = np.array([[0.0], [1.0], [2.0]])
        A = build_graph(X, mode="additive", k=2, seed=0)
        L = laplacian(A, normalized=False)

        assert L.shape == A.shape
        assert issparse(L)

        # Row sums should be zero (within numerical tolerance)
        row_sums = np.array(L.sum(axis=1)).flatten()
        assert np.allclose(row_sums, 0, atol=1e-10)

    def test_laplacian_normalized_basic(self):
        """Test normalized Laplacian properties."""
        X = np.array([[0.0], [1.0], [2.0], [3.0]])
        A = build_graph(X, mode="additive", k=2, seed=0)
        L = laplacian(A, normalized=True)

        assert L.shape == A.shape
        assert issparse(L)

        # Diagonal elements should be close to 1 for connected nodes
        diag = L.diagonal()
        assert all(0 <= d <= 2 for d in diag)

    def test_laplacian_eigenvalues_unnormalized(self):
        """Test that unnormalized Laplacian is PSD."""
        X = np.random.randn(20, 3)
        A = build_graph(X, mode="additive", k=5, seed=0)
        L = laplacian(A, normalized=False)

        # Compute smallest eigenvalues
        evals = eigsh(L, k=5, which='SM', return_eigenvectors=False)

        # All eigenvalues should be non-negative (PSD)
        assert all(e >= -1e-10 for e in evals)  # Numerical tolerance

    def test_laplacian_eigenvalues_normalized(self):
        """Test normalized Laplacian eigenvalue bounds."""
        X = np.random.randn(20, 3)
        A = build_graph(X, mode="additive", k=5, seed=0)
        L = laplacian(A, normalized=True)

        # Compute eigenvalues
        evals = eigsh(L, k=5, which='SM', return_eigenvectors=False)

        # Eigenvalues should be in [0, 2]
        assert all(-1e-10 <= e <= 2 + 1e-10 for e in evals)

    def test_disconnected_nodes(self):
        """Test Laplacian handles disconnected nodes correctly."""
        # Create a graph with an isolated node
        X = np.array([[0.0], [0.1], [10.0]])  # Third point is far
        A = build_graph(X, mode="additive", k=1, sigma=0.01, seed=0)

        # Unnormalized Laplacian
        L_unnorm = laplacian(A, normalized=False)
        row_sums = np.array(L_unnorm.sum(axis=1)).flatten()
        assert np.allclose(row_sums, 0, atol=1e-10)

        # Normalized Laplacian - isolated node should have L[i,i] = 1
        L_norm = laplacian(A, normalized=True)
        # Find isolated nodes (degree = 0)
        degrees = np.array(A.sum(axis=1)).flatten()
        isolated = np.where(degrees < 1e-10)[0]
        if len(isolated) > 0:
            for i in isolated:
                assert np.allclose(L_norm[i, i], 1.0)

    def test_laplacian_consistency(self):
        """Test that L = D - A for unnormalized case."""
        X = np.random.randn(10, 2)
        A = build_graph(X, mode="additive", k=3, seed=0)
        L = laplacian(A, normalized=False)

        # Manually compute D - A
        degrees = np.array(A.sum(axis=1)).flatten()
        D = np.diag(degrees)
        L_manual = D - A.toarray()

        assert np.allclose(L.toarray(), L_manual)


class TestIntegration:
    """Integration tests for the full pipeline."""

    def test_additive_multiplicative_separability(self):
        """Test that additive and multiplicative produce statistically different spectra."""
        np.random.seed(456)
        # Create data with structure that should differ between modes
        X = np.random.lognormal(0, 1, size=(30, 5))

        # Build graphs
        A_add = build_graph(X, mode="additive", k=5, seed=0)
        A_mult = build_graph(X, mode="multiplicative", k=5, seed=0)

        # Compute Laplacians
        L_add = laplacian(A_add, normalized=True)
        L_mult = laplacian(A_mult, normalized=True)

        # Get eigenvalues
        evals_add = eigsh(L_add, k=10, which='SM', return_eigenvectors=False)
        evals_mult = eigsh(L_mult, k=10, which='SM', return_eigenvectors=False)

        # Sort eigenvalues
        evals_add = np.sort(evals_add)
        evals_mult = np.sort(evals_mult)

        # Check that spectra are different
        # Use relative difference for meaningful comparison
        rel_diff = np.abs(evals_add - evals_mult) / (np.abs(evals_add) + 1e-10)
        assert rel_diff.mean() > 0.01  # At least 1% average difference

    def test_pipeline_completeness(self):
        """Test the complete pipeline from data to Laplacian eigenvalues."""
        # Generate synthetic data
        np.random.seed(789)
        X = np.random.randn(50, 10)

        # Build graph
        A = build_graph(X, mode="additive", k=10, sigma="median", seed=0)
        assert A.shape == (50, 50)
        assert A.nnz > 0

        # Compute Laplacian
        L = laplacian(A, normalized=True)
        assert L.shape == (50, 50)

        # Compute spectrum
        k_eigs = min(20, X.shape[0] - 1)
        evals, evecs = eigsh(L, k=k_eigs, which='SM')

        # Verify spectral properties
        assert len(evals) == k_eigs
        assert evecs.shape == (50, k_eigs)
        assert all(0 <= e <= 2 for e in evals)

        # Check orthogonality of eigenvectors
        gram = evecs.T @ evecs
        assert np.allclose(gram, np.eye(k_eigs), atol=1e-8)