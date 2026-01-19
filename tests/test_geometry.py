"""Tests for geometric operations."""

import sys

import numpy as np
import pytest

sys.path.insert(0, "src")

from geometry.fr_pullback import (
    fisher_rao_divergence,
    fisher_rao_metric,
    multinomial_fisher_info,
    rescale_by_metric,
    riemannian_distance,
)
from geometry.projection import (
    check_projection_properties,
    project_to_manifold,
    project_to_tangent,
    project_vector,
    tangent_basis,
)
from geometry.submersion import build_submersion, check_transversal


class TestSubmersion:
    """Tests for submersion construction and transversality."""

    def test_linear_submersion_basic(self):
        """Test basic linear submersion construction."""
        rng = np.random.default_rng(0)
        X = rng.normal(size=(20, 5))

        f, jacobian = build_submersion(X, method="linear", seed=42)

        # Test function output shape
        f_vals = f(X)
        assert f_vals.shape == (20, 2)

        # Test Jacobian output shape
        J_vals = jacobian(X)
        assert J_vals.shape == (20, 2, 5)

        # Test single point
        x_single = X[0]
        f_single = f(x_single)
        J_single = jacobian(x_single)
        assert f_single.shape == (1, 2)
        assert J_single.shape == (1, 2, 5)

    def test_transversality_linear_map(self):
        """Test transversality for linear submersion."""
        rng = np.random.default_rng(0)
        X = rng.normal(size=(20, 5))

        f, jacobian = build_submersion(X, method="linear", seed=0)
        ok, cert = check_transversal((f, jacobian), X)

        # Linear submersion with random vectors should be transversal
        assert ok
        assert cert["min_singular"] > 1e-6
        assert cert["max_condition"] < 1e6
        assert cert["rank_deficient"] == 0

    def test_submersion_independence(self):
        """Test that submersion components are independent."""
        rng = np.random.default_rng(0)
        X = rng.normal(size=(50, 8))

        f, jacobian = build_submersion(X, method="linear", seed=123)
        f_vals = f(X)

        # Components should not be perfectly correlated
        corr = np.corrcoef(f_vals[:, 0], f_vals[:, 1])[0, 1]
        assert abs(corr) < 0.9  # Allow some correlation but not perfect

    def test_jacobian_constant_for_linear(self):
        """Test that Jacobian is constant for linear submersion."""
        rng = np.random.default_rng(0)
        X = rng.normal(size=(10, 4))

        f, jacobian = build_submersion(X, method="linear", seed=0)
        J_vals = jacobian(X)

        # All Jacobians should be the same (constant)
        for i in range(1, len(J_vals)):
            assert np.allclose(J_vals[0], J_vals[i])


class TestProjection:
    """Tests for tangent space projections."""

    def test_projector_properties(self):
        """Test that projection matrix satisfies required properties."""
        # Create a random rank-2 Jacobian
        rng = np.random.default_rng(0)
        J_f = rng.normal(size=(2, 5))

        # Ensure rank 2 by construction
        u1, u2 = rng.normal(size=(2, 5))
        u1 = u1 / np.linalg.norm(u1)
        u2 = u2 - np.dot(u2, u1) * u1
        u2 = u2 / np.linalg.norm(u2)
        J_f = np.vstack([u1, u2])

        P = project_to_tangent(J_f)

        props = check_projection_properties(P)
        assert props["is_symmetric"]
        assert props["is_idempotent"]
        assert props["binary_eigenvalues"]
        assert props["rank"] == 3  # d - 2 for d=5

    def test_projection_methods_agree(self):
        """Test that different projection methods give same result."""
        rng = np.random.default_rng(42)
        J_f = rng.normal(size=(2, 6))

        # Make sure rank is 2
        U, s, Vt = np.linalg.svd(J_f, full_matrices=False)
        s[0] = 1.0
        s[1] = 1.0  # Set to have exactly rank 2
        J_f = U @ np.diag(s) @ Vt

        P_svd = project_to_tangent(J_f, method="svd")
        P_normal = project_to_tangent(J_f, method="normal")

        assert np.allclose(P_svd, P_normal, atol=1e-10)

    def test_tangent_basis_orthogonality(self):
        """Test that tangent basis is orthonormal."""
        rng = np.random.default_rng(0)
        J_f = rng.normal(size=(2, 7))

        # Ensure rank 2
        U, s, Vt = np.linalg.svd(J_f, full_matrices=False)
        s[:2] = [1.0, 1.0]
        J_f = U @ np.diag(s) @ Vt

        basis = tangent_basis(J_f)
        assert basis.shape == (7, 5)  # d=7, rank=5

        # Check orthonormality
        gram = basis.T @ basis
        assert np.allclose(gram, np.eye(5), atol=1e-10)

        # Check it's in tangent space
        assert np.allclose(J_f @ basis, 0, atol=1e-10)

    def test_project_vector(self):
        """Test vector projection onto tangent space."""
        rng = np.random.default_rng(0)
        J_f = rng.normal(size=(2, 4))
        v = rng.normal(size=4)

        v_proj = project_vector(v, J_f)

        # Projected vector should be in tangent space
        assert np.allclose(J_f @ v_proj, 0, atol=1e-10)

        # Projection should preserve tangent components
        P = project_to_tangent(J_f)
        assert np.allclose(v_proj, P @ v)

    def test_rank_deficient_jacobian(self):
        """Test handling of rank-deficient Jacobian."""
        # Create rank-1 Jacobian
        a = np.array([1, 2, 3])
        J_f = np.vstack([a, a])  # Two identical rows

        with pytest.raises(ValueError, match="rank deficient"):
            project_to_tangent(J_f)


class TestFisherRao:
    """Tests for Fisher-Rao metric and pullbacks."""

    def test_multinomial_fisher_info(self):
        """Test Fisher information for multinomial distributions."""
        # Simple probability distributions
        probs = np.array([[0.5, 0.3, 0.2], [0.8, 0.1, 0.1]])

        I = multinomial_fisher_info(probs)
        assert I.shape == (2, 3, 3)

        # Fisher info should be PSD
        for i in range(2):
            evals = np.linalg.eigvals(I[i])
            assert np.all(evals >= -1e-10)

    def test_fisher_rao_metric_basic(self):
        """Test Fisher-Rao metric computation."""
        # Simple logits
        logits = np.array([[1.0, 0.5, -0.5], [0.0, 1.0, 0.0]])

        # Mock Jacobian (logits w.r.t features)
        dlogits_dX = np.array(
            [
                [[1, 0], [0, 1], [1, 1]],  # 3 classes, 2 features
                [[0, 1], [1, 0], [0, 0]],
            ]
        )

        G = fisher_rao_metric(logits, dlogits_dX)
        assert G.shape == (2, 2, 2)

        # Metric should be PSD
        for i in range(2):
            evals = np.linalg.eigvals(G[i])
            assert np.all(evals >= -1e-10)

    def test_fisher_rao_divergence(self):
        """Test Fisher-Rao divergence between distributions."""
        p = np.array([0.5, 0.3, 0.2])
        q = np.array([0.4, 0.4, 0.2])

        div = fisher_rao_divergence(p, q)
        assert div >= 0  # Divergence is non-negative

        # Divergence to self should be 0
        div_self = fisher_rao_divergence(p, p)
        assert div_self < 1e-10

        # Symmetric
        div_sym = fisher_rao_divergence(q, p)
        assert np.isclose(div, div_sym)

    def test_rescale_by_metric(self):
        """Test coordinate rescaling by metric."""
        rng = np.random.default_rng(0)
        X = rng.normal(size=(5, 3))

        # Create simple metric tensors
        G = np.stack([2 * np.eye(3) for _ in range(5)])

        X_rescaled = rescale_by_metric(X, G)
        assert X_rescaled.shape == X.shape

        # With identity metric, should scale by sqrt(2)
        expected_scale = np.sqrt(2)
        for i in range(5):
            ratio = np.linalg.norm(X_rescaled[i]) / np.linalg.norm(X[i])
            assert np.isclose(ratio, expected_scale, rtol=0.1)

    def test_riemannian_distance(self):
        """Test Riemannian distance computation."""
        x = np.array([1.0, 2.0])
        y = np.array([1.5, 2.5])

        # Identity metrics
        G_x = np.eye(2)
        G_y = np.eye(2)

        dist = riemannian_distance(x, y, G_x, G_y)
        euclidean_dist = np.linalg.norm(y - x)
        assert np.isclose(dist, euclidean_dist)

        # Scaled metric
        G_scaled = 4 * np.eye(2)
        dist_scaled = riemannian_distance(x, y, G_scaled, G_scaled)
        assert dist_scaled > dist  # Larger metric gives larger distance


class TestIntegration:
    """Integration tests combining multiple geometry components."""

    def test_submersion_to_projection_pipeline(self):
        """Test complete pipeline from submersion to projection."""
        rng = np.random.default_rng(0)
        X = rng.normal(size=(10, 6))

        # Build submersion
        f, jacobian = build_submersion(X, method="linear", seed=0)

        # Check transversality
        ok, cert = check_transversal((f, jacobian), X)
        assert ok

        # Get Jacobian at first point
        J_0 = jacobian(X[:1])[0]  # Shape (2, 6)

        # Project a random vector
        v = rng.normal(size=6)
        v_proj = project_vector(v, J_0)

        # Verify it's in tangent space
        assert np.allclose(J_0 @ v_proj, 0, atol=1e-10)

    def test_manifold_projection_consistency(self):
        """Test that manifold projection preserves constraints."""
        rng = np.random.default_rng(0)
        X = rng.normal(size=(5, 4))

        # Build submersion
        f, jacobian = build_submersion(X, method="linear", seed=42)

        # Project points to manifold
        X_proj = project_to_manifold(X, f, jacobian, max_iter=20, tol=1e-6)

        # Check constraint satisfaction
        f_vals = f(X_proj)
        violations = np.linalg.norm(f_vals, axis=1)
        assert np.all(violations < 1e-4)  # Should be close to zero
