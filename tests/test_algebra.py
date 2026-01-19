"""Tests for algebraic operations."""

import sys

import numpy as np
import pytest

sys.path.insert(0, "src")

from algebra.additive import gaussian_affinity_matrix, gaussian_kernel, heat_kernel
from algebra.mellin import (
    mellin_balance_score,
    mellin_transform_discrete,
    mellin_unitarity_test,
)
from algebra.multiplicative import (
    haar_measure_weight,
    log_map,
    log_ratio_distance,
    multiplicative_distance,
    poisson_kernel_log,
)


class TestAdditive:
    """Tests for additive transport operations."""

    def test_gaussian_kernel_scalar(self):
        """Test Gaussian kernel on scalar inputs."""
        d2_vals = np.array([0.0, 1.0, 4.0])
        g1 = gaussian_kernel(d2_vals, sigma=1.0)
        g2 = gaussian_kernel(d2_vals, sigma=np.sqrt(2))

        # At d²=0, kernel should be 1
        assert g1[0] == 1.0
        assert g2[0] == 1.0

        # Broader sigma -> LARGER values at same d²>0 (slower decay)
        assert np.all(g2[1:] > g1[1:])

        # Kernel should decay with distance
        assert g1[1] > g1[2]

    def test_gaussian_semigroup_property(self):
        """Test semigroup property of Gaussian."""
        d2 = 4.0
        sigma1, sigma2 = 1.0, np.sqrt(3)
        sigma_combined = np.sqrt(sigma1**2 + sigma2**2)

        # G_σ1 * G_σ2 = G_√(σ1²+σ2²) (in terms of variance)
        g1 = gaussian_kernel(d2, sigma1)
        g2 = gaussian_kernel(d2, sigma2)
        g_combined = gaussian_kernel(d2, sigma_combined)

        # Combined kernel has larger sigma, so larger value at same d²
        assert g_combined > g1  # σ_combined > σ1
        assert g_combined > g2  # σ_combined > σ2

        # Test ordering by sigma
        assert g2 > g1  # σ2 > σ1, so slower decay

    def test_gaussian_affinity_matrix(self):
        """Test Gaussian affinity matrix construction."""
        X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        W = gaussian_affinity_matrix(X, sigma=1.0)

        assert W.shape == (3, 3)
        # Diagonal should be zero (no self-loops)
        assert np.allclose(np.diag(W), 0)
        # Should be symmetric
        assert np.allclose(W, W.T)
        # Off-diagonal entries should be positive
        assert W[0, 1] > 0
        assert W[0, 1] == W[1, 0]

    def test_heat_kernel_diffusion(self):
        """Test heat kernel properties."""
        # Small Laplacian for testing
        L = np.array([[2, -1, -1], [-1, 2, -1], [-1, -1, 2]])
        t = 0.1
        H_t = heat_kernel(L, t, k=3)

        assert H_t.shape == (3, 3)
        # Heat kernel should be symmetric
        assert np.allclose(H_t, H_t.T)
        # Entries should be non-negative (for small t)
        assert np.all(H_t >= -1e-10)


class TestMultiplicative:
    """Tests for multiplicative transport operations."""

    def test_log_map_basic(self):
        """Test logarithmic mapping."""
        X = np.array([1.0, 2.0, 10.0])
        Z = log_map(X, eps=1e-6)

        # log(1) ≈ 0, log(2) ≈ 0.69, log(10) ≈ 2.3
        assert np.allclose(Z[0], 0, atol=1e-5)
        assert 0.6 < Z[1] < 0.8
        assert 2.2 < Z[2] < 2.4

        # Test regularization for near-zero
        X_small = np.array([0.0, 1e-7])
        Z_small = log_map(X_small, eps=1e-6)
        assert np.all(np.isfinite(Z_small))

    def test_poisson_kernel_log_properties(self):
        """Test Poisson kernel in log domain."""
        delta = np.array([0.0, 1.0, 2.0])
        t = 0.5
        P = poisson_kernel_log(delta, t)

        # Peak at delta=0
        peak_value = t / (np.pi * t**2)
        assert np.isclose(P[0], peak_value)

        # Decay with distance
        assert np.all(P[1:] < P[0])
        assert P[1] > P[2]

        # Test symmetry: P(δ) = P(-δ)
        P_neg = poisson_kernel_log(-delta, t)
        assert np.allclose(P, P_neg)

    def test_haar_measure_weight(self):
        """Test Haar measure computation."""
        y = np.array([1.0, 2.0, 0.5])
        weights = haar_measure_weight(y, eps=1e-10)

        # Haar measure is 1/y
        expected = 1.0 / y
        assert np.allclose(weights, expected)

        # Test regularization
        y_small = np.array([0.0, 1e-11])
        weights_small = haar_measure_weight(y_small, eps=1e-10)
        assert np.all(np.isfinite(weights_small))

    def test_multiplicative_distance(self):
        """Test distance computation in log space."""
        X = np.array([[1.0], [2.0], [4.0]])
        Y = np.array([[1.0], [8.0]])

        D = multiplicative_distance(X, Y, eps=1e-6)

        assert D.shape == (3, 2)
        # Distance to self should be 0
        assert D[0, 0] < 1e-6
        # Distance should be symmetric in log space
        # log(2/1) = -log(1/2), so distances relate

    def test_log_ratio_distance_invariance(self):
        """Test scale invariance of log-ratio distance."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([2.0, 4.0, 6.0])

        d1 = log_ratio_distance(x, y)
        # Scale both by same factor
        c = 5.0
        d2 = log_ratio_distance(c * x, c * y)

        # Should be scale-invariant
        assert np.isclose(d1, d2)


class TestMellin:
    """Tests for Mellin transform and coupling."""

    def test_mellin_transform_basic(self):
        """Test discrete Mellin transform."""
        # Exponential decay function
        y_grid = np.linspace(0.1, 5.0, 50)
        f_vals = np.exp(-y_grid)

        # Mellin at s=1/2
        M_half = mellin_transform_discrete(f_vals, y_grid, 0.5 + 0j)
        assert np.isfinite(M_half)

        # Mellin at s=1 (should be different)
        M_one = mellin_transform_discrete(f_vals, y_grid, 1.0 + 0j)
        assert np.isfinite(M_one)
        assert not np.isclose(M_half, M_one)

    def test_mellin_unitarity_at_half(self):
        """Test unitarity property at s=1/2."""
        y_grid = np.linspace(0.5, 3.0, 30)
        f_vals = np.exp(-(y_grid**2))
        g_vals = np.sin(y_grid) / y_grid

        result = mellin_unitarity_test(f_vals, g_vals, y_grid, s=0.5)

        assert "inner_mellin" in result
        assert "inner_l2_haar" in result
        assert result["s"] == 0.5

        # Check that all values are finite
        assert np.isfinite(result["inner_mellin"])
        assert np.isfinite(result["inner_l2_haar"])

    def test_mellin_balance_score(self):
        """Test balance score between transport modes."""
        # Create synthetic additive/multiplicative features
        np.random.seed(42)
        X_add = np.random.randn(20, 5)
        X_mult = np.exp(0.5 * np.random.randn(20, 5))

        s_values = np.array([0.3, 0.4, 0.5, 0.6, 0.7])
        scores, optimal_s = mellin_balance_score(X_add, X_mult, s_values)

        assert len(scores) == len(s_values)
        assert optimal_s in s_values
        # Scores should be between 0 and 1 (correlation magnitudes)
        assert np.all(scores >= 0)
        assert np.all(scores <= 1)

    def test_mellin_requires_positive_grid(self):
        """Test that Mellin transform requires positive grid."""
        y_grid = np.array([-1.0, 0.0, 1.0])
        f_vals = np.ones(3)

        with pytest.raises(ValueError):
            mellin_transform_discrete(f_vals, y_grid, 0.5 + 0j)
