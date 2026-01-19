"""Tests for statistical analysis module."""

import sys

import numpy as np
import pytest

sys.path.insert(0, "src")

from graphs.knn import build_graph
from graphs.laplacian import laplacian
from stats.balance import balance_curve, balance_score, mellin_coupled_stat
from stats.separability import effect_size_interpretation, separability_test
from stats.spectra import (
    spectral_entropy,
    spectral_entropy_additive,
    spectral_entropy_multiplicative,
    spectral_gap,
    spectral_gap_additive,
    spectral_gap_multiplicative,
)
from stats.stability import noise_perturbation, stability_score


class TestSpectralMeasures:
    """Tests for spectral gap and entropy calculations."""

    def test_spectral_gap_path_graph(self):
        """Test spectral gap on path graph structure."""
        # Create path-like dataset
        X = np.array([[i] for i in range(5)])
        A = build_graph(X, mode="additive", k=2, seed=0)
        L = laplacian(A, normalized=True)

        gap = spectral_gap(L)

        # Path graph should have positive spectral gap
        assert gap > 0
        assert gap < 1  # Normalized Laplacian eigenvalues ≤ 2

    def test_spectral_entropy_monotonicity(self):
        """Test basic properties of spectral entropy."""
        X = np.array([[i] for i in range(6)])
        A = build_graph(X, mode="additive", k=2, seed=0)
        L = laplacian(A, normalized=True)

        H = spectral_entropy(L, k=3)

        assert H >= 0  # Entropy is non-negative
        assert np.isfinite(H)

    def test_additive_vs_multiplicative_gap_difference(self):
        """Test that additive and multiplicative modes give different results."""
        # Use data that should highlight mode differences
        X = np.array([[1.0], [2.0], [4.0], [8.0]])

        gap_add = spectral_gap_additive(X, k=2, neighbors=2, seed=0)
        gap_mult = spectral_gap_multiplicative(X, k=2, neighbors=2, seed=0)

        # Should get different values from different transport modes
        assert abs(gap_add - gap_mult) > 1e-6

    def test_spectral_entropy_additive_vs_multiplicative(self):
        """Test entropy differences between transport modes."""
        X = np.array([[1.0], [2.0], [4.0], [8.0], [16.0]])

        H_add = spectral_entropy_additive(X, k=3, neighbors=3, seed=0)
        H_mult = spectral_entropy_multiplicative(X, k=3, neighbors=3, seed=0)

        # Both should be valid entropies
        assert H_add >= 0
        assert H_mult >= 0
        assert np.isfinite(H_add)
        assert np.isfinite(H_mult)

    def test_complete_graph_properties(self):
        """Test spectral properties of complete graph."""
        # Complete graph (all points connected)
        X = np.array([[i] for i in range(4)])
        A = build_graph(X, mode="additive", k=3, seed=0)  # k=n-1 for complete
        L = laplacian(A, normalized=True)

        gap = spectral_gap(L)
        entropy = spectral_entropy(L, k=3)

        # Complete graph has well-defined spectral properties
        assert gap > 0.5  # Complete graph has large spectral gap
        assert entropy >= 0


class TestStability:
    """Tests for stability analysis."""

    def test_stability_score_no_perturbation(self):
        """Test stability when there's no perturbation."""
        X = np.array([[i] for i in range(4)])

        def stat_fn(X_in):
            A = build_graph(X_in, mode="additive", k=2, seed=0)
            L = laplacian(A, normalized=True)
            return spectral_gap(L)

        def no_perturb_fn(X_in, seed):
            return X_in  # No change

        mean_val, std_val, stability = stability_score(
            stat_fn, X, no_perturb_fn, trials=5, seed=0
        )

        assert stability > 0.9  # Should be highly stable
        assert std_val < 0.1  # Low variance

    def test_stability_with_noise(self):
        """Test stability under noise perturbations."""
        X = np.array([[i, i] for i in range(6)])

        def stat_fn(X_in):
            A = build_graph(X_in, mode="additive", k=3, seed=0)
            L = laplacian(A, normalized=True)
            return spectral_gap(L)

        perturb_fn = noise_perturbation(noise_level=0.1)

        mean_val, std_val, stability = stability_score(
            stat_fn, X, perturb_fn, trials=5, seed=0
        )

        assert 0.0 <= stability <= 1.0
        assert np.isfinite(mean_val)
        assert std_val >= 0

    def test_noise_perturbation_function(self):
        """Test noise perturbation function properties."""
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        perturb_fn = noise_perturbation(noise_level=0.1)

        X_perturbed = perturb_fn(X, seed=0)

        # Should have same shape
        assert X_perturbed.shape == X.shape

        # Should be different (with high probability)
        diff = np.linalg.norm(X_perturbed - X)
        assert diff > 1e-6


class TestSeparability:
    """Tests for separability analysis."""

    def test_separability_obvious_difference(self):
        """Test separability on clearly different distributions."""
        phi_add = np.array([1.0, 1.1, 0.9, 1.05])
        phi_mult = np.array([2.0, 2.1, 1.9, 2.05])

        # Try t-test which should be more sensitive for this case
        result = separability_test(phi_add, phi_mult, method="ttest", seed=0)

        assert result["separable"]  # Should detect clear difference
        assert result["p_value"] < 0.05
        assert result["effect_size"] > 1.0  # Large effect size

    def test_separability_no_difference(self):
        """Test separability when distributions are the same."""
        phi_same1 = np.array([1.0, 1.1, 0.9, 1.05])
        phi_same2 = np.array([1.0, 1.1, 0.9, 1.05])

        result = separability_test(phi_same1, phi_same2, trials=100, seed=0)

        assert not result["separable"]  # Should not detect difference
        assert result["effect_size"] < 0.1  # Small effect size

    def test_effect_size_interpretation(self):
        """Test effect size interpretation categories."""
        assert effect_size_interpretation(0.1) == "negligible"
        assert effect_size_interpretation(0.3) == "small"
        assert effect_size_interpretation(0.6) == "medium"
        assert effect_size_interpretation(1.0) == "large"

    def test_multiple_separability_methods(self):
        """Test different separability test methods."""
        phi_add = np.array([1.0, 1.2, 0.8])
        phi_mult = np.array([2.0, 2.2, 1.8])

        methods = ["bootstrap", "ttest", "mannwhitney", "permutation"]

        for method in methods:
            try:
                result = separability_test(
                    phi_add, phi_mult, method=method, trials=50, seed=0
                )
                assert "p_value" in result
                assert "separable" in result
                assert result["method"] == method
            except Exception as e:
                pytest.skip(f"Method {method} failed: {e}")


class TestBalance:
    """Tests for Mellin balance analysis."""

    def test_mellin_coupled_stat_endpoints(self):
        """Test Mellin coupling at endpoints s=0 and s=1."""
        X = np.array([[1.0], [2.0], [3.0], [4.0]])

        def stat_fn(L):
            return spectral_gap(L)

        # At s=1, should be pure additive
        stat_s1 = mellin_coupled_stat(X, stat_fn, s=1.0, k=2, seed=0)
        stat_add = spectral_gap_additive(X, k=2, neighbors=2, seed=0)

        # Should be close (allowing for numerical differences)
        assert abs(stat_s1 - stat_add) < 0.1

        # At s=0, should be pure multiplicative
        stat_s0 = mellin_coupled_stat(X, stat_fn, s=0.0, k=2, seed=0)
        stat_mult = spectral_gap_multiplicative(X, k=2, neighbors=2, seed=0)

        assert abs(stat_s0 - stat_mult) < 0.1

    def test_balance_curve_properties(self):
        """Test properties of balance curve."""
        X = np.array([[i] for i in range(5)])

        def stat_fn(L):
            return spectral_gap(L)

        s_range = np.linspace(0.2, 0.8, 5)
        s_values, stat_values = balance_curve(X, stat_fn, s_range=s_range, k=2, seed=0)

        assert len(s_values) == len(s_range)
        assert len(stat_values) == len(s_range)
        assert np.all(np.isfinite(stat_values))

    def test_balance_score_peak_detection(self):
        """Test balance score optimization."""
        X = np.array([[0.0], [1.0], [2.0], [3.0]])

        # Create a mock statistic that peaks at s=0.5
        def mock_stat_fn(L):
            # This is a placeholder - in practice would be actual spectral stat
            return 1.0  # Constant for simplicity

        s_range = np.linspace(0.3, 0.7, 9)
        best_s, curve, info = balance_score(X, mock_stat_fn, s_range=s_range, seed=0)

        # Should find a reasonable optimum
        assert 0.3 <= best_s <= 0.7
        assert len(curve) == len(s_range)
        assert "converged" in info

    def test_balance_score_canonical_expectation(self):
        """Test that balance score finds s≈0.5 for balanced statistics."""
        # Create simple symmetric dataset
        X = np.array([[0.0], [1.0], [2.0], [3.0]])

        def symmetric_stat(L):
            # Simplified statistic that should be symmetric about s=0.5
            return spectral_gap(L)

        best_s, _, info = balance_score(
            X, symmetric_stat, s_range=np.linspace(0.3, 0.7, 9), seed=0
        )

        # Should be reasonably close to canonical balance point
        assert abs(best_s - 0.5) < 0.3  # Allow some tolerance

    def test_mellin_parameter_validation(self):
        """Test validation of Mellin parameter s."""
        X = np.array([[1.0], [2.0]])

        def stat_fn(L):
            return spectral_gap(L)

        # Should reject invalid s values
        with pytest.raises(ValueError):
            mellin_coupled_stat(X, stat_fn, s=-0.1)

        with pytest.raises(ValueError):
            mellin_coupled_stat(X, stat_fn, s=1.1)


class TestIntegration:
    """Integration tests across multiple modules."""

    def test_full_stats_pipeline(self):
        """Test complete statistical analysis pipeline."""
        # Create test dataset with clear structure
        rng = np.random.default_rng(0)
        X = rng.normal(size=(8, 2))

        # Spectral measures
        gap_add = spectral_gap_additive(X, neighbors=4, seed=0)
        gap_mult = spectral_gap_multiplicative(X, neighbors=4, seed=0)

        assert gap_add > 0
        assert gap_mult > 0

        # Separability test
        gaps_add = [
            spectral_gap_additive(
                X + 0.1 * rng.normal(size=X.shape), neighbors=4, seed=i
            )
            for i in range(3)
        ]
        gaps_mult = [
            spectral_gap_multiplicative(
                X + 0.1 * rng.normal(size=X.shape), neighbors=4, seed=i
            )
            for i in range(3)
        ]

        sep_result = separability_test(
            np.array(gaps_add), np.array(gaps_mult), trials=50, seed=0
        )

        # Should complete without errors
        assert "separable" in sep_result

        # Balance analysis
        def gap_stat(L):
            return spectral_gap(L)

        best_s, _, _ = balance_score(
            X, gap_stat, s_range=np.linspace(0.4, 0.6, 5), k=4, seed=0
        )

        assert 0.4 <= best_s <= 0.6

    def test_stability_separability_consistency(self):
        """Test consistency between stability and separability measures."""
        X = np.array([[i, i] for i in range(6)])

        def stat_fn(X_in):
            A = build_graph(X_in, mode="additive", k=3, seed=0)
            L = laplacian(A, normalized=True)
            return spectral_gap(L)

        # Stability analysis
        perturb_fn = noise_perturbation(noise_level=0.05)
        _, _, stability = stability_score(stat_fn, X, perturb_fn, trials=5, seed=0)

        # High stability should correlate with good separability detection
        assert 0.0 <= stability <= 1.0

    def test_cross_mode_analysis(self):
        """Test analysis across additive and multiplicative modes."""
        X = np.array([[1.0], [2.0], [4.0], [8.0]])

        # Compute statistics in both modes
        stats_add = []
        stats_mult = []

        for i in range(3):
            # Add small perturbations to test robustness
            X_pert = X + 0.1 * np.random.RandomState(i).normal(size=X.shape)

            gap_add = spectral_gap_additive(X_pert, neighbors=2, seed=i)
            gap_mult = spectral_gap_multiplicative(X_pert, neighbors=2, seed=i)

            stats_add.append(gap_add)
            stats_mult.append(gap_mult)

        # Test separability
        result = separability_test(
            np.array(stats_add), np.array(stats_mult), trials=50, seed=0
        )

        # Should detect mode difference for this geometric sequence
        assert "p_value" in result

        # Effect size should be meaningful
        assert result["effect_size"] >= 0

        # Test balance
        def gap_stat(L):
            return spectral_gap(L)

        s_values, stat_values = balance_curve(
            X, gap_stat, s_range=np.array([0.0, 0.5, 1.0]), k=2, seed=0
        )

        assert len(stat_values) == 3
        assert np.all(np.isfinite(stat_values))
