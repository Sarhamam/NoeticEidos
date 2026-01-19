"""Tests for topology metric seam-compatibility."""

import sys

import numpy as np
import pytest

sys.path.insert(0, "src")

from topology.coords import Strip
from topology.metric import (
    SeamCompatibilityError,
    check_metric_positive_definite,
    enforce_seam_compatibility,
    make_seam_compatible_metric,
    regularize_metric,
    seam_compatible_metric,
    seam_compatible_operator,
    symmetrize_metric,
    validate_component_symmetries,
    validate_metric_grid,
)


class TestSeamCompatibleMetric:
    """Test seam-compatibility checking for metrics."""

    def test_identity_metric_compatible(self):
        """Test that identity metric is seam-compatible."""

        def identity_metric(q):
            return np.eye(2)

        strip = Strip(w=1.0)
        q = np.array([0.5, 0.3])

        assert seam_compatible_metric(identity_metric, q, strip)

    def test_diagonal_metric_compatible(self):
        """Test diagonal metric with even diagonal elements."""

        def diagonal_metric(q):
            u, v = q[0], q[1]
            return np.diag([2.0 + np.cos(2 * u), 1.5 + np.cos(2 * v)])  # Even functions

        strip = Strip(w=1.0)
        q = np.array([0.5, 0.3])

        assert seam_compatible_metric(diagonal_metric, q, strip)

    def test_incompatible_metric(self):
        """Test metric that violates seam-compatibility."""

        def bad_metric(q):
            u, v = q[0], q[1]
            # g12 component depends on u + v (not properly transformed)
            return np.array([[1.0, np.sin(u + v)], [np.sin(u + v), 1.0]])

        strip = Strip(w=1.0)
        q = np.array([0.5, 0.3])

        assert not seam_compatible_metric(bad_metric, q, strip)

    def test_seam_compatible_metric_formula(self):
        """Test metric satisfying seam-compatibility formula explicitly."""

        def metric_fn(q):
            u, v = q[0], q[1]
            # Design to satisfy g(u+π, -v) = dT^T g(u,v) dT
            g11 = 2.0 + np.cos(2 * u) + np.cos(2 * v)  # Even in v
            g22 = 1.5 + np.sin(2 * u) + np.cos(4 * v)  # Even in v
            g12 = 0.1 * np.sin(2 * u) * np.sin(2 * v)  # Odd in v
            return np.array([[g11, g12], [g12, g22]])

        strip = Strip(w=1.0)

        # Test at multiple points
        test_points = [
            np.array([0.0, 0.0]),
            np.array([0.5, 0.3]),
            np.array([np.pi / 2, 0.8]),
            np.array([np.pi, -0.5]),
        ]

        for q in test_points:
            assert seam_compatible_metric(metric_fn, q, strip, tolerance=1e-10)

    def test_tolerance_sensitivity(self):
        """Test sensitivity to tolerance parameter."""

        def almost_compatible_metric(q):
            _u, v = q[0], q[1]
            # Add small violation
            epsilon = 1e-7
            g11 = 1.0 + epsilon * v  # Slight v-dependence
            return np.array([[g11, 0.0], [0.0, 1.0]])

        strip = Strip(w=1.0)
        q = np.array([0.5, 0.8])

        # Should fail with tight tolerance
        assert not seam_compatible_metric(
            almost_compatible_metric, q, strip, tolerance=1e-8
        )

        # Should pass with loose tolerance
        assert seam_compatible_metric(
            almost_compatible_metric, q, strip, tolerance=1e-6
        )

    def test_invalid_input_shapes(self):
        """Test error handling for invalid inputs."""

        def metric_fn(q):
            return np.eye(2)

        strip = Strip(w=1.0)

        # Wrong coordinate shape
        with pytest.raises(ValueError, match="Coordinates must be 2D array"):
            seam_compatible_metric(metric_fn, np.array([1.0]), strip)

        # Wrong metric shape
        def bad_metric_fn(q):
            return np.eye(3)  # Wrong size

        with pytest.raises(ValueError, match="Metric function must return 2×2 matrix"):
            seam_compatible_metric(bad_metric_fn, np.array([0.5, 0.3]), strip)


class TestEnforceSeamCompatibility:
    """Test seam-compatibility enforcement."""

    def test_enforce_compatible_metric(self):
        """Test enforcement passes for compatible metric."""

        def compatible_metric(q):
            return np.eye(2)

        strip = Strip(w=1.0)
        q = np.array([0.5, 0.3])

        # Should not raise
        enforce_seam_compatibility(compatible_metric, q, strip)

    def test_enforce_incompatible_metric(self):
        """Test enforcement raises error for incompatible metric."""

        def incompatible_metric(q):
            u, v = q[0], q[1]
            return np.array([[1.0, u * v], [u * v, 1.0]])  # Violates compatibility

        strip = Strip(w=1.0)
        q = np.array([0.5, 0.3])

        with pytest.raises(
            SeamCompatibilityError, match="Metric violates seam-compatibility"
        ):
            enforce_seam_compatibility(incompatible_metric, q, strip)

    def test_error_message_details(self):
        """Test that error message contains useful details."""

        def bad_metric(q):
            return np.array([[1.0, 1.0], [1.0, 1.0]])  # Constant off-diagonal

        strip = Strip(w=1.0)
        q = np.array([0.5, 0.3])

        with pytest.raises(SeamCompatibilityError) as exc_info:
            enforce_seam_compatibility(bad_metric, q, strip)

        error_msg = str(exc_info.value)
        assert "Max error:" in error_msg
        assert "tolerance:" in error_msg
        assert "dT^T g(u,v) dT" in error_msg


class TestValidateMetricGrid:
    """Test grid-based metric validation."""

    def test_compatible_metric_grid(self):
        """Test grid validation for compatible metric."""

        def compatible_metric(q):
            u, v = q[0], q[1]
            return np.array([[2.0 + np.cos(2 * u), 0.0], [0.0, 1.5 + np.cos(2 * v)]])

        strip = Strip(w=1.0)
        report = validate_metric_grid(compatible_metric, strip, n_u=5, n_v=5)

        assert report["compatible"]
        assert report["violations"] == 0
        assert report["violation_rate"] == 0.0
        assert report["max_error"] == 0.0

    def test_incompatible_metric_grid(self):
        """Test grid validation for incompatible metric."""

        def incompatible_metric(q):
            u, v = q[0], q[1]
            return np.array(
                [[1.0, np.sin(u + v)], [np.sin(u + v), 1.0]]  # Violates compatibility
            )

        strip = Strip(w=1.0)
        report = validate_metric_grid(incompatible_metric, strip, n_u=5, n_v=5)

        assert not report["compatible"]
        assert report["violations"] > 0
        assert report["violation_rate"] > 0
        assert report["max_error"] > 0

    def test_grid_report_structure(self):
        """Test structure of grid validation report."""

        def metric_fn(q):
            return np.eye(2)

        strip = Strip(w=1.0)
        report = validate_metric_grid(metric_fn, strip, n_u=3, n_v=4)

        required_keys = [
            "grid_size",
            "total_points",
            "violations",
            "violation_rate",
            "max_error",
            "mean_error",
            "tolerance",
            "compatible",
            "violation_details",
        ]
        for key in required_keys:
            assert key in report

        assert report["grid_size"] == (3, 4)
        assert report["total_points"] == 12


class TestSeamCompatibleOperator:
    """Test seam-compatibility for operators."""

    def test_identity_operator_compatible(self):
        """Test that identity operator is seam-compatible."""

        def identity_op(q):
            return np.eye(2)

        strip = Strip(w=1.0)
        q = np.array([0.5, 0.3])

        # Identity should satisfy A(T(q)) = dT A(q) dT^T
        assert seam_compatible_operator(identity_op, q, strip)

    def test_diagonal_operator_compatible(self):
        """Test compatible diagonal operator."""

        def diag_op(q):
            u, v = q[0], q[1]
            # Diagonal elements even in v for compatibility
            return np.diag([2.0 + np.cos(2 * v), 1.0 + np.sin(2 * u)])

        strip = Strip(w=1.0)
        q = np.array([0.5, 0.3])

        assert seam_compatible_operator(diag_op, q, strip)

    def test_incompatible_operator(self):
        """Test operator that violates seam-compatibility."""

        def bad_op(q):
            u, v = q[0], q[1]
            # Off-diagonal elements that don't transform correctly
            return np.array([[1.0, np.sin(u)], [np.cos(v), 1.0]])

        strip = Strip(w=1.0)
        q = np.array([0.5, 0.3])

        assert not seam_compatible_operator(bad_op, q, strip)


class TestMetricUtilities:
    """Test metric utility functions."""

    def test_symmetrize_metric(self):
        """Test metric symmetrization."""
        g = np.array([[1.0, 0.5], [0.3, 2.0]])  # Asymmetric
        g_sym = symmetrize_metric(g)

        expected = np.array([[1.0, 0.4], [0.4, 2.0]])
        assert np.allclose(g_sym, expected)
        assert np.allclose(g_sym, g_sym.T)  # Check symmetry

    def test_check_positive_definite(self):
        """Test positive definiteness checking."""
        # Positive definite matrix
        g_pd = np.array([[2.0, 0.5], [0.5, 1.0]])
        assert check_metric_positive_definite(g_pd)

        # Not positive definite
        g_not_pd = np.array([[1.0, 2.0], [2.0, 1.0]])  # Negative eigenvalue
        assert not check_metric_positive_definite(g_not_pd)

        # Boundary case
        g_boundary = np.array([[1.0, 0.0], [0.0, 1e-10]])
        assert check_metric_positive_definite(g_boundary, min_eigenvalue=1e-11)
        assert not check_metric_positive_definite(g_boundary, min_eigenvalue=1e-9)

    def test_regularize_metric(self):
        """Test metric regularization."""
        g = np.array([[1.0, 0.5], [0.3, 0.5]])  # Nearly singular

        g_reg = regularize_metric(g, regularization=0.1)

        # Should be positive definite after regularization
        assert check_metric_positive_definite(g_reg)

        # Check that regularization was added to diagonal
        expected_diag = np.diag(g) + 0.1
        assert np.allclose(np.diag(g_reg), expected_diag)


class TestMakeSeamCompatibleMetric:
    """Test construction of seam-compatible metrics."""

    def test_construct_compatible_metric(self):
        """Test construction from component functions."""

        def g11_fn(q):
            u, v = q[0], q[1]
            return 2.0 + np.cos(2 * u) + np.cos(2 * v)  # Even in v

        def g22_fn(q):
            u, v = q[0], q[1]
            return 1.5 + np.sin(2 * u) + np.cos(4 * v)  # Even in v

        def g12_fn(q):
            u, v = q[0], q[1]
            return 0.1 * np.sin(2 * u) * np.sin(2 * v)  # Odd in v

        metric_fn = make_seam_compatible_metric(g11_fn, g22_fn, g12_fn)

        strip = Strip(w=1.0)
        q = np.array([0.5, 0.3])

        # Test that constructed metric is seam-compatible
        assert seam_compatible_metric(metric_fn, q, strip)

        # Test metric structure
        g = metric_fn(q)
        assert g.shape == (2, 2)
        assert np.isclose(g[0, 0], g11_fn(q))
        assert np.isclose(g[1, 1], g22_fn(q))
        assert np.isclose(g[0, 1], g12_fn(q))
        assert np.isclose(g[1, 0], g12_fn(q))

    def test_construct_diagonal_metric(self):
        """Test construction with only diagonal components."""

        def g11_fn(q):
            return 2.0

        def g22_fn(q):
            return 1.5

        metric_fn = make_seam_compatible_metric(g11_fn, g22_fn)  # No g12

        q = np.array([0.5, 0.3])
        g = metric_fn(q)

        expected = np.array([[2.0, 0.0], [0.0, 1.5]])
        assert np.allclose(g, expected)


class TestValidateComponentSymmetries:
    """Test validation of metric component symmetries."""

    def test_correct_symmetries(self):
        """Test components with correct symmetries."""

        def g11_fn(q):
            _u, v = q[0], q[1]
            return 2.0 + np.cos(2 * v)  # Even in v

        def g22_fn(q):
            u, _v = q[0], q[1]
            return 1.5 + np.cos(2 * u)  # Periodic in u

        def g12_fn(q):
            _u, v = q[0], q[1]
            return 0.1 * np.sin(2 * v)  # Odd in v

        strip = Strip(w=1.0)
        results = validate_component_symmetries(g11_fn, g22_fn, g12_fn, strip)

        assert results["g11_even_in_v"]
        assert results["g22_even_in_v"]
        assert results["g12_odd_in_v"]

    def test_incorrect_symmetries(self):
        """Test components with incorrect symmetries."""

        def g11_fn(q):
            _u, v = q[0], q[1]
            return 2.0 + v  # Linear in v (not even)

        def g22_fn(q):
            _u, _v = q[0], q[1]
            return 1.5

        def g12_fn(q):
            _u, v = q[0], q[1]
            return 0.1 * np.cos(v)  # Even in v (should be odd)

        strip = Strip(w=1.0)
        results = validate_component_symmetries(g11_fn, g22_fn, g12_fn, strip)

        assert not results["g11_even_in_v"]
        assert not results["g12_odd_in_v"]

    def test_periodicity_validation(self):
        """Test periodicity requirements."""

        def g11_fn(q):
            u, _v = q[0], q[1]
            return 2.0 + np.sin(u)  # Not π-periodic

        def g22_fn(q):
            return 1.5

        def g12_fn(q):
            return 0.0

        strip = Strip(w=1.0)
        results = validate_component_symmetries(g11_fn, g22_fn, g12_fn, strip)

        # sin(u) is not π-periodic, so should fail
        assert not results["g11_periodic_in_u"]


if __name__ == "__main__":
    pytest.main([__file__])
