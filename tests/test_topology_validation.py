"""Tests for topology validation utilities."""

import sys

import numpy as np
import pytest

sys.path.insert(0, "src")

from topology.coords import Strip
from topology.validation import (
    TopologicalValidationError,
    comprehensive_topology_validation,
    seam_invariance,
    seam_invariance_grid,
    validate_geodesic_invariance,
    validate_metric_invariance,
    validate_operator_invariance,
    validate_spectral_invariance,
)


class TestSeamInvariance:
    """Test seam invariance checking for functions."""

    def test_invariant_constant_function(self):
        """Test that constant function is invariant."""

        def constant_fn(q):
            return 5.0

        strip = Strip(w=1.0)
        q = np.array([0.5, 0.3])

        assert seam_invariance(constant_fn, q, strip)

    def test_invariant_even_function(self):
        """Test function that is truly invariant under deck map."""

        def invariant_fn(q):
            u, v = q[0], q[1]
            # Function that's even in v and has period π in u
            return np.cos(2 * u) + np.cos(
                2 * v
            )  # cos(2(u+π)) = cos(2u), cos(2(-v)) = cos(2v)

        strip = Strip(w=1.0)
        q = np.array([0.5, 0.3])

        assert seam_invariance(invariant_fn, q, strip)

    def test_non_invariant_function(self):
        """Test function that violates invariance."""

        def non_invariant_fn(q):
            u, v = q[0], q[1]
            return u + v  # Changes under deck map

        strip = Strip(w=1.0)
        q = np.array([0.5, 0.3])

        assert not seam_invariance(non_invariant_fn, q, strip)

    def test_invariant_vector_function(self):
        """Test vector-valued invariant function."""

        def vector_fn(q):
            u, v = q[0], q[1]
            return np.array([np.cos(2 * u), np.cos(2 * v)])  # Both even/periodic

        strip = Strip(w=1.0)
        q = np.array([0.5, 0.3])

        assert seam_invariance(vector_fn, q, strip)

    def test_non_invariant_vector_function(self):
        """Test vector-valued function that violates invariance."""

        def bad_vector_fn(q):
            u, v = q[0], q[1]
            return np.array([u, v])  # Not invariant

        strip = Strip(w=1.0)
        q = np.array([0.5, 0.3])

        assert not seam_invariance(bad_vector_fn, q, strip)

    def test_tolerance_sensitivity(self):
        """Test sensitivity to tolerance parameter."""

        def almost_invariant_fn(q):
            u, v = q[0], q[1]
            # Base invariant function
            base_value = np.cos(2 * u) + np.cos(2 * v)
            # Add small coordinate-dependent error
            error = 1e-9 * (u + v)  # Small violation
            return base_value + error

        strip = Strip(w=1.0)
        q = np.array([0.5, 0.3])

        # Should fail with tight tolerance (error ~1e-9 * 0.8 = 8e-10)
        assert not seam_invariance(almost_invariant_fn, q, strip, tolerance=1e-10)

        # Should pass with loose tolerance
        assert seam_invariance(almost_invariant_fn, q, strip, tolerance=1e-8)

    def test_function_evaluation_error(self):
        """Test error handling when function evaluation fails."""

        def failing_fn(q):
            raise ValueError("Function evaluation failed")

        strip = Strip(w=1.0)
        q = np.array([0.5, 0.3])

        with pytest.raises(
            TopologicalValidationError, match="Function evaluation failed"
        ):
            seam_invariance(failing_fn, q, strip)

    def test_invalid_coordinate_shape(self):
        """Test error for invalid coordinate shape."""

        def dummy_fn(q):
            return 1.0

        strip = Strip(w=1.0)

        with pytest.raises(ValueError, match="Coordinates must be 2D array"):
            seam_invariance(dummy_fn, np.array([1.0]), strip)

    def test_mismatched_return_shapes(self):
        """Test error when function returns different shapes at equivalent points."""

        def shape_changing_fn(q):
            u, _v = q[0], q[1]
            if u < 1.0:
                return np.array([1.0, 2.0])
            else:
                return np.array([1.0, 2.0, 3.0])  # Different shape

        strip = Strip(w=1.0)
        q = np.array([0.5, 0.3])  # Will map to u > π

        with pytest.raises(TopologicalValidationError, match="different shapes"):
            seam_invariance(shape_changing_fn, q, strip)


class TestSeamInvarianceGrid:
    """Test grid-based seam invariance validation."""

    def test_invariant_function_grid(self):
        """Test grid validation for invariant function."""

        def invariant_fn(q):
            u, v = q[0], q[1]
            return 2.0 + np.cos(2 * u) + np.cos(2 * v)

        strip = Strip(w=1.0)
        u_vals = np.linspace(0, 2 * np.pi, 5, endpoint=False)
        v_vals = np.linspace(-1.0, 1.0, 4)
        U, V = np.meshgrid(u_vals, v_vals)

        report = seam_invariance_grid(invariant_fn, U, V, strip)

        assert report["invariant"]
        assert report["violations"] == 0
        assert report["violation_rate"] == 0.0
        assert report["max_error"] == 0.0

    def test_non_invariant_function_grid(self):
        """Test grid validation for non-invariant function."""

        def non_invariant_fn(q):
            u, v = q[0], q[1]
            return u * v  # Violates invariance

        strip = Strip(w=1.0)
        u_vals = np.linspace(0, 2 * np.pi, 4, endpoint=False)
        v_vals = np.linspace(-1.0, 1.0, 3)
        U, V = np.meshgrid(u_vals, v_vals)

        report = seam_invariance_grid(non_invariant_fn, U, V, strip)

        assert not report["invariant"]
        assert report["violations"] > 0
        assert report["violation_rate"] > 0
        assert report["max_error"] > 0

    def test_grid_report_structure(self):
        """Test structure of grid validation report."""

        def dummy_fn(q):
            return 1.0

        strip = Strip(w=1.0)
        U = np.array([[0.0, 1.0], [2.0, 3.0]])
        V = np.array([[0.0, 0.5], [-0.5, -1.0]])

        report = seam_invariance_grid(dummy_fn, U, V, strip)

        required_keys = [
            "grid_shape",
            "total_points",
            "violations",
            "failed_evaluations",
            "violation_rate",
            "failure_rate",
            "max_error",
            "mean_error",
            "std_error",
            "tolerance",
            "invariant",
            "violation_details",
            "failure_details",
        ]

        for key in required_keys:
            assert key in report

        assert report["grid_shape"] == (2, 2)
        assert report["total_points"] == 4

    def test_function_failures_in_grid(self):
        """Test handling of function evaluation failures in grid."""

        def sometimes_failing_fn(q):
            u, v = q[0], q[1]
            if u > 1.0:
                raise ValueError("Function failed")
            return np.cos(v)

        strip = Strip(w=1.0)
        U = np.array([[0.5, 1.5], [0.3, 2.0]])  # Some values > 1.0
        V = np.array([[0.0, 0.5], [0.2, 0.8]])

        report = seam_invariance_grid(sometimes_failing_fn, U, V, strip)

        assert report["failed_evaluations"] > 0
        assert report["failure_rate"] > 0
        assert not report["invariant"]  # Failures count as non-invariant

    def test_mismatched_grid_shapes(self):
        """Test error for mismatched grid shapes."""

        def dummy_fn(q):
            return 1.0

        strip = Strip(w=1.0)
        U = np.array([[0.0, 1.0]])
        V = np.array([[0.0], [1.0]])  # Different shape

        with pytest.raises(ValueError, match="U and V grids must have same shape"):
            seam_invariance_grid(dummy_fn, U, V, strip)


class TestValidateMetricInvariance:
    """Test metric invariance validation."""

    def test_constant_metric_invariance(self):
        """Test validation for constant (invariant) metric."""

        def constant_metric(q):
            return np.array([[2.0, 0.5], [0.5, 1.5]])

        strip = Strip(w=1.0)
        report = validate_metric_invariance(constant_metric, strip)

        assert report["invariant"]
        assert report["violations"] == 0
        assert report["max_error"] == 0.0

    def test_variable_metric_non_invariance(self):
        """Test validation for variable (non-invariant) metric."""

        def variable_metric(q):
            u, v = q[0], q[1]
            return np.array([[1.0 + u, 0.0], [0.0, 1.0 + v]])

        strip = Strip(w=1.0)
        report = validate_metric_invariance(variable_metric, strip)

        assert not report["invariant"]
        assert report["violations"] > 0
        assert report["max_error"] > 0

    def test_metric_invariance_report_structure(self):
        """Test structure of metric invariance report."""

        def metric_fn(q):
            return np.eye(2)

        strip = Strip(w=1.0)
        report = validate_metric_invariance(metric_fn, strip, n_test=10)

        required_keys = [
            "test_type",
            "n_test",
            "violations",
            "violation_rate",
            "max_error",
            "mean_error",
            "tolerance",
            "invariant",
            "violation_details",
        ]

        for key in required_keys:
            assert key in report

        assert report["test_type"] == "metric_invariance"
        assert report["n_test"] == 10


class TestValidateSpectralInvariance:
    """Test spectral statistics invariance validation."""

    def test_constant_spectrum_invariance(self):
        """Test invariant spectral function."""

        def constant_spectrum(q):
            return np.array([1.0, 2.0, 3.0])  # Constant spectrum

        strip = Strip(w=1.0)
        report = validate_spectral_invariance(constant_spectrum, strip)

        assert report["invariant"]
        assert report["violations"] == 0

    def test_variable_spectrum_non_invariance(self):
        """Test non-invariant spectral function."""

        def variable_spectrum(q):
            u, v = q[0], q[1]
            return np.array([1.0 + u, 2.0 + v, 3.0])

        strip = Strip(w=1.0)
        report = validate_spectral_invariance(variable_spectrum, strip)

        assert not report["invariant"]
        assert report["violations"] > 0

    def test_spectral_report_structure(self):
        """Test spectral validation report structure."""

        def spectrum_fn(q):
            return np.array([1.0, 2.0])

        strip = Strip(w=1.0)
        report = validate_spectral_invariance(spectrum_fn, strip, n_test=5)

        assert report["test_type"] == "spectral_invariance"
        assert "grid_shape" in report


class TestValidateOperatorInvariance:
    """Test operator seam-compatibility validation."""

    def test_identity_operator_compatible(self):
        """Test that identity operator is seam-compatible."""

        def identity_op(q):
            return np.eye(2)

        strip = Strip(w=1.0)
        report = validate_operator_invariance(identity_op, strip)

        assert report["seam_compatible"]
        assert report["violations"] == 0

    def test_incompatible_operator(self):
        """Test operator that violates seam-compatibility."""

        def bad_operator(q):
            u, v = q[0], q[1]
            return np.array([[1.0, u], [v, 1.0]])  # Not compatible

        strip = Strip(w=1.0)
        report = validate_operator_invariance(bad_operator, strip)

        assert not report["seam_compatible"]
        assert report["violations"] > 0

    def test_operator_report_structure(self):
        """Test operator validation report structure."""

        def op_fn(q):
            return np.eye(2)

        strip = Strip(w=1.0)
        report = validate_operator_invariance(op_fn, strip, n_test=15)

        required_keys = [
            "test_type",
            "n_test",
            "violations",
            "violation_rate",
            "max_error",
            "mean_error",
            "tolerance",
            "seam_compatible",
            "violation_details",
        ]

        for key in required_keys:
            assert key in report

        assert report["test_type"] == "operator_seam_compatibility"
        assert report["n_test"] == 15


class TestValidateGeodesicInvariance:
    """Test geodesic invariance validation."""

    def test_straight_line_geodesics(self):
        """Test geodesics in flat metric (should be invariant)."""

        def simple_geodesic_fn(q0, v0, t):
            # Simple straight-line "geodesics"
            n_points = 10
            t_vals = np.linspace(0, t, n_points)
            traj_q = np.array([q0 + t_val * v0 for t_val in t_vals])
            traj_v = np.array([v0 for _ in range(n_points)])
            info = {"success": True}
            return traj_q, traj_v, info

        strip = Strip(w=1.0)
        report = validate_geodesic_invariance(simple_geodesic_fn, strip, n_test=5)

        # Straight lines should be approximately invariant
        assert report["geodesics_invariant"] or report["violation_rate"] < 0.5

    def test_failing_geodesic_integration(self):
        """Test handling of geodesic integration failures."""

        def failing_geodesic_fn(q0, v0, t):
            return None, None, {"success": False}

        strip = Strip(w=1.0)
        report = validate_geodesic_invariance(failing_geodesic_fn, strip, n_test=3)

        assert not report["geodesics_invariant"]
        assert report["violations"] > 0

    def test_geodesic_report_structure(self):
        """Test geodesic validation report structure."""

        def dummy_geodesic_fn(q0, v0, t):
            return np.array([q0]), np.array([v0]), {"success": True}

        strip = Strip(w=1.0)
        report = validate_geodesic_invariance(dummy_geodesic_fn, strip, n_test=2)

        required_keys = [
            "test_type",
            "n_test",
            "violations",
            "violation_rate",
            "max_trajectory_error",
            "mean_trajectory_error",
            "tolerance",
            "geodesics_invariant",
            "violation_details",
        ]

        for key in required_keys:
            assert key in report

        assert report["test_type"] == "geodesic_invariance"
        assert report["n_test"] == 2


class TestComprehensiveTopologyValidation:
    """Test comprehensive validation suite."""

    def test_minimal_validation_suite(self):
        """Test validation with only metric function."""

        def metric_fn(q):
            return np.eye(2)

        report = comprehensive_topology_validation(metric_fn)

        assert "metric_seam_compatibility" in report
        assert "tests_run" in report
        assert "metric_seam_compatibility" in report["tests_run"]
        assert report["all_passed"]

    def test_full_validation_suite(self):
        """Test validation with all components."""

        def metric_fn(q):
            return np.eye(2)

        def spectrum_fn(q):
            return np.array([1.0, 2.0, 3.0])

        def operator_fn(q):
            return np.eye(2)

        def geodesic_fn(q0, v0, t):
            return np.array([q0]), np.array([v0]), {"success": True}

        report = comprehensive_topology_validation(
            metric_fn, spectrum_fn, operator_fn, geodesic_fn
        )

        expected_tests = [
            "metric_seam_compatibility",
            "spectral_invariance",
            "operator_seam_compatibility",
            "geodesic_invariance",
        ]

        for test in expected_tests:
            assert test in report
            assert test in report["tests_run"]

    def test_validation_with_failures(self):
        """Test validation suite with some failures."""

        def good_metric(q):
            return np.eye(2)

        def bad_spectrum(q):
            u, v = q[0], q[1]
            return np.array([u, v])  # Not invariant

        report = comprehensive_topology_validation(good_metric, bad_spectrum)

        assert not report["all_passed"]
        assert report["metric_seam_compatibility"]["compatible"]
        assert not report["spectral_invariance"]["invariant"]

    def test_custom_strip_configuration(self):
        """Test validation with custom strip configuration."""

        def metric_fn(q):
            return np.eye(2)

        custom_strip = Strip(w=2.0, period=4 * np.pi)

        report = comprehensive_topology_validation(
            metric_fn, strip=custom_strip, tolerance=1e-10
        )

        assert report["strip_config"]["w"] == 2.0
        assert report["strip_config"]["period"] == 4 * np.pi
        assert report["tolerance"] == 1e-10

    def test_validation_report_structure(self):
        """Test structure of comprehensive validation report."""

        def metric_fn(q):
            return np.eye(2)

        report = comprehensive_topology_validation(metric_fn)

        required_top_level_keys = [
            "strip_config",
            "tolerance",
            "tests_run",
            "all_passed",
        ]

        for key in required_top_level_keys:
            assert key in report

        assert isinstance(report["tests_run"], list)
        assert isinstance(report["all_passed"], bool)


if __name__ == "__main__":
    pytest.main([__file__])
