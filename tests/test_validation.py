"""Tests for the validation framework."""

import sys

import numpy as np
import pytest
from scipy.sparse import csr_matrix

sys.path.insert(0, "src")

from validation.mathematical import (
    ConnectivityError,
    TransversalityError,
    check_graph_connectivity,
    validate_transversality,
)
from validation.numerical import (
    NumericalStabilityError,
    check_eigenvalue_validity,
    monitor_mass_conservation,
    validate_cg_convergence,
)
from validation.performance import (
    PerformanceError,
    check_memory_limits,
    detect_scaling_cliffs,
)
from validation.reproducibility import (
    ReproducibilityError,
    compute_data_hash,
    ensure_reproducibility,
    verify_data_integrity,
)
from validation.statistical import (
    StatisticalValidityError,
    apply_multiple_testing_correction,
    check_separability_null,
    validate_bootstrap_size,
)


class TestMathematicalValidation:
    """Tests for mathematical validity guards."""

    def test_connectivity_check_connected_graph(self):
        """Test connectivity check on connected graph."""
        # Create connected path graph
        A = np.array([[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]])
        A_sparse = csr_matrix(A)

        # Should pass without error
        is_connected = check_graph_connectivity(A_sparse)
        assert is_connected

        # With details
        connected, n_comp, labels = check_graph_connectivity(
            A_sparse, return_components=True
        )
        assert connected
        assert n_comp == 1
        assert len(labels) == 4

    def test_connectivity_check_disconnected_graph(self):
        """Test connectivity check on disconnected graph."""
        # Create disconnected graph (two components)
        A = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
        A_sparse = csr_matrix(A)

        # Should detect disconnection
        is_connected = check_graph_connectivity(A_sparse, require_connected=False)
        assert not is_connected

        # Should raise error when required
        with pytest.raises(ConnectivityError):
            check_graph_connectivity(A_sparse, require_connected=True)

    def test_transversality_validation_success(self):
        """Test transversality validation on well-conditioned Jacobian."""
        # Create full-rank 2x4 Jacobian
        J_f = np.array([[1.0, 0.0, 0.5, 0.0], [0.0, 1.0, 0.0, 0.5]])

        cert = validate_transversality(J_f, expected_rank=2)

        assert cert["is_transversal"]
        assert cert["rank"] == 2
        assert cert["condition_number"] < 1e6

    def test_transversality_validation_failure(self):
        """Test transversality validation on rank-deficient Jacobian."""
        # Create rank-1 Jacobian (second row is multiple of first)
        J_f = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]])  # Rank deficient

        with pytest.raises(TransversalityError):
            validate_transversality(J_f, expected_rank=2)

    def test_transversality_ill_conditioned(self):
        """Test transversality validation on ill-conditioned Jacobian."""
        # Create ill-conditioned Jacobian
        J_f = np.array([[1.0, 0.0], [0.0, 1e-8]])  # Very small singular value

        with pytest.raises(TransversalityError):
            validate_transversality(J_f, expected_rank=2, condition_threshold=1e6)


class TestNumericalStability:
    """Tests for numerical stability guards."""

    def test_mass_conservation_success(self):
        """Test mass conservation monitoring with conserved mass."""
        u_initial = np.array([1.0, 0.5, 0.3, 0.2])
        u_final = np.array([0.8, 0.6, 0.35, 0.25])  # Small redistribution

        report = monitor_mass_conservation(u_initial, u_final, tolerance=1e-2)

        assert report["conserved"]
        assert abs(report["mass_change"]) < 1e-2

    def test_mass_conservation_failure(self):
        """Test mass conservation monitoring with violated conservation."""
        u_initial = np.array([1.0, 1.0, 1.0])
        u_final = np.array([0.5, 0.5, 0.5])  # Mass lost

        with pytest.raises(NumericalStabilityError):
            monitor_mass_conservation(u_initial, u_final, tolerance=1e-6)

    def test_cg_convergence_success(self):
        """Test CG convergence validation with good convergence."""
        # Simulated CG residual history (decreasing)
        residuals = [1.0, 0.5, 0.1, 0.01, 0.001, 1e-6]

        report = validate_cg_convergence(residuals, tolerance=1e-5)

        assert report["converged"]
        assert not report["stagnation_detected"]

    def test_cg_convergence_stagnation(self):
        """Test CG convergence validation with stagnation."""
        # Simulated stagnation (residual barely decreases)
        residuals = [
            1.0,
            0.999,
            0.9989,
            0.9988,
            0.9987,
            0.9986,
            0.9985,
            0.9984,
            0.9983,
            0.9982,
            0.9981,
        ]

        with pytest.raises(NumericalStabilityError):
            validate_cg_convergence(
                residuals, max_stagnation_ratio=0.999, min_progress_steps=5
            )

    def test_eigenvalue_validity_laplacian(self):
        """Test eigenvalue validation for Laplacian matrix."""
        # Valid Laplacian eigenvalues (non-negative, one zero)
        evals = np.array([0.0, 0.5, 1.0, 1.5, 2.0])

        report = check_eigenvalue_validity(evals, matrix_type="laplacian")

        assert report["valid"]
        assert report["spectral_gap"] == 0.5

    def test_eigenvalue_validity_negative(self):
        """Test eigenvalue validation with negative eigenvalues."""
        # Invalid Laplacian eigenvalues (negative)
        evals = np.array([-0.1, 0.0, 1.0, 2.0])

        with pytest.raises(NumericalStabilityError):
            check_eigenvalue_validity(evals, matrix_type="laplacian")

    def test_eigenvalue_validity_complex(self):
        """Test eigenvalue validation with complex eigenvalues."""
        # Complex eigenvalues (invalid for symmetric matrices)
        evals = np.array([1.0 + 0.1j, 2.0, 3.0])

        with pytest.raises(NumericalStabilityError):
            check_eigenvalue_validity(evals, matrix_type="symmetric")


class TestStatisticalValidation:
    """Tests for statistical rigor guards."""

    def test_bootstrap_size_validation_sufficient(self):
        """Test bootstrap size validation with sufficient samples."""
        report = validate_bootstrap_size(2000)

        assert report["valid"]
        assert report["reliability_level"] == "good"

    def test_bootstrap_size_validation_insufficient(self):
        """Test bootstrap size validation with insufficient samples."""
        with pytest.raises(StatisticalValidityError):
            validate_bootstrap_size(50)

    def test_bootstrap_size_validation_minimal(self):
        """Test bootstrap size validation with minimal samples."""
        report = validate_bootstrap_size(500)

        assert report["valid"]
        assert report["reliability_level"] == "minimal"
        assert len(report["recommendations"]) > 0

    def test_separability_null_validation(self):
        """Test separability null validation with identical samples."""
        # Create identical samples
        sample1 = np.array([1.0, 1.1, 0.9, 1.05])
        sample2 = np.array([1.0, 1.1, 0.9, 1.05])

        # Mock separability test function that should NOT detect difference
        def mock_test(s1, s2):
            from scipy import stats

            _, p_val = stats.ttest_ind(s1, s2)
            return {"p_value": p_val, "separable": p_val < 0.05}

        report = check_separability_null(sample1, sample2, mock_test, n_trials=10)

        assert report["valid_null_behavior"]
        assert (
            report["observed_type1_rate"] <= 0.1
        )  # Should be low for identical samples

    def test_multiple_testing_correction_bonferroni(self):
        """Test Bonferroni multiple testing correction."""
        p_values = np.array([0.01, 0.02, 0.03, 0.04, 0.05])

        result = apply_multiple_testing_correction(p_values, method="bonferroni")

        assert len(result["corrected_p_values"]) == 5
        assert np.all(result["corrected_p_values"] >= p_values)
        assert result["n_significant_corrected"] <= result["n_significant_raw"]

    def test_multiple_testing_correction_holm(self):
        """Test Holm step-down multiple testing correction."""
        p_values = np.array([0.001, 0.01, 0.02, 0.03])

        result = apply_multiple_testing_correction(p_values, method="holm")

        assert result["method"] == "holm"
        assert np.all(result["corrected_p_values"] >= p_values)

    def test_multiple_testing_correction_bh(self):
        """Test Benjamini-Hochberg FDR correction."""
        p_values = np.array([0.001, 0.01, 0.02, 0.03])

        result = apply_multiple_testing_correction(
            p_values, method="benjamini_hochberg"
        )

        assert result["method"] == "benjamini_hochberg"
        # BH is less conservative than Bonferroni
        assert result["n_significant_corrected"] >= 0


class TestReproducibility:
    """Tests for reproducibility framework."""

    def test_ensure_reproducibility_success(self):
        """Test successful reproducibility setup."""
        report = ensure_reproducibility(42)

        assert report["reproducible"]
        assert report["seed"] == 42
        assert "numpy" in report["seed_status"]
        assert report["seed_status"]["numpy"] == "success"

    def test_compute_data_hash_consistency(self):
        """Test data hash computation consistency."""
        data = np.array([[1, 2, 3], [4, 5, 6]])

        hash1 = compute_data_hash(data)
        hash2 = compute_data_hash(data)

        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 hex length

    def test_compute_data_hash_sensitivity(self):
        """Test data hash sensitivity to changes."""
        data1 = np.array([1, 2, 3])
        data2 = np.array([1, 2, 4])  # Small change

        hash1 = compute_data_hash(data1)
        hash2 = compute_data_hash(data2)

        assert hash1 != hash2

    def test_verify_data_integrity_success(self):
        """Test successful data integrity verification."""
        data = np.array([1, 2, 3, 4, 5])
        expected_hash = compute_data_hash(data)

        report = verify_data_integrity(data, expected_hash)

        assert report["integrity_valid"]

    def test_verify_data_integrity_failure(self):
        """Test data integrity verification failure."""
        data = np.array([1, 2, 3, 4, 5])
        wrong_hash = "0123456789abcdef" * 4  # Wrong hash

        with pytest.raises(ReproducibilityError):
            verify_data_integrity(data, wrong_hash)

    def test_data_hash_dict_input(self):
        """Test data hash computation with dictionary input."""
        data_dict = {"X": np.array([[1, 2], [3, 4]]), "y": np.array([0, 1])}

        hash1 = compute_data_hash(data_dict)
        hash2 = compute_data_hash(data_dict)

        assert hash1 == hash2

        # Different order should give same hash (sorted keys)
        data_dict_reordered = {"y": data_dict["y"], "X": data_dict["X"]}
        hash3 = compute_data_hash(data_dict_reordered)

        assert hash1 == hash3


class TestPerformanceMonitoring:
    """Tests for performance monitoring."""

    def test_memory_limits_within_bounds(self):
        """Test memory limits check for reasonable matrix size."""
        # Small matrix should pass
        report = check_memory_limits((100, 100), max_memory_gb=1.0)

        assert report["within_limits"]

    def test_memory_limits_exceeded(self):
        """Test memory limits check for oversized matrix."""
        # Huge matrix should fail
        with pytest.raises(PerformanceError):
            check_memory_limits((100000, 100000), max_memory_gb=0.1)

    def test_scaling_cliff_detection_no_cliff(self):
        """Test scaling cliff detection with smooth scaling."""

        def smooth_function(n):
            return 0.001 * n**2  # O(n²) scaling

        sizes = [10, 20, 40, 80]

        report = detect_scaling_cliffs(sizes, smooth_function, cliff_threshold=5.0)

        assert report["cliff_count"] == 0

    def test_scaling_cliff_detection_with_cliff(self):
        """Test scaling cliff detection with sudden degradation."""

        def cliff_function(n):
            if n <= 50:
                return 0.001 * n**2
            else:
                return 0.001 * n**3  # Sudden complexity increase

        sizes = [20, 40, 60, 80]

        with pytest.raises(PerformanceError):
            detect_scaling_cliffs(sizes, cliff_function, cliff_threshold=3.0)


class TestIntegrationValidation:
    """Integration tests for validation framework."""

    def test_complete_validation_pipeline(self):
        """Test complete validation pipeline on realistic data."""
        # Create test graph
        rng = np.random.default_rng(42)
        X = rng.normal(size=(20, 3))

        # Build adjacency matrix
        from graphs.knn import build_graph

        A = build_graph(X, mode="additive", k=5, seed=42)

        # Test connectivity
        check_graph_connectivity(A, require_connected=False)
        # Should be connected for this size and k

        # Create test Jacobian
        J_f = rng.normal(size=(2, 3))
        J_f = J_f + 0.1 * np.eye(2, 3)  # Ensure full rank

        # Test transversality
        cert = validate_transversality(J_f, expected_rank=2)
        assert cert["is_transversal"]

        # Test reproducibility
        report = ensure_reproducibility(42)
        assert report["reproducible"]

        # Test data hashing
        data_hash = compute_data_hash(X)
        assert len(data_hash) == 64

    def test_validation_error_handling(self):
        """Test that validation framework handles edge cases gracefully."""
        # Empty array
        with pytest.raises(ValueError):
            check_graph_connectivity(np.array([]).reshape(0, 0))

        # Non-square matrix
        with pytest.raises(ValueError):
            check_graph_connectivity(np.array([[1, 2, 3]]))

        # Insufficient residual history
        with pytest.raises(ValueError):
            validate_cg_convergence([1.0])  # Need at least 2 values

        # Invalid bootstrap size
        with pytest.raises(StatisticalValidityError):
            validate_bootstrap_size(10)

    def test_validation_warnings(self):
        """Test that validation framework issues appropriate warnings."""
        # Test precision warning
        from validation.numerical import validate_float64_precision

        arrays = {"test_array": np.array([1, 2, 3], dtype=np.float32)}  # Low precision

        with pytest.warns(UserWarning):
            report = validate_float64_precision(arrays)

        # Should issue precision warning
        assert not report["precision_adequate"]
