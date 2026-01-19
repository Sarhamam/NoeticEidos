"""Tests for topology geodesic integration."""

import numpy as np
import pytest
import warnings
import sys
sys.path.insert(0, 'src')

from topology.coords import Strip
from topology.geodesic import (
    GeodesicIntegrationError,
    christoffel,
    geodesic_acceleration,
    geodesic_leapfrog_step,
    geodesic_energy,
    integrate_geodesic,
    adaptive_geodesic_step,
    geodesic_distance
)


class TestChristoffel:
    """Test Christoffel symbol computation."""

    def test_flat_metric_christoffel(self):
        """Test Christoffel symbols for flat metric are zero."""
        g = np.eye(2)
        du_g = np.zeros((2, 2))
        dv_g = np.zeros((2, 2))

        Gamma = christoffel(g, du_g, dv_g)

        assert Gamma.shape == (2, 2, 2)
        assert np.allclose(Gamma, 0.0)

    def test_diagonal_metric_christoffel(self):
        """Test Christoffel symbols for diagonal metric."""
        # Metric: g = diag(u^2, v^2)
        u, v = 1.0, 2.0
        g = np.diag([u**2, v**2])

        # Derivatives: ∂g/∂u and ∂g/∂v
        du_g = np.diag([2*u, 0])
        dv_g = np.diag([0, 2*v])

        Gamma = christoffel(g, du_g, dv_g)

        # For diagonal metric g = diag(g11, g22):
        # Γ¹₁₁ = (1/2g11) ∂g11/∂u = (1/2u²)(2u) = 1/u
        # Γ²₂₂ = (1/2g22) ∂g22/∂v = (1/2v²)(2v) = 1/v
        assert np.isclose(Gamma[0, 0, 0], 1.0/u)  # Γ¹₁₁
        assert np.isclose(Gamma[1, 1, 1], 1.0/v)  # Γ²₂₂

        # Off-diagonal should be zero for diagonal metric
        assert np.isclose(Gamma[0, 0, 1], 0.0)
        assert np.isclose(Gamma[0, 1, 0], 0.0)
        assert np.isclose(Gamma[1, 0, 0], 0.0)
        assert np.isclose(Gamma[1, 1, 0], 0.0)

    def test_singular_metric_error(self):
        """Test error handling for singular metric."""
        g = np.array([[1.0, 1.0], [1.0, 1.0]])  # Singular
        du_g = np.zeros((2, 2))
        dv_g = np.zeros((2, 2))

        with pytest.raises(GeodesicIntegrationError, match="Metric is singular"):
            christoffel(g, du_g, dv_g)

    def test_christoffel_symmetry(self):
        """Test symmetry property: Γᵏᵢⱼ = Γᵏⱼᵢ."""
        # Random symmetric metric
        g = np.array([[2.0, 0.5], [0.5, 1.5]])
        du_g = np.array([[0.1, 0.2], [0.2, -0.1]])
        dv_g = np.array([[-0.2, 0.3], [0.3, 0.1]])

        Gamma = christoffel(g, du_g, dv_g)

        # Check symmetry in lower indices
        for k in range(2):
            for i in range(2):
                for j in range(2):
                    assert np.isclose(Gamma[k, i, j], Gamma[k, j, i])


class TestGeodesicAcceleration:
    """Test geodesic acceleration computation."""

    def test_flat_metric_acceleration(self):
        """Test acceleration is zero for flat metric with constant velocity."""
        def flat_metric(q):
            return np.eye(2)

        def flat_grad(q):
            return np.zeros((2, 2)), np.zeros((2, 2))

        q = np.array([1.0, 2.0])
        v = np.array([0.5, -0.3])

        accel = geodesic_acceleration(q, v, flat_metric, flat_grad)
        assert np.allclose(accel, 0.0)

    def test_acceleration_input_validation(self):
        """Test input validation for acceleration computation."""
        def metric_fn(q):
            return np.eye(2)

        def grad_fn(q):
            return np.zeros((2, 2)), np.zeros((2, 2))

        # Wrong shapes
        with pytest.raises(ValueError, match="Position and velocity must be 2D vectors"):
            geodesic_acceleration(np.array([1.0]), np.array([0.5, 0.3]), metric_fn, grad_fn)

        with pytest.raises(ValueError, match="Position and velocity must be 2D vectors"):
            geodesic_acceleration(np.array([1.0, 2.0]), np.array([0.5]), metric_fn, grad_fn)

    def test_spherical_metric_acceleration(self):
        """Test acceleration for spherical metric (non-trivial example)."""
        def spherical_metric(q):
            # Metric in spherical coordinates (θ, φ)
            theta = q[1]
            return np.diag([1.0, np.sin(theta)**2])

        def spherical_grad(q):
            theta = q[1]
            du_g = np.zeros((2, 2))
            dv_g = np.array([[0.0, 0.0], [0.0, 2*np.sin(theta)*np.cos(theta)]])
            return du_g, dv_g

        q = np.array([0.0, np.pi/4])  # 45 degrees
        v = np.array([0.0, 1.0])      # Moving in θ direction (to get curvature effect)

        accel = geodesic_acceleration(q, v, spherical_metric, spherical_grad)

        # Should have non-zero acceleration due to curvature
        # For motion in θ direction with varying metric coefficient
        assert not np.allclose(accel, 0.0)
        assert np.abs(accel[1]) > 1e-10  # Non-zero θ acceleration


class TestGeodesicLeapfrogStep:
    """Test single leapfrog integration step."""

    def test_flat_metric_straight_line(self):
        """Test that geodesics in flat metric are straight lines."""
        def flat_metric(q):
            return np.eye(2)

        def flat_grad(q):
            return np.zeros((2, 2)), np.zeros((2, 2))

        strip = Strip(w=2.0)  # Large strip to avoid seam crossings
        q = np.array([0.5, 0.3])
        v = np.array([0.1, 0.2])
        dt = 0.01

        q_new, v_new = geodesic_leapfrog_step(q, v, dt, flat_metric, flat_grad, strip)

        # Should move along straight line
        expected_q = q + dt * v
        assert np.allclose(q_new, expected_q, atol=1e-10)
        assert np.allclose(v_new, v, atol=1e-10)

    def test_seam_crossing_handling(self):
        """Test seam crossing during integration step."""
        def flat_metric(q):
            return np.eye(2)

        def flat_grad(q):
            return np.zeros((2, 2)), np.zeros((2, 2))

        strip = Strip(w=1.0)
        q = np.array([0.5, 0.9])   # Near top boundary
        v = np.array([0.0, 0.5])   # Moving toward boundary
        dt = 0.5  # Large step to ensure crossing

        q_new, v_new = geodesic_leapfrog_step(q, v, dt, flat_metric, flat_grad, strip)

        # Should have crossed seam and been transformed
        assert np.abs(q_new[1]) <= strip.w  # Within strip bounds
        if q_new[1] == -strip.w:  # Crossed to bottom
            # Velocity should be transformed
            assert v_new[1] < 0  # v-velocity should flip sign


class TestGeodesicEnergy:
    """Test geodesic energy computation."""

    def test_energy_flat_metric(self):
        """Test energy computation for flat metric."""
        def flat_metric(q):
            return np.eye(2)

        q = np.array([1.0, 2.0])
        v = np.array([0.5, -0.3])

        energy = geodesic_energy(q, v, flat_metric)
        expected = 0.5 * (0.5**2 + (-0.3)**2)
        assert np.isclose(energy, expected)

    def test_energy_scaled_metric(self):
        """Test energy with scaled metric."""
        def scaled_metric(q):
            return 2.0 * np.eye(2)

        q = np.array([0.0, 0.0])
        v = np.array([1.0, 1.0])

        energy = geodesic_energy(q, v, scaled_metric)
        expected = 0.5 * 2.0 * (1.0**2 + 1.0**2)  # 2.0 from metric scaling
        assert np.isclose(energy, expected)

    def test_energy_positive(self):
        """Test that energy is always non-negative."""
        def positive_metric(q):
            return np.array([[2.0, 0.5], [0.5, 1.0]])  # Positive definite

        q = np.array([0.5, -0.3])
        v = np.array([-0.2, 0.8])

        energy = geodesic_energy(q, v, positive_metric)
        assert energy >= 0.0


class TestIntegrateGeodesic:
    """Test full geodesic integration."""

    def test_straight_line_integration(self):
        """Test integration of straight line in flat metric."""
        def flat_metric(q):
            return np.eye(2)

        def flat_grad(q):
            return np.zeros((2, 2)), np.zeros((2, 2))

        strip = Strip(w=2.0)  # Large strip
        q0 = np.array([0.0, 0.0])
        v0 = np.array([1.0, 0.5])
        t_final = 1.0
        dt = 0.01

        traj_q, traj_v, info = integrate_geodesic(
            q0, v0, t_final, dt, flat_metric, flat_grad, strip
        )

        assert info["success"]
        assert info["final_time"] >= t_final

        # Check straight line motion
        expected_final = q0 + t_final * v0
        assert np.allclose(traj_q[-1], expected_final, atol=1e-3)

        # Check constant velocity
        assert np.allclose(traj_v[-1], v0, atol=1e-6)

    def test_energy_conservation(self):
        """Test energy conservation during integration."""
        def constant_metric(q):
            return np.array([[2.0, 0.5], [0.5, 1.5]])

        def constant_grad(q):
            return np.zeros((2, 2)), np.zeros((2, 2))

        strip = Strip(w=2.0)
        q0 = np.array([0.5, 0.3])
        v0 = np.array([0.2, -0.1])
        t_final = 2.0
        dt = 0.001  # Small time step for accuracy

        traj_q, traj_v, info = integrate_geodesic(
            q0, v0, t_final, dt, constant_metric, constant_grad, strip,
            energy_tolerance=1e-6
        )

        assert info["success"]
        assert info["energy_conservation"]

        # Check energy arrays
        energy_drift = np.abs(info["energy_array"] - info["initial_energy"])
        max_drift = np.max(energy_drift) / info["initial_energy"]
        assert max_drift < 1e-5

    def test_integration_error_handling(self):
        """Test error handling during integration."""
        def bad_metric(q):
            # Metric becomes singular at origin
            u, v = q[0], q[1]
            if np.abs(u) < 1e-6 and np.abs(v) < 1e-6:
                return np.zeros((2, 2))  # Singular
            return np.eye(2)

        def bad_grad(q):
            return np.zeros((2, 2)), np.zeros((2, 2))

        strip = Strip(w=1.0)
        q0 = np.array([0.1, 0.1])  # Start near singularity
        v0 = np.array([-1.0, -1.0])  # Move toward origin
        t_final = 1.0
        dt = 0.1

        with pytest.raises(GeodesicIntegrationError):
            integrate_geodesic(q0, v0, t_final, dt, bad_metric, bad_grad, strip)

    def test_excessive_energy_drift_warning(self):
        """Test warning for excessive energy drift."""
        def bad_metric(q):
            # Metric with large gradient for numerical instability
            u, v = q[0], q[1]
            return np.array([[1.0 + 100*u**2, 0.0], [0.0, 1.0 + 100*v**2]])

        def bad_grad(q):
            u, v = q[0], q[1]
            du_g = np.array([[200*u, 0.0], [0.0, 0.0]])
            dv_g = np.array([[0.0, 0.0], [0.0, 200*v]])
            return du_g, dv_g

        strip = Strip(w=2.0)
        q0 = np.array([0.5, 0.3])
        v0 = np.array([1.0, 1.0])
        t_final = 0.5
        dt = 0.1  # Large time step for instability

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                traj_q, traj_v, info = integrate_geodesic(
                    q0, v0, t_final, dt, bad_metric, bad_grad, strip,
                    energy_tolerance=1e-3
                )
            except GeodesicIntegrationError:
                pass  # Expected for this test

            # Should have warnings about energy drift
            assert len(w) > 0
            assert any("Energy drift" in str(warning.message) for warning in w)

    def test_save_every_parameter(self):
        """Test save_every parameter for trajectory storage."""
        def flat_metric(q):
            return np.eye(2)

        def flat_grad(q):
            return np.zeros((2, 2)), np.zeros((2, 2))

        strip = Strip(w=2.0)
        q0 = np.array([0.0, 0.0])
        v0 = np.array([1.0, 0.0])
        t_final = 1.0
        dt = 0.01
        save_every = 10

        traj_q, traj_v, info = integrate_geodesic(
            q0, v0, t_final, dt, flat_metric, flat_grad, strip,
            save_every=save_every
        )

        n_steps = int(np.ceil(t_final / dt))
        expected_saved = int(np.ceil(n_steps / save_every)) + 1

        assert len(traj_q) <= expected_saved + 1  # Allow for final point
        assert len(traj_v) == len(traj_q)
        assert info["n_saved"] == len(traj_q)

    def test_integration_statistics(self):
        """Test integration statistics reporting."""
        def flat_metric(q):
            return np.eye(2)

        def flat_grad(q):
            return np.zeros((2, 2)), np.zeros((2, 2))

        strip = Strip(w=1.0)
        q0 = np.array([0.0, 0.8])   # Near seam
        v0 = np.array([1.0, 0.5])   # Will cross seam
        t_final = 2.0
        dt = 0.01

        traj_q, traj_v, info = integrate_geodesic(
            q0, v0, t_final, dt, flat_metric, flat_grad, strip
        )

        # Check required info fields
        required_fields = [
            "success", "n_steps", "n_saved", "final_time", "dt",
            "initial_energy", "final_energy", "energy_drift",
            "max_energy_error", "seam_crossings", "energy_conservation",
            "trajectory_length"
        ]

        for field in required_fields:
            assert field in info

        assert info["seam_crossings"] > 0  # Should cross seam
        assert info["trajectory_length"] > 0


class TestAdaptiveGeodesicStep:
    """Test adaptive step size integration."""

    def test_adaptive_step_accuracy(self):
        """Test adaptive step improves accuracy."""
        def constant_metric(q):
            return np.eye(2)

        def constant_grad(q):
            return np.zeros((2, 2)), np.zeros((2, 2))

        strip = Strip(w=2.0)
        q = np.array([0.5, 0.3])
        v = np.array([0.1, 0.2])
        dt = 0.1

        q_new, v_new, dt_new, accept = adaptive_geodesic_step(
            q, v, constant_metric, constant_grad, strip, dt, target_error=1e-8
        )

        # For constant coefficient case, should be very accurate
        assert accept
        assert dt_new > 0

    def test_adaptive_step_rejection(self):
        """Test step rejection for large errors."""
        def varying_metric(q):
            u, v = q[0], q[1]
            return np.array([[1.0 + u**2, 0.0], [0.0, 1.0 + v**2]])

        def varying_grad(q):
            u, v = q[0], q[1]
            du_g = np.array([[2*u, 0.0], [0.0, 0.0]])
            dv_g = np.array([[0.0, 0.0], [0.0, 2*v]])
            return du_g, dv_g

        strip = Strip(w=2.0)
        q = np.array([1.0, 1.0])  # Large coordinates
        v = np.array([1.0, 1.0])
        dt = 1.0  # Very large time step

        q_new, v_new, dt_new, accept = adaptive_geodesic_step(
            q, v, varying_metric, varying_grad, strip, dt, target_error=1e-10
        )

        # Should suggest smaller step
        assert dt_new < dt


class TestGeodesicDistance:
    """Test geodesic distance computation."""

    def test_distance_flat_metric(self):
        """Test geodesic distance in flat metric equals Euclidean."""
        def flat_metric(q):
            return np.eye(2)

        def flat_grad(q):
            return np.zeros((2, 2)), np.zeros((2, 2))

        strip = Strip(w=2.0)
        q1 = np.array([0.0, 0.0])
        q2 = np.array([1.0, 1.0])

        distance = geodesic_distance(q1, q2, flat_metric, flat_grad, strip)
        euclidean = np.linalg.norm(q2 - q1)

        assert np.isclose(distance, euclidean, rtol=1e-2)

    def test_distance_same_point(self):
        """Test distance from point to itself is zero."""
        def metric_fn(q):
            return np.eye(2)

        def grad_fn(q):
            return np.zeros((2, 2)), np.zeros((2, 2))

        strip = Strip(w=1.0)
        q = np.array([0.5, 0.3])

        distance = geodesic_distance(q, q, metric_fn, grad_fn, strip)
        assert distance < 1e-10

    def test_distance_symmetry(self):
        """Test that geodesic distance is symmetric."""
        def metric_fn(q):
            return np.eye(2)

        def grad_fn(q):
            return np.zeros((2, 2)), np.zeros((2, 2))

        strip = Strip(w=1.0)
        q1 = np.array([0.2, 0.3])
        q2 = np.array([0.8, 0.7])

        dist12 = geodesic_distance(q1, q2, metric_fn, grad_fn, strip)
        dist21 = geodesic_distance(q2, q1, metric_fn, grad_fn, strip)

        assert np.isclose(dist12, dist21, rtol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__])