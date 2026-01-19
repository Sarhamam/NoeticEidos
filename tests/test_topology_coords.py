"""Tests for topology coordinate handling."""

import numpy as np
import pytest
import sys
sys.path.insert(0, 'src')

from topology.coords import (
    Strip, wrap_u, deck_map, pushforward_velocity,
    apply_seam_if_needed, is_on_seam, seam_equivalent_points,
    distance_on_quotient
)


class TestStrip:
    """Test Strip configuration class."""

    def test_strip_creation(self):
        """Test Strip dataclass creation."""
        strip = Strip(w=1.5, period=4*np.pi)
        assert strip.w == 1.5
        assert strip.period == 4*np.pi

    def test_strip_defaults(self):
        """Test Strip default values."""
        strip = Strip(w=1.0)
        assert strip.w == 1.0
        assert strip.period == 2*np.pi


class TestWrapU:
    """Test u-coordinate wrapping."""

    def test_wrap_single_value(self):
        """Test wrapping single u value."""
        assert np.isclose(wrap_u(0.0), 0.0)
        assert np.isclose(wrap_u(2*np.pi), 0.0)
        assert np.isclose(wrap_u(3*np.pi), np.pi)
        assert np.isclose(wrap_u(-np.pi), np.pi)

    def test_wrap_array(self):
        """Test wrapping array of u values."""
        u = np.array([0.0, np.pi, 2*np.pi, 3*np.pi, -np.pi])
        u_wrapped = wrap_u(u)
        expected = np.array([0.0, np.pi, 0.0, np.pi, np.pi])
        assert np.allclose(u_wrapped, expected)

    def test_wrap_custom_period(self):
        """Test wrapping with custom period."""
        u = np.array([0.0, 2.0, 4.0, 6.0, -2.0])
        u_wrapped = wrap_u(u, period=4.0)
        expected = np.array([0.0, 2.0, 0.0, 2.0, 2.0])
        assert np.allclose(u_wrapped, expected)


class TestDeckMap:
    """Test deck map transformation."""

    def test_deck_map_single_point(self):
        """Test deck map on single point."""
        strip = Strip(w=1.0)
        u, v = 0.0, 0.5
        u_mapped, v_mapped = deck_map(u, v, strip)

        assert np.isclose(u_mapped, np.pi)
        assert np.isclose(v_mapped, -0.5)

    def test_deck_map_array(self):
        """Test deck map on arrays."""
        strip = Strip(w=1.0)
        u = np.array([0.0, np.pi/2, np.pi])
        v = np.array([0.5, -0.3, 0.8])

        u_mapped, v_mapped = deck_map(u, v, strip)

        expected_u = np.array([np.pi, 3*np.pi/2, 0.0])  # Wrapped
        expected_v = np.array([-0.5, 0.3, -0.8])

        assert np.allclose(u_mapped, expected_u)
        assert np.allclose(v_mapped, expected_v)

    def test_deck_map_involution(self):
        """Test that deck map is an involution: T(T(q)) = q."""
        strip = Strip(w=1.0)
        u, v = 0.3, 0.7

        # Apply deck map twice
        u1, v1 = deck_map(u, v, strip)
        u2, v2 = deck_map(u1, v1, strip)

        # Should return to original (modulo wrapping)
        assert np.isclose(u2, u)
        assert np.isclose(v2, v)

    def test_deck_map_custom_strip(self):
        """Test deck map with custom strip configuration."""
        strip = Strip(w=2.0, period=4*np.pi)
        u, v = np.pi, 1.5

        u_mapped, v_mapped = deck_map(u, v, strip)

        assert np.isclose(u_mapped, 2*np.pi)  # u + π
        assert np.isclose(v_mapped, -1.5)     # -v


class TestPushforwardVelocity:
    """Test velocity pushforward under deck map."""

    def test_pushforward_single_velocity(self):
        """Test pushforward of single velocity vector."""
        du, dv = 0.5, -0.3
        du_pushed, dv_pushed = pushforward_velocity(du, dv)

        assert np.isclose(du_pushed, 0.5)   # Unchanged
        assert np.isclose(dv_pushed, 0.3)   # Sign flipped

    def test_pushforward_array(self):
        """Test pushforward of velocity arrays."""
        du = np.array([0.5, -0.2, 1.0])
        dv = np.array([-0.3, 0.8, -0.1])

        du_pushed, dv_pushed = pushforward_velocity(du, dv)

        assert np.allclose(du_pushed, du)      # u-component unchanged
        assert np.allclose(dv_pushed, -dv)     # v-component flipped

    def test_pushforward_differential_properties(self):
        """Test that pushforward represents dT = diag(1, -1)."""
        # Test multiple velocities
        velocities = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, -0.3], [-0.8, 0.7]])

        for vel in velocities:
            du, dv = vel[0], vel[1]
            du_pushed, dv_pushed = pushforward_velocity(du, dv)

            # Check differential matrix application
            dT = np.array([[1.0, 0.0], [0.0, -1.0]])
            expected = dT @ vel
            result = np.array([du_pushed, dv_pushed])

            assert np.allclose(result, expected)


class TestApplySeamIfNeeded:
    """Test seam handling logic."""

    def test_interior_point_unchanged(self):
        """Test that interior points are unchanged."""
        strip = Strip(w=1.0)
        u, v = 0.5, 0.3
        du, dv = 0.1, -0.2

        u_new, v_new, du_new, dv_new = apply_seam_if_needed(u, v, du, dv, strip)

        assert np.isclose(u_new, u)
        assert np.isclose(v_new, v)
        assert np.isclose(du_new, du)
        assert np.isclose(dv_new, dv)

    def test_top_seam_crossing(self):
        """Test crossing at top seam (v = +w)."""
        strip = Strip(w=1.0)
        u, v = 0.5, 1.1  # Slightly past top boundary
        du, dv = 0.1, 0.2

        u_new, v_new, du_new, dv_new = apply_seam_if_needed(u, v, du, dv, strip)

        # Should apply deck map and clamp to bottom
        expected_u = wrap_u(0.5 + np.pi, strip.period)
        assert np.isclose(u_new, expected_u)
        assert np.isclose(v_new, -1.0)  # Clamped to -w
        assert np.isclose(du_new, 0.1)  # u-velocity unchanged
        assert np.isclose(dv_new, -0.2) # v-velocity flipped

    def test_bottom_seam_crossing(self):
        """Test crossing at bottom seam (v = -w)."""
        strip = Strip(w=1.0)
        u, v = 1.2, -1.1  # Slightly past bottom boundary
        du, dv = -0.3, -0.1

        u_new, v_new, du_new, dv_new = apply_seam_if_needed(u, v, du, dv, strip)

        # Should apply deck map and clamp to top
        expected_u = wrap_u(1.2 + np.pi, strip.period)
        assert np.isclose(u_new, expected_u)
        assert np.isclose(v_new, 1.0)   # Clamped to +w
        assert np.isclose(du_new, -0.3) # u-velocity unchanged
        assert np.isclose(dv_new, 0.1)  # v-velocity flipped

    def test_array_seam_handling(self):
        """Test seam handling with arrays."""
        strip = Strip(w=1.0)
        u = np.array([0.5, 1.0, 1.5])
        v = np.array([0.3, 1.1, -1.2])  # Middle crosses top, last crosses bottom
        du = np.array([0.1, 0.2, -0.1])
        dv = np.array([0.05, 0.15, -0.25])

        u_new, v_new, du_new, dv_new = apply_seam_if_needed(u, v, du, dv, strip)

        # First point unchanged
        assert np.isclose(u_new[0], 0.5)
        assert np.isclose(v_new[0], 0.3)
        assert np.isclose(du_new[0], 0.1)
        assert np.isclose(dv_new[0], 0.05)

        # Second point crosses top seam
        assert np.isclose(u_new[1], wrap_u(1.0 + np.pi, strip.period))
        assert np.isclose(v_new[1], -1.0)
        assert np.isclose(du_new[1], 0.2)
        assert np.isclose(dv_new[1], -0.15)

        # Third point crosses bottom seam
        assert np.isclose(u_new[2], wrap_u(1.5 + np.pi, strip.period))
        assert np.isclose(v_new[2], 1.0)
        assert np.isclose(du_new[2], -0.1)
        assert np.isclose(dv_new[2], 0.25)


class TestIsOnSeam:
    """Test seam detection."""

    def test_interior_point_not_on_seam(self):
        """Test that interior points are not on seam."""
        strip = Strip(w=1.0)
        u, v = 0.5, 0.3
        assert not is_on_seam(u, v, strip)

    def test_top_seam_detection(self):
        """Test detection of points on top seam."""
        strip = Strip(w=1.0)
        u, v = 0.5, 1.0
        assert is_on_seam(u, v, strip)

    def test_bottom_seam_detection(self):
        """Test detection of points on bottom seam."""
        strip = Strip(w=1.0)
        u, v = 0.5, -1.0
        assert is_on_seam(u, v, strip)

    def test_seam_detection_tolerance(self):
        """Test seam detection with numerical tolerance."""
        strip = Strip(w=1.0)
        u, v = 0.5, 1.0 + 1e-13  # Very close to seam
        assert is_on_seam(u, v, strip, tolerance=1e-12)
        assert not is_on_seam(u, v, strip, tolerance=1e-14)

    def test_array_seam_detection(self):
        """Test seam detection for arrays."""
        strip = Strip(w=1.0)
        u = np.array([0.5, 1.0, 1.5])
        v = np.array([0.3, 1.0, -1.0])

        on_seam = is_on_seam(u, v, strip)
        expected = np.array([False, True, True])
        assert np.array_equal(on_seam, expected)


class TestSeamEquivalentPoints:
    """Test seam equivalence relationships."""

    def test_interior_point_unchanged(self):
        """Test that interior points return themselves."""
        strip = Strip(w=1.0)
        u, v = 0.5, 0.3
        u_equiv, v_equiv = seam_equivalent_points(u, v, strip)

        assert np.isclose(u_equiv, u)
        assert np.isclose(v_equiv, v)

    def test_seam_point_equivalence(self):
        """Test that seam points return their equivalents."""
        strip = Strip(w=1.0)
        u, v = 0.5, 1.0  # On top seam
        u_equiv, v_equiv = seam_equivalent_points(u, v, strip)

        expected_u = wrap_u(0.5 + np.pi, strip.period)
        assert np.isclose(u_equiv, expected_u)
        assert np.isclose(v_equiv, -1.0)

    def test_array_equivalence(self):
        """Test equivalence for arrays."""
        strip = Strip(w=1.0)
        u = np.array([0.5, 1.0])  # Interior and seam points
        v = np.array([0.3, 1.0])

        u_equiv, v_equiv = seam_equivalent_points(u, v, strip)

        # First point unchanged
        assert np.isclose(u_equiv[0], 0.5)
        assert np.isclose(v_equiv[0], 0.3)

        # Second point transformed
        assert np.isclose(u_equiv[1], wrap_u(1.0 + np.pi, strip.period))
        assert np.isclose(v_equiv[1], -1.0)


class TestDistanceOnQuotient:
    """Test distance computation on quotient space."""

    def test_interior_distance(self):
        """Test distance between interior points."""
        strip = Strip(w=1.0)
        u1, v1 = 0.5, 0.3
        u2, v2 = 1.0, 0.7

        dist = distance_on_quotient(u1, v1, u2, v2, strip)
        expected = np.sqrt((1.0 - 0.5)**2 + (0.7 - 0.3)**2)
        assert np.isclose(dist, expected)

    def test_seam_equivalent_distance(self):
        """Test distance to seam-equivalent point should be zero."""
        strip = Strip(w=1.0)
        u1, v1 = 0.5, 1.0  # On seam
        u2, v2 = deck_map(u1, v1, strip)

        dist = distance_on_quotient(u1, v1, u2, v2, strip)
        assert dist < 1e-10  # Should be essentially zero

    def test_shorter_path_via_deck_map(self):
        """Test that shorter path via deck map is chosen."""
        strip = Strip(w=1.0, period=2*np.pi)
        u1, v1 = 0.1, 0.5
        u2, v2 = 2*np.pi - 0.1, -0.5  # Close via deck map

        dist = distance_on_quotient(u1, v1, u2, v2, strip)

        # Direct distance would be large, deck map distance should be small
        direct_dist = np.sqrt((u2 - u1)**2 + (v2 - v1)**2)
        assert dist < direct_dist

    def test_array_distances(self):
        """Test distance computation for arrays."""
        strip = Strip(w=1.0)
        u1 = np.array([0.5, 1.0])
        v1 = np.array([0.3, 0.7])
        u2 = np.array([1.0, 1.5])
        v2 = np.array([0.7, 0.3])

        distances = distance_on_quotient(u1, v1, u2, v2, strip)

        assert len(distances) == 2
        assert all(d >= 0 for d in distances)


if __name__ == "__main__":
    pytest.main([__file__])