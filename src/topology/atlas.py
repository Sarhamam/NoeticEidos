"""Topology atlas for multiple quotient spaces.

Implements a unified framework for various 2D quotient topologies:
- Cylinder vs Möbius band (orientable vs non-orientable strips)
- Torus vs Klein bottle (orientable vs non-orientable rectangles)
- Sphere vs projective plane (orientable vs non-orientable spheres)

Each topology is defined by its fundamental domain and identification maps.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum

from .coords import wrap_u


class Orientability(Enum):
    """Topology orientability classification."""
    ORIENTABLE = "orientable"
    NON_ORIENTABLE = "non_orientable"


class TopologyType(Enum):
    """Supported topology types."""
    CYLINDER = "cylinder"
    MOBIUS = "mobius"
    TORUS = "torus"
    KLEIN = "klein"
    SPHERE = "sphere"
    PROJECTIVE = "projective"


@dataclass
class QuotientSpace(ABC):
    """Abstract base class for quotient space topologies."""

    @property
    @abstractmethod
    def orientability(self) -> Orientability:
        """Return orientability of the topology."""
        pass

    @property
    @abstractmethod
    def euler_characteristic(self) -> int:
        """Return Euler characteristic χ = V - E + F."""
        pass

    @property
    @abstractmethod
    def genus(self) -> int:
        """Return genus g (number of handles)."""
        pass

    @abstractmethod
    def identification_maps(self) -> List[Callable]:
        """Return list of identification maps defining the quotient."""
        pass

    @abstractmethod
    def fundamental_domain_bounds(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Return ((u_min, u_max), (v_min, v_max)) for fundamental domain."""
        pass

    @abstractmethod
    def apply_identifications(self, u: np.ndarray, v: np.ndarray,
                            du: np.ndarray, dv: np.ndarray) -> Tuple[np.ndarray, np.ndarray,
                                                                   np.ndarray, np.ndarray]:
        """Apply identification maps when crossing domain boundaries."""
        pass

    @abstractmethod
    def metric_compatibility_condition(self, g_fn: Callable,
                                     q: np.ndarray, tolerance: float = 1e-8) -> bool:
        """Check metric compatibility with all identification maps."""
        pass


@dataclass
class Strip(QuotientSpace):
    """Base class for strip-based topologies (cylinder/Möbius)."""
    w: float                    # Half-width of strip
    period: float = 2 * np.pi   # Period in u-direction

    def fundamental_domain_bounds(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Strip domain: [0, period) × [-w, w]."""
        return ((0.0, self.period), (-self.w, self.w))


@dataclass
class Cylinder(Strip):
    """Cylinder topology: [0,2π) × [-w,w] with (u,v) ~ (u,v) at boundaries."""

    @property
    def orientability(self) -> Orientability:
        return Orientability.ORIENTABLE

    @property
    def euler_characteristic(self) -> int:
        return 0  # χ = 2 - 2g = 2 - 2(0) = 2 for sphere, but cylinder is open

    @property
    def genus(self) -> int:
        return 0  # Topologically equivalent to plane

    def identification_maps(self) -> List[Callable]:
        """Cylinder identifications: only u-direction wrapping."""
        def u_wrap(u, v):
            return wrap_u(u, self.period), v
        return [u_wrap]

    def apply_identifications(self, u: np.ndarray, v: np.ndarray,
                            du: np.ndarray, dv: np.ndarray) -> Tuple[np.ndarray, np.ndarray,
                                                                   np.ndarray, np.ndarray]:
        """Handle cylinder boundary: only u-wrapping, v-clamping."""
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        du = np.asarray(du, dtype=float)
        dv = np.asarray(dv, dtype=float)

        # Wrap u-coordinate
        u_new = wrap_u(u, self.period)

        # Clamp v-coordinate and reflect velocity at boundaries
        v_new = np.clip(v, -self.w, self.w)
        dv_new = np.where((v <= -self.w) | (v >= self.w), -dv, dv)

        return u_new, v_new, du, dv_new

    def metric_compatibility_condition(self, g_fn: Callable,
                                     q: np.ndarray, tolerance: float = 1e-8) -> bool:
        """Cylinder metric only needs u-periodicity."""
        u, v = q[0], q[1]
        g_orig = g_fn(q)

        # Check u-periodicity: g(u + period, v) = g(u, v)
        u_wrapped = wrap_u(u + self.period, self.period)
        q_wrapped = np.array([u_wrapped, v])
        g_wrapped = g_fn(q_wrapped)

        error = np.max(np.abs(g_wrapped - g_orig))
        return error <= tolerance


@dataclass
class MobiusBand(Strip):
    """Möbius band topology: [0,2π) × [-w,w] with deck map T(u,v) = (u+π,-v) mod 2π.

    The Möbius band has period 2π in the u-direction. The deck map shifts u by π
    (half the period) and reflects v. This creates the characteristic twist that
    makes the Möbius band non-orientable.

    Key invariant: After going around twice (u → u+2π), we return to the original point.
    """

    @property
    def orientability(self) -> Orientability:
        return Orientability.NON_ORIENTABLE

    @property
    def euler_characteristic(self) -> int:
        return 0  # χ = 0 for Möbius band

    @property
    def genus(self) -> int:
        return 0  # Genus undefined for non-orientable; crosscap number = 1

    def identification_maps(self) -> List[Callable]:
        """Möbius identification: deck map T(u,v) = (u+π,-v) mod period."""
        def mobius_deck_map(u, v):
            # Shift by π (half period) and reflect v
            return wrap_u(u + np.pi, self.period), -v
        return [mobius_deck_map]

    def apply_identifications(self, u: np.ndarray, v: np.ndarray,
                            du: np.ndarray, dv: np.ndarray) -> Tuple[np.ndarray, np.ndarray,
                                                                   np.ndarray, np.ndarray]:
        """Handle Möbius seam: apply deck map at v = ±w."""
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        du = np.asarray(du, dtype=float)
        dv = np.asarray(dv, dtype=float)

        # Apply deck map at boundaries
        at_top = v >= self.w
        at_bottom = v <= -self.w

        if np.any(at_top):
            u[at_top] = wrap_u(u[at_top] + np.pi, self.period)
            v[at_top] = -self.w
            du[at_top] = du[at_top]  # u-velocity unchanged
            dv[at_top] = -dv[at_top]  # v-velocity flipped

        if np.any(at_bottom):
            u[at_bottom] = wrap_u(u[at_bottom] + np.pi, self.period)
            v[at_bottom] = self.w
            du[at_bottom] = du[at_bottom]  # u-velocity unchanged
            dv[at_bottom] = -dv[at_bottom]  # v-velocity flipped

        return u, v, du, dv

    def metric_compatibility_condition(self, g_fn: Callable,
                                     q: np.ndarray, tolerance: float = 1e-8) -> bool:
        """Möbius metric must satisfy g(u+π,-v) = dT^T g(u,v) dT."""
        u, v = q[0], q[1]
        g_orig = g_fn(q)

        # Apply deck map
        u_deck = wrap_u(u + np.pi, self.period)
        v_deck = -v
        q_deck = np.array([u_deck, v_deck])
        g_deck = g_fn(q_deck)

        # Deck map differential dT = diag(1, -1)
        dT = np.array([[1.0, 0.0], [0.0, -1.0]])
        g_expected = dT.T @ g_orig @ dT

        error = np.max(np.abs(g_deck - g_expected))
        return error <= tolerance


@dataclass
class Rectangle(QuotientSpace):
    """Base class for rectangular fundamental domains (torus/Klein)."""
    width: float                # Width in u-direction
    height: float               # Height in v-direction

    def fundamental_domain_bounds(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Rectangle domain: [0, width) × [0, height)."""
        return ((0.0, self.width), (0.0, self.height))


@dataclass
class Torus(Rectangle):
    """Torus topology: [0,a) × [0,b) with (u,v) ~ (u,v) at opposite edges."""

    @property
    def orientability(self) -> Orientability:
        return Orientability.ORIENTABLE

    @property
    def euler_characteristic(self) -> int:
        return 0  # χ = 2 - 2g = 2 - 2(1) = 0

    @property
    def genus(self) -> int:
        return 1  # Torus has genus 1

    def identification_maps(self) -> List[Callable]:
        """Torus identifications: wrap both u and v."""
        def u_wrap(u, v):
            return np.mod(u, self.width), v
        def v_wrap(u, v):
            return u, np.mod(v, self.height)
        return [u_wrap, v_wrap]

    def apply_identifications(self, u: np.ndarray, v: np.ndarray,
                            du: np.ndarray, dv: np.ndarray) -> Tuple[np.ndarray, np.ndarray,
                                                                   np.ndarray, np.ndarray]:
        """Handle torus wrapping: both directions wrap with unchanged velocities."""
        u_new = np.mod(u, self.width)
        v_new = np.mod(v, self.height)
        return u_new, v_new, du, dv

    def metric_compatibility_condition(self, g_fn: Callable,
                                     q: np.ndarray, tolerance: float = 1e-8) -> bool:
        """Torus metric must be doubly periodic."""
        u, v = q[0], q[1]
        g_orig = g_fn(q)

        # Check u-periodicity
        q_u_shifted = np.array([np.mod(u + self.width, self.width), v])
        g_u_shifted = g_fn(q_u_shifted)

        # Check v-periodicity
        q_v_shifted = np.array([u, np.mod(v + self.height, self.height)])
        g_v_shifted = g_fn(q_v_shifted)

        error_u = np.max(np.abs(g_u_shifted - g_orig))
        error_v = np.max(np.abs(g_v_shifted - g_orig))

        return max(error_u, error_v) <= tolerance


@dataclass
class KleinBottle(Rectangle):
    """Klein bottle topology: [0,2π) × [0,2π) with two identification maps.

    The Klein bottle is constructed from a rectangle with:
    1. Horizontal identification: (0, v) ~ (2π, v) (regular gluing)
    2. Vertical identification: (u, 0) ~ (2π-u, 2π) (Möbius twist)

    This creates a non-orientable surface with crosscap number 2.
    """

    @property
    def orientability(self) -> Orientability:
        return Orientability.NON_ORIENTABLE

    @property
    def euler_characteristic(self) -> int:
        return 0  # χ = 0 for Klein bottle

    @property
    def genus(self) -> int:
        return 0  # Genus undefined for non-orientable; crosscap number = 2

    def identification_maps(self) -> List[Callable]:
        """Klein bottle identifications: horizontal gluing and vertical Möbius twist."""
        def horizontal_gluing(u, v):
            # (0, v) ~ (2π, v): regular cylindrical gluing
            return np.mod(u, self.width), v

        def vertical_mobius_twist(u, v):
            # (u, 0) ~ (2π-u, 2π): Möbius twist at top/bottom
            # When v crosses 0 or 2π boundary, reflect u
            return self.width - u, np.mod(v, self.height)

        return [horizontal_gluing, vertical_mobius_twist]

    def apply_identifications(self, u: np.ndarray, v: np.ndarray,
                            du: np.ndarray, dv: np.ndarray) -> Tuple[np.ndarray, np.ndarray,
                                                                   np.ndarray, np.ndarray]:
        """Handle Klein bottle boundaries with correct identification maps."""
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        du = np.asarray(du, dtype=float)
        dv = np.asarray(dv, dtype=float)

        # Handle horizontal wrapping (cylindrical gluing): (0, v) ~ (2π, v)
        u = np.mod(u, self.width)

        # Handle vertical boundary with Möbius twist: (u, 0) ~ (2π-u, 2π)
        # When v crosses 0 or height boundary
        v_low = v < 0
        v_high = v >= self.height

        if np.any(v_low):
            # Bottom boundary: map (u, 0) -> (2π-u, 2π)
            u[v_low] = self.width - u[v_low]
            v[v_low] = self.height + v[v_low]  # Wrap to top
            du[v_low] = -du[v_low]  # u-velocity flipped due to reflection

        if np.any(v_high):
            # Top boundary: map (u, 2π) -> (2π-u, 0)
            u[v_high] = self.width - u[v_high]
            v[v_high] = v[v_high] - self.height  # Wrap to bottom
            du[v_high] = -du[v_high]  # u-velocity flipped due to reflection

        return u, v, du, dv

    def metric_compatibility_condition(self, g_fn: Callable,
                                     q: np.ndarray, tolerance: float = 1e-8) -> bool:
        """Klein bottle metric must satisfy both horizontal periodicity and vertical twist.

        For Klein bottle, we need:
        1. Horizontal: g(0, v) = g(2π, v) (regular periodicity)
        2. Vertical: g(u, 0) must be compatible with g(2π-u, 2π) under twist
        """
        u, v = q[0], q[1]
        g_orig = g_fn(q)

        # Check horizontal periodicity: (0, v) ~ (2π, v)
        q_u_shifted = np.array([np.mod(u + self.width, self.width), v])
        g_u_shifted = g_fn(q_u_shifted)
        error_horizontal = np.max(np.abs(g_u_shifted - g_orig))

        # Check vertical Möbius twist: (u, 0) ~ (2π-u, 2π)
        # The identification map has differential dφ = diag(-1, 1)
        u_twisted = self.width - u
        v_twisted = np.mod(v + self.height, self.height)
        q_twisted = np.array([u_twisted, v_twisted])
        g_twisted = g_fn(q_twisted)

        # Differential of the twist map: reflects u but not v
        dφ = np.array([[-1.0, 0.0], [0.0, 1.0]])
        g_expected = dφ.T @ g_orig @ dφ
        # This gives: g₁₁ → g₁₁, g₂₂ → g₂₂, g₁₂ → -g₁₂

        error_twist = np.max(np.abs(g_twisted - g_expected))

        return max(error_horizontal, error_twist) <= tolerance


@dataclass
class Sphere(QuotientSpace):
    """Sphere S² topology using spherical coordinates.

    The sphere is parameterized with spherical coordinates (θ, φ) where:
    - θ ∈ [0, π] is the colatitude (0 = north pole, π = south pole)
    - φ ∈ [0, 2π) is the azimuth

    Identifications:
    1. Azimuthal periodicity: (θ, 0) ~ (θ, 2π) for all θ
    2. North pole degeneracy: (0, φ) ~ (0, 0) for all φ
    3. South pole degeneracy: (π, φ) ~ (π, 0) for all φ
    """
    radius: float = 1.0

    @property
    def orientability(self) -> Orientability:
        return Orientability.ORIENTABLE

    @property
    def euler_characteristic(self) -> int:
        return 2  # χ = 2 for S²

    @property
    def genus(self) -> int:
        return 0  # Sphere has genus 0

    def fundamental_domain_bounds(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Sphere in spherical coordinates: θ ∈ [0,π], φ ∈ [0,2π)."""
        return ((0.0, np.pi), (0.0, 2*np.pi))

    def identification_maps(self) -> List[Callable]:
        """Sphere identifications: azimuthal periodicity and pole degeneracies."""
        def azimuthal_periodicity(theta, phi):
            # (θ, 0) ~ (θ, 2π)
            return theta, np.mod(phi, 2*np.pi)

        def north_pole_degeneracy(theta, phi):
            # At north pole (θ=0): all φ values map to same point
            if np.abs(theta) < 1e-10:
                return 0.0, 0.0
            return theta, phi

        def south_pole_degeneracy(theta, phi):
            # At south pole (θ=π): all φ values map to same point
            if np.abs(theta - np.pi) < 1e-10:
                return np.pi, 0.0
            return theta, phi

        return [azimuthal_periodicity, north_pole_degeneracy, south_pole_degeneracy]

    def apply_identifications(self, theta: np.ndarray, phi: np.ndarray,
                            dtheta: np.ndarray, dphi: np.ndarray) -> Tuple[np.ndarray, np.ndarray,
                                                                          np.ndarray, np.ndarray]:
        """Handle sphere boundaries with azimuthal periodicity and pole degeneracies."""
        theta = np.asarray(theta, dtype=float)
        phi = np.asarray(phi, dtype=float)
        dtheta = np.asarray(dtheta, dtype=float)
        dphi = np.asarray(dphi, dtype=float)

        # Azimuthal periodicity
        phi = np.mod(phi, 2*np.pi)

        # Handle poles: at poles, φ is undefined, so set to 0
        at_north_pole = np.abs(theta) < 1e-10
        at_south_pole = np.abs(theta - np.pi) < 1e-10

        if np.any(at_north_pole):
            phi[at_north_pole] = 0.0
            dphi[at_north_pole] = 0.0  # No azimuthal motion at poles

        if np.any(at_south_pole):
            phi[at_south_pole] = 0.0
            dphi[at_south_pole] = 0.0  # No azimuthal motion at poles

        # Clamp theta to valid range
        theta = np.clip(theta, 0.0, np.pi)

        return theta, phi, dtheta, dphi

    def metric_compatibility_condition(self, g_fn: Callable,
                                     q: np.ndarray, tolerance: float = 1e-8) -> bool:
        """Sphere metric must satisfy azimuthal periodicity and be regular at poles.

        For S² with spherical coordinates:
        - Azimuthal periodicity: g(θ, 0) = g(θ, 2π)
        - Regularity at poles: metric should be well-defined (though coordinates degenerate)
        """
        theta, phi = q[0], q[1]
        g_orig = g_fn(q)

        # Check azimuthal periodicity (except at poles)
        if np.abs(theta) > 1e-10 and np.abs(theta - np.pi) > 1e-10:
            q_phi_shifted = np.array([theta, np.mod(phi + 2*np.pi, 2*np.pi)])
            g_phi_shifted = g_fn(q_phi_shifted)
            error_phi = np.max(np.abs(g_phi_shifted - g_orig))
        else:
            error_phi = 0.0  # Skip check at poles

        return error_phi <= tolerance


@dataclass
class ProjectivePlane(QuotientSpace):
    """Real projective plane ℝP² via spherical coordinates with antipodal identification.

    The projective plane is constructed from a hemisphere [0, π/2] × [0, 2π) with
    antipodal identification on the equator: (π/2, φ) ~ (π/2, φ+π).

    This is topologically equivalent to a sphere with antipodal points identified:
    (θ, φ) ~ (π-θ, φ+π) on the full sphere.
    """

    @property
    def orientability(self) -> Orientability:
        return Orientability.NON_ORIENTABLE

    @property
    def euler_characteristic(self) -> int:
        return 1  # χ = 1 for ℝP²

    @property
    def genus(self) -> int:
        return 0  # Genus undefined for non-orientable; crosscap number = 1

    def fundamental_domain_bounds(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Projective plane as hemisphere: θ ∈ [0, π/2], φ ∈ [0, 2π)."""
        return ((0.0, np.pi/2), (0.0, 2*np.pi))

    def identification_maps(self) -> List[Callable]:
        """Projective plane: antipodal identification on equator."""
        def antipodal_on_equator(theta, phi):
            # At equator (θ = π/2): (π/2, φ) ~ (π/2, φ+π)
            return np.pi/2, np.mod(phi + np.pi, 2*np.pi)

        def azimuthal_periodicity(theta, phi):
            # Standard φ periodicity: (θ, 0) ~ (θ, 2π)
            return theta, np.mod(phi, 2*np.pi)

        return [antipodal_on_equator, azimuthal_periodicity]

    def apply_identifications(self, theta: np.ndarray, phi: np.ndarray,
                            dtheta: np.ndarray, dphi: np.ndarray) -> Tuple[np.ndarray, np.ndarray,
                                                                          np.ndarray, np.ndarray]:
        """Handle projective plane boundaries with antipodal identification.

        Uses spherical coordinates: θ (colatitude), φ (azimuth).
        """
        theta = np.asarray(theta, dtype=float)
        phi = np.asarray(phi, dtype=float)
        dtheta = np.asarray(dtheta, dtype=float)
        dphi = np.asarray(dphi, dtype=float)

        # Handle azimuthal periodicity: φ ∈ [0, 2π)
        phi = np.mod(phi, 2*np.pi)

        # Handle equatorial antipodal identification
        # When θ > π/2, reflect through antipodal point
        beyond_equator = theta > np.pi/2

        if np.any(beyond_equator):
            # Antipodal map: (θ, φ) → (π-θ, φ+π)
            theta[beyond_equator] = np.pi - theta[beyond_equator]
            phi[beyond_equator] = np.mod(phi[beyond_equator] + np.pi, 2*np.pi)
            dtheta[beyond_equator] = -dtheta[beyond_equator]  # θ-velocity reflected
            # dphi stays the same due to the symmetry of the identification

        return theta, phi, dtheta, dphi

    def metric_compatibility_condition(self, g_fn: Callable,
                                     q: np.ndarray, tolerance: float = 1e-8) -> bool:
        """Projective plane metric must satisfy antipodal compatibility.

        For ℝP² with spherical coordinates:
        - Azimuthal periodicity: g(θ, 0) = g(θ, 2π)
        - Antipodal at equator: g(π/2, φ) compatible with g(π/2, φ+π)
        """
        theta, phi = q[0], q[1]
        g_orig = g_fn(q)

        # Check azimuthal periodicity
        q_phi_shifted = np.array([theta, np.mod(phi + 2*np.pi, 2*np.pi)])
        g_phi_shifted = g_fn(q_phi_shifted)
        error_phi = np.max(np.abs(g_phi_shifted - g_orig))

        # Check antipodal compatibility at equator
        if np.abs(theta - np.pi/2) < 0.1:  # Near equator
            # Antipodal point on equator
            q_antipodal = np.array([np.pi/2, np.mod(phi + np.pi, 2*np.pi)])
            g_antipodal = g_fn(q_antipodal)

            # For the projective plane antipodal map at equator
            # The differential preserves the metric structure
            error_antipodal = np.max(np.abs(g_antipodal - g_orig))
        else:
            error_antipodal = 0.0

        return max(error_phi, error_antipodal) <= tolerance


class TopologyAtlas:
    """Atlas of quotient topologies for geometric ML."""

    def __init__(self):
        """Initialize topology atlas with supported surfaces."""
        self._topologies = {}
        self._register_standard_topologies()

    def _register_standard_topologies(self):
        """Register standard 2D quotient topologies."""
        # Strip-based topologies
        self._topologies[TopologyType.CYLINDER] = lambda **kwargs: Cylinder(**kwargs)
        self._topologies[TopologyType.MOBIUS] = lambda **kwargs: MobiusBand(**kwargs)

        # Rectangle-based topologies
        self._topologies[TopologyType.TORUS] = lambda **kwargs: Torus(**kwargs)
        self._topologies[TopologyType.KLEIN] = lambda **kwargs: KleinBottle(**kwargs)

        # Sphere-based topologies
        self._topologies[TopologyType.SPHERE] = lambda **kwargs: Sphere(**kwargs)
        self._topologies[TopologyType.PROJECTIVE] = lambda **kwargs: ProjectivePlane(**kwargs)

    def create_topology(self, topology_type: TopologyType, **kwargs) -> QuotientSpace:
        """Create topology of specified type with parameters."""
        if topology_type not in self._topologies:
            raise ValueError(f"Unsupported topology type: {topology_type}")

        return self._topologies[topology_type](**kwargs)

    def list_topologies(self) -> List[TopologyType]:
        """List all available topology types."""
        return list(self._topologies.keys())

    def get_orientable_pairs(self) -> List[Tuple[TopologyType, TopologyType]]:
        """Get pairs of (orientable, non-orientable) topologies."""
        return [
            (TopologyType.CYLINDER, TopologyType.MOBIUS),
            (TopologyType.TORUS, TopologyType.KLEIN),
            (TopologyType.SPHERE, TopologyType.PROJECTIVE)
        ]

    def topology_comparison_matrix(self) -> Dict[str, Dict[TopologyType, Any]]:
        """Create comparison matrix of topological properties."""
        matrix = {}

        for topo_type in self._topologies:
            # Create default instance for property extraction
            if topo_type in [TopologyType.CYLINDER, TopologyType.MOBIUS]:
                topo = self.create_topology(topo_type, w=1.0)
            elif topo_type in [TopologyType.TORUS, TopologyType.KLEIN]:
                topo = self.create_topology(topo_type, width=2*np.pi, height=2*np.pi)
            elif topo_type in [TopologyType.PROJECTIVE]:
                topo = self.create_topology(topo_type)
            else:  # Sphere/Projective
                topo = self.create_topology(topo_type, radius=1.0)

            matrix[topo_type] = {
                "orientable": topo.orientability == Orientability.ORIENTABLE,
                "euler_char": topo.euler_characteristic,
                "genus": topo.genus,
                "fundamental_domain": topo.fundamental_domain_bounds(),
                "num_identifications": len(topo.identification_maps())
            }

        return matrix

    def detect_orientability(self, topology: QuotientSpace) -> Dict[str, Any]:
        """Detect orientability through identification map analysis.

        Algorithm:
        1. Compute Jacobian determinants of all identification maps
        2. Check for orientation-reversing maps (det < 0)
        3. Verify consistency around non-contractible loops

        Returns:
            Dict with orientability status and mathematical evidence
        """
        id_maps = topology.identification_maps()
        results = {
            "topology": type(topology).__name__,
            "declared_orientability": topology.orientability.value,
            "identification_maps": [],
            "has_orientation_reversing": False,
            "is_orientable": True
        }

        # Test points for evaluating identification maps
        (u_min, u_max), (v_min, v_max) = topology.fundamental_domain_bounds()
        test_u = (u_min + u_max) / 2
        test_v = (v_min + v_max) / 2

        for i, id_map in enumerate(id_maps):
            # Compute numerical Jacobian
            eps = 1e-8
            u0, v0 = id_map(test_u, test_v)

            # Partial derivatives
            u_plus_eps, v_plus_eps = id_map(test_u + eps, test_v)
            du_du = (u_plus_eps - u0) / eps
            dv_du = (v_plus_eps - v0) / eps

            u_plus_eps, v_plus_eps = id_map(test_u, test_v + eps)
            du_dv = (u_plus_eps - u0) / eps
            dv_dv = (v_plus_eps - v0) / eps

            # Jacobian determinant
            det_jacobian = du_du * dv_dv - du_dv * dv_du

            map_info = {
                "map_index": i,
                "jacobian_det": float(det_jacobian),
                "preserves_orientation": det_jacobian > 0
            }
            results["identification_maps"].append(map_info)

            if det_jacobian < 0:
                results["has_orientation_reversing"] = True
                results["is_orientable"] = False

        # Verify consistency with declared orientability
        results["consistent"] = (
            (results["is_orientable"] and topology.orientability == Orientability.ORIENTABLE) or
            (not results["is_orientable"] and topology.orientability == Orientability.NON_ORIENTABLE)
        )

        return results

    def validate_metric_on_topology(self, metric_fn: Callable,
                                   topology: QuotientSpace,
                                   test_points: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Validate metric compatibility across a topology."""
        if test_points is None:
            # Generate default test points in fundamental domain
            (u_min, u_max), (v_min, v_max) = topology.fundamental_domain_bounds()
            u_test = np.linspace(u_min, u_max, 10, endpoint=False)
            v_test = np.linspace(v_min, v_max, 10, endpoint=False)
            U, V = np.meshgrid(u_test, v_test)
            test_points = np.column_stack([U.flatten(), V.flatten()])

        results = {
            "topology_type": type(topology).__name__,
            "orientable": topology.orientability == Orientability.ORIENTABLE,
            "n_test_points": len(test_points),
            "compatible_points": 0,
            "incompatible_points": 0,
            "errors": []
        }

        for point in test_points:
            try:
                is_compatible = topology.metric_compatibility_condition(metric_fn, point)
                if is_compatible:
                    results["compatible_points"] += 1
                else:
                    results["incompatible_points"] += 1
            except Exception as e:
                results["errors"].append(f"Point {point}: {e}")

        results["compatibility_rate"] = results["compatible_points"] / len(test_points)
        results["valid"] = results["incompatible_points"] == 0 and len(results["errors"]) == 0

        return results


# Global topology atlas instance
topology_atlas = TopologyAtlas()


def create_topology(topology_type: TopologyType, **kwargs) -> QuotientSpace:
    """Convenience function to create topology from atlas."""
    return topology_atlas.create_topology(topology_type, **kwargs)


def get_orientability_pairs() -> List[Tuple[QuotientSpace, QuotientSpace]]:
    """Get pairs of orientable/non-orientable topologies for comparison."""
    pairs = []

    # Cylinder vs Möbius
    pairs.append((
        create_topology(TopologyType.CYLINDER, w=1.0),
        create_topology(TopologyType.MOBIUS, w=1.0)
    ))

    # Torus vs Klein bottle
    pairs.append((
        create_topology(TopologyType.TORUS, width=2*np.pi, height=2*np.pi),
        create_topology(TopologyType.KLEIN, width=2*np.pi, height=2*np.pi)
    ))

    # Sphere vs Projective plane
    pairs.append((
        create_topology(TopologyType.SPHERE, radius=1.0),
        create_topology(TopologyType.PROJECTIVE)
    ))

    return pairs