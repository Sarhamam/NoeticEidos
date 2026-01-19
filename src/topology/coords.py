"""Coordinate handling for topological quotients.

Implements deck maps, wrapping, and velocity pushforward for the Möbius band
quotient topology. The Möbius band is realized as [0,2π) × [-w,w] with
seam identification T(u,v) = (u+π, -v) mod 2π.
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class Strip:
    """Configuration for the fundamental strip of Möbius band quotient.

    The Möbius band is parameterized as [0, period) × [-w, w] with
    seam identification at v = ±w via the deck map T(u,v) = (u+π, -v).

    Parameters
    ----------
    w : float
        Half-width of the strip in v-direction
    period : float
        Period in u-direction (default 2π)
    """
    w: float
    period: float = 2 * np.pi


def wrap_u(u: np.ndarray, period: float = 2 * np.pi) -> np.ndarray:
    """Wrap u-coordinate to [0, period).

    Parameters
    ----------
    u : ndarray
        u-coordinates to wrap
    period : float
        Period for wrapping (default 2π)

    Returns
    -------
    u_wrapped : ndarray
        u-coordinates wrapped to [0, period)
    """
    u = np.asarray(u)
    return np.mod(u, period)


def deck_map(u: np.ndarray, v: np.ndarray, strip: Strip) -> Tuple[np.ndarray, np.ndarray]:
    """Apply deck map T(u,v) = (u+π, -v) for Möbius band seam identification.

    The deck map implements the fundamental identification that creates
    the Möbius band topology from the strip.

    Parameters
    ----------
    u, v : ndarray
        Coordinates on the strip
    strip : Strip
        Strip configuration

    Returns
    -------
    u_mapped, v_mapped : ndarray
        Coordinates after deck map application

    Notes
    -----
    The deck map differential is dT = diag(1, -1).
    """
    u = np.asarray(u)
    v = np.asarray(v)

    # Apply deck map: T(u,v) = (u + π, -v)
    u_mapped = wrap_u(u + np.pi, strip.period)
    v_mapped = -v

    return u_mapped, v_mapped


def pushforward_velocity(du: np.ndarray, dv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Push forward velocities under deck map differential dT = diag(1, -1).

    Parameters
    ----------
    du, dv : ndarray
        Velocity components in (u,v) directions

    Returns
    -------
    du_pushed, dv_pushed : ndarray
        Velocity components after pushforward

    Notes
    -----
    The deck map differential is dT = [[1, 0], [0, -1]], so:
    - du component unchanged
    - dv component flipped sign
    """
    du = np.asarray(du)
    dv = np.asarray(dv)

    # Apply differential dT = diag(1, -1)
    du_pushed = du  # u-component unchanged
    dv_pushed = -dv  # v-component sign flip

    return du_pushed, dv_pushed


def apply_seam_if_needed(u: np.ndarray, v: np.ndarray,
                        du: np.ndarray, dv: np.ndarray,
                        strip: Strip) -> Tuple[np.ndarray, np.ndarray,
                                             np.ndarray, np.ndarray]:
    """Apply seam identification when crossing strip boundaries.

    When a point reaches v = ±w, apply the deck map to transition
    to the opposite side of the strip with coordinates and velocities
    properly transformed.

    Parameters
    ----------
    u, v : ndarray
        Position coordinates
    du, dv : ndarray
        Velocity components
    strip : Strip
        Strip configuration

    Returns
    -------
    u_new, v_new, du_new, dv_new : ndarray
        Updated coordinates and velocities after seam handling

    Notes
    -----
    Seam crossing rules:
    - At v = +w: (u, +w, du, dv) → (u+π, -w, du, -dv)
    - At v = -w: (u, -w, du, dv) → (u+π, +w, du, -dv)
    """
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    du = np.asarray(du, dtype=float)
    dv = np.asarray(dv, dtype=float)

    # Ensure arrays are at least 1D for indexing
    u_shape = u.shape
    u_flat = u.flatten()
    v_flat = v.flatten()
    du_flat = du.flatten()
    dv_flat = dv.flatten()

    # Check for seam crossings
    at_top_seam = v_flat >= strip.w
    at_bottom_seam = v_flat <= -strip.w

    # Apply seam transformations
    # Top seam: v ≥ w → apply deck map and clamp to -w
    if np.any(at_top_seam):
        u_flat[at_top_seam] = wrap_u(u_flat[at_top_seam] + np.pi, strip.period)
        v_flat[at_top_seam] = -strip.w
        du_flat[at_top_seam] = du_flat[at_top_seam]  # u-velocity unchanged
        dv_flat[at_top_seam] = -dv_flat[at_top_seam]  # v-velocity flipped

    # Bottom seam: v ≤ -w → apply deck map and clamp to +w
    if np.any(at_bottom_seam):
        u_flat[at_bottom_seam] = wrap_u(u_flat[at_bottom_seam] + np.pi, strip.period)
        v_flat[at_bottom_seam] = strip.w
        du_flat[at_bottom_seam] = du_flat[at_bottom_seam]  # u-velocity unchanged
        dv_flat[at_bottom_seam] = -dv_flat[at_bottom_seam]  # v-velocity flipped

    # Reshape back to original shape
    u_new = u_flat.reshape(u_shape)
    v_new = v_flat.reshape(u_shape)
    du_new = du_flat.reshape(u_shape)
    dv_new = dv_flat.reshape(u_shape)

    return u_new, v_new, du_new, dv_new


def is_on_seam(u: np.ndarray, v: np.ndarray, strip: Strip,
               tolerance: float = 1e-12) -> np.ndarray:
    """Check if points are on the seam boundaries.

    Parameters
    ----------
    u, v : ndarray
        Coordinates to check
    strip : Strip
        Strip configuration
    tolerance : float
        Numerical tolerance for seam detection

    Returns
    -------
    on_seam : ndarray of bool
        True for points on seam boundaries
    """
    v = np.asarray(v)
    return (np.abs(v - strip.w) < tolerance) | (np.abs(v + strip.w) < tolerance)


def seam_equivalent_points(u: np.ndarray, v: np.ndarray,
                          strip: Strip) -> Tuple[np.ndarray, np.ndarray]:
    """Get seam-equivalent points under deck map.

    For points on the seam, returns the equivalent point on the
    opposite side. For interior points, returns the original coordinates.

    Parameters
    ----------
    u, v : ndarray
        Original coordinates
    strip : Strip
        Strip configuration

    Returns
    -------
    u_equiv, v_equiv : ndarray
        Seam-equivalent coordinates
    """
    u = np.asarray(u)
    v = np.asarray(v)

    on_seam = is_on_seam(u, v, strip)

    u_equiv = u.copy()
    v_equiv = v.copy()

    if np.any(on_seam):
        u_equiv[on_seam], v_equiv[on_seam] = deck_map(u[on_seam], v[on_seam], strip)

    return u_equiv, v_equiv


def distance_on_quotient(u1: np.ndarray, v1: np.ndarray,
                        u2: np.ndarray, v2: np.ndarray,
                        strip: Strip) -> np.ndarray:
    """Compute distance between points accounting for quotient topology.

    Parameters
    ----------
    u1, v1 : ndarray
        First set of coordinates
    u2, v2 : ndarray
        Second set of coordinates
    strip : Strip
        Strip configuration

    Returns
    -------
    distance : ndarray
        Euclidean distances accounting for seam identification
    """
    u1, v1 = np.asarray(u1), np.asarray(v1)
    u2, v2 = np.asarray(u2), np.asarray(v2)

    # Direct distance
    du_direct = u2 - u1
    dv_direct = v2 - v1
    dist_direct = np.sqrt(du_direct**2 + dv_direct**2)

    # Distance via deck map
    u2_deck, v2_deck = deck_map(u2, v2, strip)
    du_deck = u2_deck - u1
    dv_deck = v2_deck - v1

    # Handle u-coordinate wrapping
    du_deck = np.minimum(np.abs(du_deck), strip.period - np.abs(du_deck))

    dist_deck = np.sqrt(du_deck**2 + dv_deck**2)

    # Return minimum distance
    return np.minimum(dist_direct, dist_deck)