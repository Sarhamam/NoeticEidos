"""Symplectic geodesic integration with seam awareness for Möbius band.

Implements geodesic integration on the Möbius band quotient using symplectic
leapfrog integration with proper seam handling. Tracks energy conservation
and handles seam crossings via deck map transformations.
"""

import numpy as np
import warnings
from typing import Callable, Tuple, Optional, Dict, Any, List
from .coords import Strip, apply_seam_if_needed


class GeodesicIntegrationError(Exception):
    """Raised when geodesic integration encounters errors."""
    pass


def christoffel(g: np.ndarray, du_g: np.ndarray, dv_g: np.ndarray) -> np.ndarray:
    """Compute Christoffel symbols Γᵏᵢⱼ from metric and its derivatives.

    Parameters
    ----------
    g : ndarray, shape (2, 2)
        Metric tensor at current point
    du_g : ndarray, shape (2, 2)
        Partial derivative ∂g/∂u
    dv_g : ndarray, shape (2, 2)
        Partial derivative ∂g/∂v

    Returns
    -------
    Gamma : ndarray, shape (2, 2, 2)
        Christoffel symbols Γᵏᵢⱼ where first index is k, second is i, third is j

    Notes
    -----
    Christoffel symbols are computed as:
    Γᵏᵢⱼ = (1/2) gᵏˡ (∂gᵢˡ/∂xʲ + ∂gⱼˡ/∂xⁱ - ∂gᵢⱼ/∂xˡ)
    """
    g = np.asarray(g)
    du_g = np.asarray(du_g)
    dv_g = np.asarray(dv_g)

    if g.shape != (2, 2):
        raise ValueError("Metric must be 2×2 matrix")
    if du_g.shape != (2, 2) or dv_g.shape != (2, 2):
        raise ValueError("Metric derivatives must be 2×2 matrices")

    # Compute inverse metric
    try:
        g_inv = np.linalg.inv(g)
    except np.linalg.LinAlgError:
        raise GeodesicIntegrationError("Metric is singular, cannot compute Christoffel symbols")

    # Gradient array: dg[l][i][j] = ∂g_ij/∂x^l
    dg = np.array([du_g, dv_g])  # shape (2, 2, 2)

    # Initialize Christoffel symbols Gamma[k][i][j]
    Gamma = np.zeros((2, 2, 2))

    for k in range(2):
        for i in range(2):
            for j in range(2):
                # Γᵏᵢⱼ = (1/2) Σₗ gᵏˡ (∂gᵢˡ/∂xʲ + ∂gⱼˡ/∂xⁱ - ∂gᵢⱼ/∂xˡ)
                term = 0.0
                for l in range(2):
                    term += g_inv[k, l] * (dg[j, i, l] + dg[i, j, l] - dg[l, i, j])
                Gamma[k, i, j] = 0.5 * term

    return Gamma


def geodesic_acceleration(q: np.ndarray, v: np.ndarray,
                         g_fn: Callable[[np.ndarray], np.ndarray],
                         grad_g_fn: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """Compute geodesic acceleration using Christoffel symbols.

    Parameters
    ----------
    q : ndarray, shape (2,)
        Position coordinates [u, v]
    v : ndarray, shape (2,)
        Velocity vector [du/dt, dv/dt]
    g_fn : callable
        Metric function g(q) returning 2×2 metric tensor
    grad_g_fn : callable
        Gradient function returning (∂g/∂u, ∂g/∂v)

    Returns
    -------
    accel : ndarray, shape (2,)
        Geodesic acceleration d²q/dt²

    Notes
    -----
    Geodesic equation: d²qᵏ/dt² = -Γᵏᵢⱼ (dqⁱ/dt)(dqʲ/dt)
    """
    q = np.asarray(q)
    v = np.asarray(v)

    if q.shape != (2,) or v.shape != (2,):
        raise ValueError("Position and velocity must be 2D vectors")

    # Get metric and its derivatives
    g = g_fn(q)
    du_g, dv_g = grad_g_fn(q)

    # Compute Christoffel symbols
    Gamma = christoffel(g, du_g, dv_g)

    # Compute acceleration: aᵏ = -Γᵏᵢⱼ vⁱ vʲ
    accel = np.zeros(2)
    for k in range(2):
        for i in range(2):
            for j in range(2):
                accel[k] -= Gamma[k, i, j] * v[i] * v[j]

    return accel


def geodesic_leapfrog_step(q: np.ndarray, v: np.ndarray, dt: float,
                          g_fn: Callable[[np.ndarray], np.ndarray],
                          grad_g_fn: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]],
                          strip: Strip) -> Tuple[np.ndarray, np.ndarray]:
    """Single leapfrog step for geodesic integration with seam handling.

    Implements symplectic leapfrog integration:
    1. Half velocity step: v(t+dt/2) = v(t) + (dt/2) * a(q(t))
    2. Full position step: q(t+dt) = q(t) + dt * v(t+dt/2)
    3. Handle seam crossings
    4. Half velocity step: v(t+dt) = v(t+dt/2) + (dt/2) * a(q(t+dt))

    Parameters
    ----------
    q : ndarray, shape (2,)
        Current position [u, v]
    v : ndarray, shape (2,)
        Current velocity [du/dt, dv/dt]
    dt : float
        Time step
    g_fn : callable
        Metric function
    grad_g_fn : callable
        Metric gradient function
    strip : Strip
        Strip configuration for seam handling

    Returns
    -------
    q_new : ndarray, shape (2,)
        New position after time step
    v_new : ndarray, shape (2,)
        New velocity after time step
    """
    # Step 1: Half velocity step
    accel = geodesic_acceleration(q, v, g_fn, grad_g_fn)
    v_half = v + 0.5 * dt * accel

    # Step 2: Full position step
    q_new = q + dt * v_half

    # Step 3: Handle seam crossings
    u_new, v_new_coord, du_half, dv_half = apply_seam_if_needed(q_new[0], q_new[1], v_half[0], v_half[1], strip)
    q_new = np.array([u_new, v_new_coord])
    v_half = np.array([du_half, dv_half])

    # Step 4: Second half velocity step
    accel_new = geodesic_acceleration(q_new, v_half, g_fn, grad_g_fn)
    v_new = v_half + 0.5 * dt * accel_new

    return q_new, v_new


def geodesic_energy(q: np.ndarray, v: np.ndarray,
                   g_fn: Callable[[np.ndarray], np.ndarray]) -> float:
    """Compute kinetic energy of geodesic motion.

    Parameters
    ----------
    q : ndarray, shape (2,)
        Position coordinates
    v : ndarray, shape (2,)
        Velocity vector
    g_fn : callable
        Metric function

    Returns
    -------
    energy : float
        Kinetic energy E = (1/2) vᵀ g(q) v
    """
    q = np.asarray(q)
    v = np.asarray(v)

    g = g_fn(q)
    return 0.5 * np.dot(v, np.dot(g, v))


def integrate_geodesic(q0: np.ndarray, v0: np.ndarray, t_final: float, dt: float,
                      g_fn: Callable[[np.ndarray], np.ndarray],
                      grad_g_fn: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]],
                      strip: Strip,
                      energy_tolerance: float = 1e-3,
                      save_every: int = 1) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Integrate geodesic on Möbius band with energy monitoring.

    Parameters
    ----------
    q0 : ndarray, shape (2,)
        Initial position [u₀, v₀]
    v0 : ndarray, shape (2,)
        Initial velocity [du/dt₀, dv/dt₀]
    t_final : float
        Final integration time
    dt : float
        Time step size
    g_fn : callable
        Metric function g(q) → 2×2 metric tensor
    grad_g_fn : callable
        Metric gradient function q → (∂g/∂u, ∂g/∂v)
    strip : Strip
        Strip configuration for seam handling
    energy_tolerance : float
        Maximum allowed relative energy drift
    save_every : int
        Save trajectory points every N steps

    Returns
    -------
    traj_q : ndarray, shape (n_saved, 2)
        Position trajectory
    traj_v : ndarray, shape (n_saved, 2)
        Velocity trajectory
    info : dict
        Integration diagnostics and statistics

    Raises
    ------
    GeodesicIntegrationError
        If energy conservation is violated beyond tolerance
    """
    q0 = np.asarray(q0, dtype=float)
    v0 = np.asarray(v0, dtype=float)

    if q0.shape != (2,) or v0.shape != (2,):
        raise ValueError("Initial position and velocity must be 2D vectors")

    if dt <= 0 or t_final <= 0:
        raise ValueError("Time step and final time must be positive")

    n_steps = int(np.ceil(t_final / dt))
    n_saved = int(np.ceil(n_steps / save_every)) + 1

    # Initialize trajectory arrays
    traj_q = np.zeros((n_saved, 2))
    traj_v = np.zeros((n_saved, 2))
    traj_energy = np.zeros(n_saved)
    traj_time = np.zeros(n_saved)

    # Initial conditions
    q, v = q0.copy(), v0.copy()
    E0 = geodesic_energy(q, v, g_fn)

    traj_q[0] = q
    traj_v[0] = v
    traj_energy[0] = E0
    traj_time[0] = 0.0

    # Integration statistics
    seam_crossings = 0
    max_energy_error = 0.0
    save_idx = 1

    for step in range(n_steps):
        t = step * dt

        # Check for seam crossing before step
        if np.abs(q[1]) >= strip.w * 0.99:  # Near seam boundary
            seam_crossings += 1

        # Take integration step
        try:
            q, v = geodesic_leapfrog_step(q, v, dt, g_fn, grad_g_fn, strip)
        except Exception as e:
            raise GeodesicIntegrationError(f"Integration failed at t={t:.3f}: {e}")

        # Energy monitoring
        if (step + 1) % save_every == 0 or step == n_steps - 1:
            E = geodesic_energy(q, v, g_fn)
            energy_error = abs(E - E0) / (abs(E0) + 1e-12)
            max_energy_error = max(max_energy_error, energy_error)

            if energy_error > energy_tolerance:
                warnings.warn(
                    f"Energy drift {energy_error:.2e} exceeds tolerance {energy_tolerance:.2e} "
                    f"at t={t:.3f}. Consider reducing time step."
                )

            # Save trajectory point
            if save_idx < n_saved:
                traj_q[save_idx] = q
                traj_v[save_idx] = v
                traj_energy[save_idx] = E
                traj_time[save_idx] = t + dt
                save_idx += 1

    # Final energy check
    E_final = geodesic_energy(q, v, g_fn)
    final_energy_error = abs(E_final - E0) / (abs(E0) + 1e-12)

    if final_energy_error > energy_tolerance:
        raise GeodesicIntegrationError(
            f"Final energy drift {final_energy_error:.2e} exceeds tolerance {energy_tolerance:.2e}. "
            f"Integration may be unstable - consider reducing time step."
        )

    # Trim arrays to actual saved points
    traj_q = traj_q[:save_idx]
    traj_v = traj_v[:save_idx]
    traj_energy = traj_energy[:save_idx]
    traj_time = traj_time[:save_idx]

    info = {
        "success": True,
        "n_steps": n_steps,
        "n_saved": save_idx,
        "final_time": traj_time[-1],
        "dt": dt,
        "initial_energy": float(E0),
        "final_energy": float(E_final),
        "energy_drift": float(final_energy_error),
        "max_energy_error": float(max_energy_error),
        "seam_crossings": seam_crossings,
        "energy_conservation": final_energy_error <= energy_tolerance,
        "trajectory_length": float(np.sum(np.linalg.norm(np.diff(traj_q, axis=0), axis=1))),
        "time_array": traj_time,
        "energy_array": traj_energy
    }

    return traj_q, traj_v, info


def adaptive_geodesic_step(q: np.ndarray, v: np.ndarray,
                          g_fn: Callable[[np.ndarray], np.ndarray],
                          grad_g_fn: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]],
                          strip: Strip,
                          dt: float,
                          target_error: float = 1e-6) -> Tuple[np.ndarray, np.ndarray, float, bool]:
    """Adaptive step size geodesic integration using Richardson extrapolation.

    Parameters
    ----------
    q, v : ndarray
        Current position and velocity
    g_fn, grad_g_fn : callable
        Metric and gradient functions
    strip : Strip
        Strip configuration
    dt : float
        Proposed time step
    target_error : float
        Target integration error

    Returns
    -------
    q_new, v_new : ndarray
        New position and velocity
    dt_new : float
        Adjusted time step for next iteration
    accept : bool
        Whether step was accepted
    """
    # Take one step of size dt
    q1, v1 = geodesic_leapfrog_step(q, v, dt, g_fn, grad_g_fn, strip)

    # Take two steps of size dt/2
    q_half, v_half = geodesic_leapfrog_step(q, v, dt/2, g_fn, grad_g_fn, strip)
    q2, v2 = geodesic_leapfrog_step(q_half, v_half, dt/2, g_fn, grad_g_fn, strip)

    # Estimate error (Richardson extrapolation)
    error_q = np.linalg.norm(q2 - q1)
    error_v = np.linalg.norm(v2 - v1)
    error = max(error_q, error_v)

    # Step size adjustment factor
    if error > 0:
        factor = min(2.0, max(0.5, (target_error / error) ** 0.25))
    else:
        factor = 1.5

    dt_new = dt * factor

    # Accept step if error is within tolerance
    accept = error <= target_error

    if accept:
        # Use higher-order estimate (Richardson extrapolation)
        q_new = q2 + (q2 - q1) / 3  # 4th order correction
        v_new = v2 + (v2 - v1) / 3
    else:
        q_new, v_new = q, v  # Reject step

    return q_new, v_new, dt_new, accept


def geodesic_distance(q1: np.ndarray, q2: np.ndarray,
                     g_fn: Callable[[np.ndarray], np.ndarray],
                     grad_g_fn: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]],
                     strip: Strip,
                     n_steps: int = 100) -> float:
    """Compute geodesic distance between two points.

    Parameters
    ----------
    q1, q2 : ndarray
        Start and end points
    g_fn, grad_g_fn : callable
        Metric and gradient functions
    strip : Strip
        Strip configuration
    n_steps : int
        Number of integration steps

    Returns
    -------
    distance : float
        Approximate geodesic distance
    """
    # Simple shooting method: integrate from q1 toward q2
    direction = q2 - q1
    distance_estimate = np.linalg.norm(direction)

    if distance_estimate < 1e-12:
        return 0.0

    # Initial velocity in direction of target
    v0 = direction / distance_estimate

    dt = distance_estimate / n_steps
    t_final = distance_estimate

    try:
        traj_q, _, info = integrate_geodesic(q1, v0, t_final, dt, g_fn, grad_g_fn, strip)
        return info["trajectory_length"]
    except GeodesicIntegrationError:
        # Fallback to Euclidean distance
        warnings.warn("Geodesic integration failed, using Euclidean distance")
        return distance_estimate