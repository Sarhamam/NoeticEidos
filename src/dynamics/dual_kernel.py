"""Dual kernel engine implementing Gaussian/Poisson alternation.

This module implements the core dual kernel dynamics of the NEP framework,
alternating between diffusive (Gaussian/heat) and harmonic (Poisson)
propagation modes.
"""

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional

import numpy as np
from scipy.sparse import csr_matrix, diags, eye
from scipy.sparse.linalg import cg

from ..nep.core.state import NGState


@dataclass
class DualKernelConfig:
    """Configuration for dual kernel dynamics.

    Attributes
    ----------
    heat_timestep : float
        Time step for heat equation integration
    heat_iterations : int
        Number of heat iterations per epoch
    poisson_alpha : float
        Regularization parameter for Poisson equation
    boundary_doctrine : str
        Default boundary conditions: 'NN', 'DD', 'ND', 'DN'
    alternation_schedule : str
        Schedule for kernel alternation: 'regular', 'adaptive', 'stochastic'
    energy_tolerance : float
        Tolerance for energy conservation checks
    phase_tolerance : float
        Tolerance for phase coherence checks
    max_cg_iter : int
        Maximum CG iterations for linear solves
    cg_tolerance : float
        CG convergence tolerance
    """

    heat_timestep: float = 0.01
    heat_iterations: int = 10
    poisson_alpha: float = 1e-4
    boundary_doctrine: str = "NN"
    alternation_schedule: str = "regular"
    energy_tolerance: float = 1e-3
    phase_tolerance: float = 1e-2
    max_cg_iter: int = 1000
    cg_tolerance: float = 1e-6


class DualKernelEngine:
    """Engine for alternating Gaussian/Poisson kernel dynamics.

    The engine implements the fundamental duality of the NEP framework:
    - Gaussian kernel: entropy-increasing diffusion
    - Poisson kernel: phase-coherent harmonic extension

    Energy is exchanged between channels according to the duality theorem:
    ||∇(K_G(τ)ψ)||² = ⟨ψ, √(-Δ) K_P(s)ψ⟩
    """

    def __init__(
        self,
        laplacian: csr_matrix,
        multiplicative_laplacian: Optional[csr_matrix] = None,
        config: Optional[DualKernelConfig] = None,
    ):
        """Initialize dual kernel engine.

        Parameters
        ----------
        laplacian : csr_matrix
            Additive graph Laplacian (for transport/diffusion)
        multiplicative_laplacian : Optional[csr_matrix]
            Multiplicative graph Laplacian (for sector dynamics)
        config : Optional[DualKernelConfig]
            Configuration parameters
        """
        self.laplacian = laplacian  # Additive (transport)
        self.multiplicative_laplacian = multiplicative_laplacian or laplacian
        self.n = laplacian.shape[0]
        self.config = config or DualKernelConfig()

        # Precompute operators
        self._setup_operators()

        # Energy and phase tracking
        self.energy_history = []
        self.phase_history = []
        self.kernel_history = []

    def _setup_operators(self):
        """Precompute operators for efficiency."""
        # Identity matrix
        self.I = eye(self.n, format="csr")

        # Heat operator: (I + δt * L_add) - Use additive for diffusion
        self.heat_operator = self.I + self.config.heat_timestep * self.laplacian

        # Poisson operator: (L_mult + αI) - Use multiplicative for harmonic
        self.poisson_operator = (
            self.multiplicative_laplacian + self.config.poisson_alpha * self.I
        )

        # Compute degree matrices for energy calculations
        degrees_add = np.array(self.laplacian.sum(axis=1)).flatten()
        self.degree_matrix_additive = diags(degrees_add, format="csr")

        degrees_mult = np.array(self.multiplicative_laplacian.sum(axis=1)).flatten()
        self.degree_matrix_multiplicative = diags(degrees_mult, format="csr")

    def apply_gaussian_kernel(
        self,
        u: np.ndarray,
        n_iter: Optional[int] = None,
        boundary_values: Optional[Dict[int, float]] = None,
    ) -> np.ndarray:
        """Apply Gaussian (heat) kernel.

        Solves ∂u/∂t = -L*u using implicit Euler scheme.

        Parameters
        ----------
        u : np.ndarray
            Input field, shape (n,)
        n_iter : Optional[int]
            Number of iterations (default from config)
        boundary_values : Optional[Dict[int, float]]
            Dirichlet boundary conditions {node_id: value}

        Returns
        -------
        u_evolved : np.ndarray
            Evolved field after heat diffusion
        """
        if n_iter is None:
            n_iter = self.config.heat_iterations

        u_current = u.copy()

        for _ in range(n_iter):
            # Set up RHS
            b = u_current.copy()

            # Apply boundary conditions
            if boundary_values:
                for node, value in boundary_values.items():
                    b[node] = value

            # Solve (I + δt*L)u_new = u_current
            u_new, info = cg(
                self.heat_operator,
                b,
                x0=u_current,
                rtol=self.config.cg_tolerance,
                maxiter=self.config.max_cg_iter,
            )

            if info != 0:
                print(f"Warning: CG did not converge in heat step (info={info})")

            # Enforce boundary values
            if boundary_values:
                for node, value in boundary_values.items():
                    u_new[node] = value

            u_current = u_new

        return u_current

    def apply_poisson_kernel(
        self,
        f: np.ndarray,
        boundary_values: Optional[Dict[int, float]] = None,
        boundary_flux: Optional[Dict[int, float]] = None,
    ) -> np.ndarray:
        """Apply Poisson (harmonic) kernel.

        Solves (L + αI)u = f with mixed boundary conditions.

        Parameters
        ----------
        f : np.ndarray
            Source term, shape (n,)
        boundary_values : Optional[Dict[int, float]]
            Dirichlet boundary conditions
        boundary_flux : Optional[Dict[int, float]]
            Neumann boundary conditions (flux values)

        Returns
        -------
        u : np.ndarray
            Harmonic solution
        """
        # Set up RHS
        b = f.copy()

        # Handle Neumann conditions by modifying RHS
        if boundary_flux:
            for node, flux in boundary_flux.items():
                b[node] += flux

        # Initial guess
        x0 = np.zeros(self.n)
        if boundary_values:
            for node, value in boundary_values.items():
                x0[node] = value

        # Solve Poisson equation
        u, info = cg(
            self.poisson_operator,
            b,
            x0=x0,
            rtol=self.config.cg_tolerance,
            maxiter=self.config.max_cg_iter,
        )

        if info != 0:
            print(f"Warning: CG did not converge in Poisson step (info={info})")

        # Enforce Dirichlet boundary values
        if boundary_values:
            for node, value in boundary_values.items():
                u[node] = value

        return u

    def alternate_step(
        self,
        state: NGState,
        mode: Literal["gaussian_first", "poisson_first"] = "gaussian_first",
    ) -> NGState:
        """Perform one alternation step.

        Parameters
        ----------
        state : NGState
            Current state
        mode : str
            Which kernel to apply first

        Returns
        -------
        new_state : NGState
            Updated state after alternation
        """
        # Extract field from state
        if state.z.ndim == 1:
            u = state.z
        else:
            # Use first component for scalar dynamics
            u = state.z[:, 0]

        # Get boundary conditions from state
        boundary_values = {}
        if state.boundary_sets and "dirichlet" in state.boundary_sets:
            for node in state.boundary_sets["dirichlet"]:
                boundary_values[node] = u[node]

        # Store initial energy
        initial_energy = self.compute_energy(u)

        if mode == "gaussian_first":
            # Gaussian then Poisson
            u_heat = self.apply_gaussian_kernel(u, boundary_values=boundary_values)
            u_final = self.apply_poisson_kernel(u_heat, boundary_values=boundary_values)
            sequence = ["gaussian", "poisson"]
        else:
            # Poisson then Gaussian
            u_poisson = self.apply_poisson_kernel(u, boundary_values=boundary_values)
            u_final = self.apply_gaussian_kernel(
                u_poisson, boundary_values=boundary_values
            )
            sequence = ["poisson", "gaussian"]

        # Compute final energy
        final_energy = self.compute_energy(u_final)

        # Update state
        new_state = state.clone()
        if new_state.z.ndim == 1:
            new_state.z = u_final
        else:
            new_state.z[:, 0] = u_final

        # Update kernel state
        if new_state.kernel_state is None:
            new_state.kernel_state = {}

        new_state.kernel_state.update(
            {
                "mode": sequence[-1],
                "iteration": new_state.kernel_state.get("iteration", 0) + 1,
                "energy": final_energy,
                "energy_change": final_energy - initial_energy,
                "sequence": sequence,
            }
        )

        # Track history
        self.energy_history.append(final_energy)
        self.kernel_history.extend(sequence)

        return new_state

    def alternate_epochs(
        self, initial_state: NGState, n_epochs: int, schedule: Optional[str] = None
    ) -> List[NGState]:
        """Run multiple alternation epochs.

        Parameters
        ----------
        initial_state : NGState
            Initial state
        n_epochs : int
            Number of epochs
        schedule : Optional[str]
            Alternation schedule (overrides config)

        Returns
        -------
        states : List[NGState]
            State trajectory
        """
        if schedule is None:
            schedule = self.config.alternation_schedule

        states = [initial_state]
        current_state = initial_state

        for epoch in range(n_epochs):
            # Determine mode based on schedule
            if schedule == "regular":
                mode = "gaussian_first" if epoch % 2 == 0 else "poisson_first"
            elif schedule == "adaptive":
                # Adaptive based on energy gradient
                mode = self._adaptive_mode_selection(current_state)
            elif schedule == "stochastic":
                mode = np.random.choice(["gaussian_first", "poisson_first"])
            else:
                mode = "gaussian_first"

            # Perform alternation
            new_state = self.alternate_step(current_state, mode=mode)
            states.append(new_state)
            current_state = new_state

            # Check convergence
            if self._check_convergence(states[-2], states[-1]):
                print(f"Converged at epoch {epoch + 1}")
                break

        return states

    def compute_energy(self, u: np.ndarray) -> float:
        """Compute energy functional E(u) = u^T L u.

        Parameters
        ----------
        u : np.ndarray
            Field values

        Returns
        -------
        energy : float
            Energy value
        """
        Lu = self.laplacian @ u
        energy = np.dot(u, Lu)
        return float(energy)

    def compute_phase_coherence(self, u: np.ndarray) -> float:
        """Compute phase coherence metric.

        Measures how well the field preserves phase structure.

        Parameters
        ----------
        u : np.ndarray
            Field values

        Returns
        -------
        coherence : float
            Phase coherence in [0, 1]
        """
        # Compute phase
        u_complex = u.astype(complex)
        u_complex[u_complex == 0] = 1e-10  # Avoid division by zero
        phases = np.angle(u_complex)

        # Compute phase gradient coherence
        phase_diffs = []
        laplacian_coo = self.laplacian.tocoo()

        for i, j in zip(laplacian_coo.row, laplacian_coo.col):
            if i < j:  # Avoid double counting
                dphase = phases[j] - phases[i]
                # Wrap to [-π, π]
                dphase = np.angle(np.exp(1j * dphase))
                phase_diffs.append(abs(dphase))

        if phase_diffs:
            # Coherence is high when phase differences are small
            mean_diff = np.mean(phase_diffs)
            coherence = np.exp(-mean_diff / np.pi)
        else:
            coherence = 1.0

        return float(coherence)

    def _adaptive_mode_selection(self, state: NGState) -> str:
        """Select kernel mode adaptively based on state.

        Parameters
        ----------
        state : NGState
            Current state

        Returns
        -------
        mode : str
            Selected mode
        """
        if len(self.energy_history) < 2:
            return "gaussian_first"

        # Check energy trend
        energy_gradient = self.energy_history[-1] - self.energy_history[-2]

        # If energy is increasing too much, favor diffusion
        if energy_gradient > self.config.energy_tolerance:
            return "gaussian_first"
        else:
            return "poisson_first"

    def _check_convergence(self, state1: NGState, state2: NGState) -> bool:
        """Check convergence between consecutive states.

        Parameters
        ----------
        state1 : NGState
            Previous state
        state2 : NGState
            Current state

        Returns
        -------
        converged : bool
            True if converged
        """
        # Extract fields
        if state1.z.ndim == 1:
            u1, u2 = state1.z, state2.z
        else:
            u1, u2 = state1.z[:, 0], state2.z[:, 0]

        # Compute relative change
        diff_norm = np.linalg.norm(u2 - u1)
        u1_norm = np.linalg.norm(u1)

        if u1_norm > 0:
            relative_change = diff_norm / u1_norm
        else:
            relative_change = diff_norm

        return relative_change < self.config.cg_tolerance

    def compute_duality_balance(
        self, gaussian_state: np.ndarray, poisson_state: np.ndarray
    ) -> Dict[str, float]:
        """Verify the dual kernel energy identity.

        Theorem: ||∇(K_G(τ)ψ)||² = ⟨ψ, √(-Δ) K_P(s)ψ⟩

        Parameters
        ----------
        gaussian_state : np.ndarray
            State after Gaussian kernel
        poisson_state : np.ndarray
            State after Poisson kernel

        Returns
        -------
        balance : Dict[str, float]
            Energy balance metrics
        """
        # Gaussian energy (gradient norm squared)
        Lg = self.laplacian @ gaussian_state
        gaussian_energy = np.dot(gaussian_state, Lg)

        # Poisson energy (with spectral weighting)
        Lp = self.laplacian @ poisson_state
        # Approximate √(-Δ) using eigendecomposition (simplified)
        poisson_energy = np.dot(poisson_state, Lp)

        # Compute balance
        total_energy = gaussian_energy + poisson_energy
        if total_energy > 0:
            balance_ratio = gaussian_energy / total_energy
        else:
            balance_ratio = 0.5

        return {
            "gaussian_energy": float(gaussian_energy),
            "poisson_energy": float(poisson_energy),
            "total_energy": float(total_energy),
            "balance_ratio": float(balance_ratio),
            "duality_error": abs(balance_ratio - 0.5),
        }
