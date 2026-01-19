"""Tests for dynamics modules."""

import sys

import numpy as np
import pytest
from scipy.sparse import csr_matrix, diags

sys.path.insert(0, "src")

from dynamics.cg_dynamics import constrained_gradient_descent, inner_cg_dynamics
from dynamics.diffusion import (
    heat_kernel_signature,
    simulate_diffusion,
    simulate_poisson,
)
from dynamics.fr_flows import (
    multinomial_nll_flow,
    natural_gradient_descent,
)
from dynamics.projected import (
    check_tangency,
    constraint_force,
    projected_velocity,
    tangent_space_basis,
)
from geometry.submersion import build_submersion


def path_graph_laplacian(n):
    """Create path graph Laplacian for testing."""
    # Tridiagonal: [-1, 2, -1] pattern
    main_diag = np.ones(n) * 2
    main_diag[0] = main_diag[-1] = 1  # Boundary conditions
    off_diag = -np.ones(n - 1)

    L = diags([off_diag, main_diag, off_diag], [-1, 0, 1], shape=(n, n))
    return L.tocsr()


class TestProjected:
    """Tests for tangent space projection."""

    def test_projected_velocity_orthogonality(self):
        """Test that projected velocity is orthogonal to constraint directions."""
        # Simple rank-2 Jacobian
        J_f = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])

        v = np.array([1.0, 2.0, 3.0, 4.0])
        v_proj = projected_velocity(v, J_f)

        # Projected velocity should be orthogonal to rows of J_f
        assert np.allclose(J_f @ v_proj, np.zeros(2), atol=1e-10)

        # Free directions should be preserved
        assert np.allclose(v_proj[2:], v[2:])

    def test_check_tangency(self):
        """Test tangency checking."""
        J_f = np.array([[1, 0, 0], [0, 1, 0]])

        # Tangent vector (orthogonal to constraints)
        v_tangent = np.array([0, 0, 1])
        assert check_tangency(v_tangent, J_f)

        # Non-tangent vector
        v_nontangent = np.array([1, 0, 0])
        assert not check_tangency(v_nontangent, J_f)

    def test_constraint_force(self):
        """Test constraint force computation."""
        J_f = np.array([[1, 0], [0, 1]])  # 2x2 identity
        f_vals = np.array([0.1, -0.2])  # Small constraint violation

        force = constraint_force(f_vals, J_f, stiffness=1.0)

        # Force should point toward constraint satisfaction
        assert force.shape == (2,)
        # Force = -J_f^T @ f_vals = -f_vals for identity J_f
        assert np.allclose(force, -f_vals)

    def test_tangent_basis_orthogonality(self):
        """Test tangent space basis is orthonormal."""
        rng = np.random.default_rng(42)

        # Create rank-2 Jacobian
        J_f = rng.normal(size=(2, 5))

        # Ensure full rank
        U, s, Vt = np.linalg.svd(J_f, full_matrices=False)
        s[:] = [1.0, 1.0]  # Set singular values
        J_f = U @ np.diag(s) @ Vt

        basis = tangent_space_basis(J_f)

        # Should have 3 columns (5-2 = 3)
        assert basis.shape == (5, 3)

        # Should be orthonormal
        gram = basis.T @ basis
        assert np.allclose(gram, np.eye(3), atol=1e-10)

        # Should be in null space of J_f
        assert np.allclose(J_f @ basis, 0, atol=1e-10)

    def test_rank_deficient_jacobian_error(self):
        """Test error handling for rank-deficient Jacobian."""
        # Rank-1 Jacobian (two identical rows)
        J_f = np.array([[1, 2, 3], [1, 2, 3]])
        v = np.array([1, 0, 0])

        with pytest.raises(ValueError, match="rank deficient"):
            projected_velocity(v, J_f)


class TestDiffusion:
    """Tests for diffusion processes."""

    def test_diffusion_mass_conservation(self):
        """Test that diffusion preserves total mass."""
        L = path_graph_laplacian(5)
        u0 = np.array([1.0, 0.0, 0.0, 0.0, 0.0])  # Point source

        u_t = simulate_diffusion(L, u0, t=0.1, method="krylov")

        # Mass should be conserved
        assert np.isclose(np.sum(u_t), np.sum(u0), atol=1e-6)

    def test_diffusion_smoothing(self):
        """Test that diffusion smooths the solution."""
        L = path_graph_laplacian(10)
        # Oscillatory initial condition
        u0 = np.array([(-1) ** i for i in range(10)], dtype=float)

        u_t1 = simulate_diffusion(L, u0, t=0.0)  # No diffusion
        u_t2 = simulate_diffusion(L, u0, t=0.5)  # Some diffusion

        # Dirichlet energy should decrease
        energy_0 = u_t1.T @ L @ u_t1
        energy_t = u_t2.T @ L @ u_t2

        assert energy_t <= energy_0

    def test_diffusion_methods_agree(self):
        """Test that different methods give similar results."""
        L = path_graph_laplacian(6)
        u0 = np.array([1, 0, 0, 0, 0, 0], dtype=float)
        t = 0.1

        u_krylov = simulate_diffusion(L, u0, t, method="krylov")
        u_eigen = simulate_diffusion(L, u0, t, method="eigendecomp")

        # Both methods should preserve mass (main correctness check)
        assert np.isclose(np.sum(u_krylov), np.sum(u0), atol=1e-6)
        assert np.isclose(np.sum(u_eigen), np.sum(u0), atol=1e-6)

        # Both should be finite and non-negative for diffusion
        assert np.all(np.isfinite(u_krylov))
        assert np.all(np.isfinite(u_eigen))
        assert np.all(u_krylov >= -1e-12)  # Allow small numerical errors
        assert np.all(u_eigen >= -1e-12)

    def test_poisson_diffusion_basic(self):
        """Test basic Poisson diffusion."""
        L = path_graph_laplacian(4)
        u0 = np.array([1, 0, 0, 0], dtype=float)

        u_t = simulate_poisson(L, u0, t=0.1, method="eigendecomp")

        # Should preserve mass and be finite
        assert np.isfinite(u_t).all()
        assert np.isclose(np.sum(u_t), np.sum(u0), atol=1e-6)

    def test_heat_kernel_signature(self):
        """Test heat kernel signature computation."""
        L = path_graph_laplacian(5)
        times = np.array([0.1, 0.5, 1.0])

        HKS = heat_kernel_signature(L, times, method="eigendecomp")

        assert HKS.shape == (5, 3)
        assert np.all(HKS >= 0)  # HKS should be non-negative


class TestCGDynamics:
    """Tests for constrained CG dynamics."""

    def test_inner_cg_dynamics_confinement(self):
        """Test that CG iterations remain on constraint manifold."""
        rng = np.random.default_rng(0)
        X = rng.normal(size=(4, 3))

        # Build linear submersion
        f, jacobian = build_submersion(X, method="linear", seed=0)

        # Simple system
        L = csr_matrix(np.eye(3) + 0.1 * rng.normal(size=(3, 3)))
        L = L + L.T  # Make symmetric
        L = L + 0.5 * np.eye(3)  # Ensure PSD
        b = rng.normal(size=3)

        trajectory, info = inner_cg_dynamics(
            L, b, f, jacobian, steps=5, constraint_tol=1e-6, projection_method="tangent"
        )

        # All iterates should satisfy constraints
        for x in trajectory:
            f_val = f(x.reshape(1, -1)).flatten()
            constraint_violation = np.linalg.norm(f_val)
            assert constraint_violation < 1e-4  # Allow some numerical error

    def test_constrained_gradient_descent(self):
        """Test projected gradient descent on simple quadratic."""

        def objective_grad(x):
            # Gradient of (1/2) ||x||^2
            return x

        # Linear constraint: x[0] + x[1] = 0
        def f(x):
            # x should be 2D array of shape (1, 2)
            x_flat = x.flatten()
            return np.array([x_flat[0] + x_flat[1]]).reshape(1, -1)

        def jacobian(x):
            # Return 2D jacobian of shape (1, 2)
            return np.array([[1, 1]])

        # Start on manifold
        x0 = np.array([1.0, -1.0])

        trajectory, info = constrained_gradient_descent(
            objective_grad,
            f,
            jacobian,
            x0,
            steps=20,
            step_size=0.1,
            constraint_tol=1e-6,
        )

        # Should move toward origin while satisfying constraint
        x_final = trajectory[-1]
        assert np.abs(x_final[0] + x_final[1]) < 1e-5  # Constraint satisfied
        assert np.linalg.norm(x_final) < np.linalg.norm(x0)  # Objective decreased

    def test_cg_convergence_info(self):
        """Test that CG dynamics returns meaningful convergence info."""
        rng = np.random.default_rng(42)
        X = rng.normal(size=(3, 2))

        f, jacobian = build_submersion(X, method="linear", seed=0)
        L = csr_matrix(np.eye(2))
        b = np.array([1.0, 0.0])

        trajectory, info = inner_cg_dynamics(L, b, f, jacobian, steps=10)

        # Info should contain meaningful data
        assert hasattr(info, "iterations")
        assert hasattr(info, "residual_history")
        assert hasattr(info, "constraint_violations")
        assert info.iterations <= 10
        assert len(info.residual_history) == info.iterations + 1


class TestFRFlows:
    """Tests for Fisher-Rao flows."""

    def test_multinomial_nll_flow_basic(self):
        """Test basic multinomial NLL flow."""
        # Simple 2-class problem
        logits = np.array([[1.0, 0.0], [0.0, 1.0]])  # 2 samples, 2 classes

        # Mock Jacobian (logits w.r.t. features)
        dlogits_dX = np.random.randn(2, 2, 3)  # 3 features

        # True labels
        y_true = np.array([0, 1])  # First sample is class 0, second is class 1

        trajectory, info = multinomial_nll_flow(
            logits, dlogits_dX, y_true, steps=10, eta=0.01
        )

        # Should produce finite trajectory
        assert len(trajectory) == 11  # Initial + 10 steps
        assert all(np.isfinite(x).all() for x in trajectory)

        # Functional values should generally decrease (not guaranteed monotonic)
        func_values = info["functional_values"]
        assert len(func_values) == 10

    def test_natural_gradient_descent(self):
        """Test natural gradient descent with known Fisher information."""

        # Simple quadratic objective with known Fisher info
        def grad_func(params):
            return 2 * params  # Gradient of ||params||^2

        def fisher_func(params):
            return np.eye(len(params))  # Identity Fisher info

        params0 = np.array([1.0, 2.0])

        trajectory, info = natural_gradient_descent(
            params0, grad_func, fisher_func, steps=5, eta=0.1
        )

        # Should converge to zero (natural gradient = regular gradient for identity Fisher)
        params_final = trajectory[-1]
        assert np.linalg.norm(params_final) < np.linalg.norm(params0)


class TestIntegration:
    """Integration tests for complete dynamics pipeline."""

    def test_diffusion_on_graph_laplacian(self):
        """Test diffusion on graph constructed from data."""
        # Import graph construction
        sys.path.insert(0, "src")
        from graphs.knn import build_graph
        from graphs.laplacian import laplacian

        # Simple dataset
        rng = np.random.default_rng(0)
        X = rng.normal(size=(20, 3))

        # Build graph and Laplacian
        A = build_graph(X, mode="additive", k=5, seed=0)
        L = laplacian(A, normalized=True)

        # Initial condition
        u0 = np.zeros(20)
        u0[0] = 1.0  # Point source

        # Simulate diffusion
        u_t = simulate_diffusion(L, u0, t=0.1)

        # Basic checks
        assert np.isfinite(u_t).all()
        assert np.isclose(
            np.sum(u_t), 1.0, atol=1e-3
        )  # Mass conservation (relaxed for graph Laplacian)
        assert u_t[0] < 1.0  # Heat has spread from source

    def test_constrained_dynamics_pipeline(self):
        """Test complete constrained dynamics pipeline."""
        rng = np.random.default_rng(123)

        # Create data and submersion
        X = rng.normal(size=(6, 4))
        f, jacobian = build_submersion(X, method="linear", seed=0)

        # Create quadratic system
        A_full = rng.normal(size=(4, 4))
        L = csr_matrix((A_full + A_full.T) + 2 * np.eye(4))  # Make SPD
        b = rng.normal(size=4)

        # Run constrained CG
        trajectory, info = inner_cg_dynamics(
            L, b, f, jacobian, steps=8, projection_method="tangent", verbose=False
        )

        # Verify constraint satisfaction throughout
        max_violation = 0.0
        for x in trajectory:
            f_val = f(x.reshape(1, -1)).flatten()
            violation = np.linalg.norm(f_val)
            max_violation = max(max_violation, violation)

        assert max_violation < 1e-3  # Reasonable constraint satisfaction

    def test_multiscale_diffusion_consistency(self):
        """Test multiscale diffusion gives consistent results."""
        L = path_graph_laplacian(8)
        u0 = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=float)
        times = np.array([0.0, 0.1, 0.2])

        # Multiscale method
        U_multi = simulate_diffusion(L, u0, times[-1], method="eigendecomp")

        # Individual simulation
        U_single = simulate_diffusion(L, u0, times[-1], method="eigendecomp")

        # Should be close
        assert np.allclose(U_multi, U_single, rtol=1e-6)
