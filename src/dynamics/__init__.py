"""Constrained dynamics on manifolds."""

from dynamics.projected import (
    projected_velocity,
    projected_gradient_step,
    check_tangency,
    parallel_transport_approximation,
    constraint_force,
    tangent_space_basis,
    project_matrix_to_tangent
)
from dynamics.diffusion import (
    simulate_diffusion,
    simulate_poisson,
    diffusion_distance,
    heat_kernel_signature,
    multiscale_diffusion
)
from dynamics.cg_dynamics import (
    inner_cg_dynamics,
    constrained_gradient_descent,
    lagrange_multiplier_cg,
    ConstrainedCGInfo
)
from dynamics.fr_flows import (
    fr_gradient_flow,
    multinomial_nll_flow,
    natural_gradient_descent,
    exponential_family_flow
)

__all__ = [
    # Projected dynamics
    "projected_velocity",
    "projected_gradient_step",
    "check_tangency",
    "parallel_transport_approximation",
    "constraint_force",
    "tangent_space_basis",
    "project_matrix_to_tangent",
    # Diffusion
    "simulate_diffusion",
    "simulate_poisson",
    "diffusion_distance",
    "heat_kernel_signature",
    "multiscale_diffusion",
    # Constrained CG
    "inner_cg_dynamics",
    "constrained_gradient_descent",
    "lagrange_multiplier_cg",
    "ConstrainedCGInfo",
    # Fisher-Rao flows
    "fr_gradient_flow",
    "multinomial_nll_flow",
    "natural_gradient_descent",
    "exponential_family_flow",
]