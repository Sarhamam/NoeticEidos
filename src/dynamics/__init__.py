"""Constrained dynamics on manifolds."""

from dynamics.cg_dynamics import (
    ConstrainedCGInfo,
    constrained_gradient_descent,
    inner_cg_dynamics,
    lagrange_multiplier_cg,
)
from dynamics.diffusion import (
    diffusion_distance,
    heat_kernel_signature,
    multiscale_diffusion,
    simulate_diffusion,
    simulate_poisson,
)
from dynamics.fr_flows import (
    exponential_family_flow,
    fr_gradient_flow,
    multinomial_nll_flow,
    natural_gradient_descent,
)
from dynamics.projected import (
    check_tangency,
    constraint_force,
    parallel_transport_approximation,
    project_matrix_to_tangent,
    projected_gradient_step,
    projected_velocity,
    tangent_space_basis,
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
