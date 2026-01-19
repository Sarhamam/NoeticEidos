"""Geometric operations for constrained manifold dynamics."""

from .fr_pullback import (
    fisher_rao_divergence,
    fisher_rao_metric,
    multinomial_fisher_info,
    pullback_metric,
    rescale_by_metric,
    riemannian_distance,
)
from .projection import (
    check_projection_properties,
    constraint_violation,
    project_to_manifold,
    project_to_tangent,
    project_vector,
    tangent_basis,
)
from .submersion import build_submersion, check_transversal, find_zero_set

__all__ = [
    # Submersion
    "build_submersion",
    "check_transversal",
    "find_zero_set",
    # Projection
    "project_to_tangent",
    "project_vector",
    "tangent_basis",
    "check_projection_properties",
    "constraint_violation",
    "project_to_manifold",
    # Fisher-Rao
    "fisher_rao_metric",
    "multinomial_fisher_info",
    "rescale_by_metric",
    "pullback_metric",
    "riemannian_distance",
    "fisher_rao_divergence",
]
