"""Topology module for geometric ML with quotient spaces.

This module provides rigorous topological quotient handling with the Möbius band
as the primary example. Implements deck maps, seam identification, metric
compatibility, and geodesic integration on nontrivial quotients.

Key Components:
- coords: Deck maps, coordinate wrapping, velocity pushforward
- metric: Seam-compatibility for Fisher-Rao pullback metrics
- geodesic: Symplectic integration with seam-aware handling
- validation: Topological invariance tests for statistics
"""

from .atlas import (
    Cylinder,
    KleinBottle,
    MobiusBand,
    Orientability,
    ProjectivePlane,
    QuotientSpace,
    Sphere,
    TopologyAtlas,
    TopologyType,
    Torus,
    create_topology,
    get_orientability_pairs,
    topology_atlas,
)
from .coords import Strip, apply_seam_if_needed, deck_map, pushforward_velocity, wrap_u
from .geodesic import christoffel, geodesic_leapfrog_step, integrate_geodesic
from .metric import (
    enforce_seam_compatibility,
    seam_compatible_metric,
    seam_compatible_operator,
)
from .validation import (
    comprehensive_topology_validation,
    seam_invariance,
    seam_invariance_grid,
    validate_metric_invariance,
)

__all__ = [
    # Coordinate handling
    "Strip",
    "wrap_u",
    "deck_map",
    "pushforward_velocity",
    "apply_seam_if_needed",
    # Topology atlas
    "QuotientSpace",
    "Orientability",
    "TopologyType",
    "Cylinder",
    "MobiusBand",
    "Torus",
    "KleinBottle",
    "Sphere",
    "ProjectivePlane",
    "TopologyAtlas",
    "topology_atlas",
    "create_topology",
    "get_orientability_pairs",
    # Metric compatibility
    "seam_compatible_metric",
    "enforce_seam_compatibility",
    "seam_compatible_operator",
    # Geodesic integration
    "christoffel",
    "geodesic_leapfrog_step",
    "integrate_geodesic",
    # Validation
    "seam_invariance",
    "seam_invariance_grid",
    "comprehensive_topology_validation",
    "validate_metric_invariance",
]

__version__ = "0.1.0"
