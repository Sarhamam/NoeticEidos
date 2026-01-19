"""Mathematical validity guards for geometric ML operations."""

import numpy as np
from scipy.sparse import issparse, csgraph
from scipy.linalg import svd
from typing import Union, Tuple, Dict, Any, Optional
import warnings


class ValidationError(Exception):
    """Base class for validation errors."""
    pass


class ConnectivityError(ValidationError):
    """Raised when graph connectivity requirements are violated."""
    pass


class TransversalityError(ValidationError):
    """Raised when transversality conditions are violated."""
    pass


def check_graph_connectivity(adjacency_matrix: Union[np.ndarray, 'csr_matrix'],
                           require_connected: bool = True,
                           return_components: bool = False) -> Union[bool, Tuple[bool, int, np.ndarray]]:
    """Check graph connectivity and return component information.

    Parameters
    ----------
    adjacency_matrix : sparse matrix or ndarray
        Graph adjacency matrix
    require_connected : bool
        If True, raise ConnectivityError for disconnected graphs
    return_components : bool
        If True, return detailed component information

    Returns
    -------
    is_connected : bool
        Whether graph is connected (if return_components=False)
    (is_connected, n_components, component_labels) : tuple
        Full connectivity information (if return_components=True)

    Raises
    ------
    ConnectivityError
        If require_connected=True and graph is disconnected

    Notes
    -----
    Connected graphs are required for meaningful spectral analysis.
    Disconnected graphs have zero eigenvalues with multiplicity > 1.
    """
    if adjacency_matrix.shape[0] != adjacency_matrix.shape[1]:
        raise ValueError("Adjacency matrix must be square")

    if adjacency_matrix.shape[0] == 0:
        raise ValueError("Empty adjacency matrix")

    # Use scipy's connected components analysis
    n_components, component_labels = csgraph.connected_components(
        adjacency_matrix, directed=False, return_labels=True
    )

    is_connected = (n_components == 1)

    if require_connected and not is_connected:
        raise ConnectivityError(
            f"Graph is disconnected with {n_components} components. "
            f"Spectral analysis requires connected graphs. "
            f"Component sizes: {np.bincount(component_labels)}"
        )

    if return_components:
        return is_connected, n_components, component_labels
    else:
        return is_connected


def validate_transversality(jacobian: np.ndarray,
                           expected_rank: int = 2,
                           condition_threshold: float = 1e6,
                           min_singular_value: float = 1e-12) -> Dict[str, Any]:
    """Validate transversality condition for constraint manifolds.

    Parameters
    ----------
    jacobian : ndarray, shape (k, n)
        Jacobian matrix of constraint function f: R^n -> R^k
    expected_rank : int
        Expected rank of Jacobian (typically k for transversal submersion)
    condition_threshold : float
        Maximum allowed condition number for J_f^T J_f
    min_singular_value : float
        Minimum allowed singular value

    Returns
    -------
    certificate : dict
        Transversality validation certificate with:
        - "is_transversal": bool
        - "rank": int (actual rank)
        - "condition_number": float
        - "singular_values": ndarray
        - "min_singular_value": float

    Raises
    ------
    TransversalityError
        If transversality conditions are violated

    Notes
    -----
    Transversality ensures constraint manifold Z = {x : f(x) = 0}
    is a well-defined submanifold. Requires rank(J_f) = k everywhere on Z.
    """
    if jacobian.ndim != 2:
        raise ValueError("Jacobian must be 2D array")

    k, n = jacobian.shape

    if expected_rank > min(k, n):
        raise ValueError(f"Expected rank {expected_rank} cannot exceed min(k,n) = {min(k, n)}")

    # SVD for numerical rank assessment
    try:
        U, s, Vt = svd(jacobian, full_matrices=False)
    except np.linalg.LinAlgError as e:
        raise TransversalityError(f"SVD failed: {e}")

    # Assess rank
    actual_rank = np.sum(s > min_singular_value)
    min_sv = np.min(s) if len(s) > 0 else 0.0

    # Condition number of J_f^T J_f
    if actual_rank > 0:
        # Condition number of J_f is ratio of largest to smallest nonzero singular value
        nonzero_sv = s[s > min_singular_value]
        if len(nonzero_sv) > 0:
            condition_number = np.max(nonzero_sv) / np.min(nonzero_sv)
        else:
            condition_number = float('inf')
    else:
        condition_number = float('inf')

    # Transversality assessment
    is_transversal = (
        actual_rank >= expected_rank and
        condition_number <= condition_threshold and
        min_sv >= min_singular_value
    )

    certificate = {
        "is_transversal": is_transversal,
        "rank": actual_rank,
        "expected_rank": expected_rank,
        "condition_number": condition_number,
        "singular_values": s.copy(),
        "min_singular_value": min_sv,
        "condition_threshold": condition_threshold
    }

    if not is_transversal:
        error_msg = (
            f"Transversality violation detected:\n"
            f"  - Actual rank: {actual_rank} (expected: {expected_rank})\n"
            f"  - Condition number: {condition_number:.2e} (threshold: {condition_threshold:.2e})\n"
            f"  - Min singular value: {min_sv:.2e} (threshold: {min_singular_value:.2e})\n"
            f"This indicates the constraint manifold may be degenerate."
        )
        raise TransversalityError(error_msg)

    return certificate


def check_manifold_dimension_consistency(embeddings: np.ndarray,
                                       intrinsic_dim: int,
                                       tolerance: float = 0.1) -> Dict[str, Any]:
    """Check consistency between embedding and intrinsic manifold dimensions.

    Parameters
    ----------
    embeddings : ndarray, shape (n, d)
        Embedded points in R^d
    intrinsic_dim : int
        Expected intrinsic manifold dimension
    tolerance : float
        Tolerance for dimension estimation

    Returns
    -------
    result : dict
        Dimension consistency check results
    """
    n, d = embeddings.shape

    if intrinsic_dim >= d:
        warnings.warn(f"Intrinsic dimension {intrinsic_dim} >= embedding dimension {d}")

    # Simple PCA-based dimension estimation
    # Center the data
    centered = embeddings - np.mean(embeddings, axis=0)

    # SVD for principal components
    try:
        U, s, Vt = svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return {
            "consistent": False,
            "estimated_dim": None,
            "error": "SVD failed"
        }

    # Estimate intrinsic dimension via explained variance
    total_variance = np.sum(s**2)
    cumulative_variance = np.cumsum(s**2) / total_variance

    # Find dimension that explains 95% of variance
    explained_95 = np.searchsorted(cumulative_variance, 0.95) + 1

    # Check if estimated dimension is consistent with expected
    consistent = abs(explained_95 - intrinsic_dim) <= tolerance * intrinsic_dim

    return {
        "consistent": consistent,
        "estimated_dim": explained_95,
        "expected_dim": intrinsic_dim,
        "explained_variance_ratios": s**2 / total_variance,
        "cumulative_variance": cumulative_variance
    }


def validate_submersion_properties(f: callable,
                                 jacobian: callable,
                                 X: np.ndarray,
                                 sample_size: int = 100,
                                 seed: Optional[int] = None) -> Dict[str, Any]:
    """Validate submersion properties across multiple points.

    Parameters
    ----------
    f : callable
        Submersion function f: R^n -> R^k
    jacobian : callable
        Jacobian function returning J_f(x)
    X : ndarray, shape (m, n)
        Points to sample from for validation
    sample_size : int
        Number of points to validate
    seed : int or None
        Random seed for point sampling

    Returns
    -------
    validation_report : dict
        Comprehensive submersion validation report
    """
    rng = np.random.default_rng(seed)
    m, n = X.shape

    # Sample points for validation
    sample_indices = rng.choice(m, size=min(sample_size, m), replace=False)
    sample_points = X[sample_indices]

    validation_results = []
    failed_points = []

    for i, x in enumerate(sample_points):
        try:
            # Evaluate function and Jacobian
            f_val = f(x.reshape(1, -1))
            J_f = jacobian(x.reshape(1, -1))

            # Handle different return formats
            if len(J_f.shape) == 3:
                J_f = J_f[0]  # Remove batch dimension

            # Validate transversality at this point
            cert = validate_transversality(J_f, expected_rank=f_val.shape[1])

            validation_results.append({
                "point_index": sample_indices[i],
                "f_value": f_val,
                "is_transversal": cert["is_transversal"],
                "rank": cert["rank"],
                "condition_number": cert["condition_number"]
            })

        except (TransversalityError, ValueError) as e:
            failed_points.append({
                "point_index": sample_indices[i],
                "error": str(e)
            })

    # Summary statistics
    n_valid = len(validation_results)
    n_failed = len(failed_points)

    if n_valid > 0:
        n_transversal = sum(r["is_transversal"] for r in validation_results)
        avg_condition = np.mean([r["condition_number"] for r in validation_results
                               if np.isfinite(r["condition_number"])])
        success_rate = n_transversal / n_valid
    else:
        n_transversal = 0
        avg_condition = float('inf')
        success_rate = 0.0

    return {
        "overall_valid": success_rate > 0.9,  # Require 90% success rate
        "success_rate": success_rate,
        "points_validated": n_valid,
        "points_failed": n_failed,
        "transversal_points": n_transversal,
        "average_condition_number": avg_condition,
        "validation_details": validation_results,
        "failed_points": failed_points
    }