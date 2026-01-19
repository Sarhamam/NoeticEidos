"""Fisher-Rao metric pullback for model-aware embeddings."""

import numpy as np
from typing import Union, Optional, Callable
from scipy.special import softmax


def fisher_rao_metric(logits: np.ndarray, dlogits_dX: np.ndarray) -> np.ndarray:
    """Compute Fisher-Rao metric tensor for probabilistic models.

    Parameters
    ----------
    logits : np.ndarray, shape (n, k)
        Model logits (pre-softmax)
    dlogits_dX : np.ndarray, shape (n, k, d)
        Jacobian of logits w.r.t. input features

    Returns
    -------
    G : np.ndarray, shape (n, d, d)
        Fisher-Rao metric tensors at each point

    Notes
    -----
    For multinomial (softmax) models, the Fisher information matrix is:
    I_ij = Σ_k (∂log p_k/∂θ_i)(∂log p_k/∂θ_j) p_k

    The pullback metric is: G = (∂logits/∂X)^T I (∂logits/∂X)
    """
    n, k = logits.shape
    d = dlogits_dX.shape[2]

    # Compute probabilities
    probs = softmax(logits, axis=1)  # Shape (n, k)

    # Fisher information for multinomial model
    # I = diag(p) - p p^T where p is probability vector
    G = np.zeros((n, d, d))

    for i in range(n):
        p_i = probs[i]  # Shape (k,)
        J_i = dlogits_dX[i]  # Shape (k, d)

        # Fisher information matrix (k, k)
        I_i = np.diag(p_i) - np.outer(p_i, p_i)

        # Pullback: G = J^T I J
        G[i] = J_i.T @ I_i @ J_i

    return G


def multinomial_fisher_info(probs: np.ndarray) -> np.ndarray:
    """Compute Fisher information for multinomial distribution.

    Parameters
    ----------
    probs : np.ndarray, shape (n, k)
        Probability vectors

    Returns
    -------
    I : np.ndarray, shape (n, k, k)
        Fisher information matrices
    """
    n, k = probs.shape
    I = np.zeros((n, k, k))

    for i in range(n):
        p = probs[i]
        I[i] = np.diag(p) - np.outer(p, p)

    return I


def rescale_by_metric(X: np.ndarray, G: np.ndarray, reg: float = 1e-6) -> np.ndarray:
    """Rescale coordinates using Fisher-Rao metric.

    Parameters
    ----------
    X : np.ndarray, shape (n, d)
        Original coordinates
    G : np.ndarray, shape (n, d, d)
        Metric tensors
    reg : float
        Regularization for matrix square root

    Returns
    -------
    X_rescaled : np.ndarray, shape (n, d)
        Metric-aware coordinates

    Notes
    -----
    Rescaling: X_new = G^{1/2} X where G^{1/2} is matrix square root.
    This makes Euclidean distances in X_new approximate geodesics.
    """
    n, d = X.shape
    X_rescaled = np.zeros_like(X)

    for i in range(n):
        G_i = G[i] + reg * np.eye(d)  # Regularize

        # Compute matrix square root via eigendecomposition
        evals, evecs = np.linalg.eigh(G_i)
        evals = np.maximum(evals, reg)  # Ensure positive
        sqrt_G_i = evecs @ np.diag(np.sqrt(evals)) @ evecs.T

        X_rescaled[i] = sqrt_G_i @ X[i]

    return X_rescaled


def pullback_metric(
    embedding_func: Callable,
    X: np.ndarray,
    epsilon: float = 1e-6
) -> np.ndarray:
    """Compute pullback metric via finite differences.

    Parameters
    ----------
    embedding_func : callable
        Function X -> embeddings
    X : np.ndarray, shape (n, d)
        Input points
    epsilon : float
        Finite difference step size

    Returns
    -------
    G : np.ndarray, shape (n, d, d)
        Pullback metric tensors

    Notes
    -----
    Computes Jacobian via finite differences and pullback metric
    as G = J^T J where J is the embedding Jacobian.
    """
    n, d = X.shape
    G = np.zeros((n, d, d))

    for i in range(n):
        x = X[i]

        # Compute Jacobian via finite differences
        f_x = embedding_func(x.reshape(1, -1))
        embed_dim = f_x.shape[1]

        J = np.zeros((embed_dim, d))

        for j in range(d):
            x_plus = x.copy()
            x_plus[j] += epsilon
            f_plus = embedding_func(x_plus.reshape(1, -1))

            x_minus = x.copy()
            x_minus[j] -= epsilon
            f_minus = embedding_func(x_minus.reshape(1, -1))

            J[:, j] = (f_plus - f_minus).flatten() / (2 * epsilon)

        # Pullback metric: G = J^T J
        G[i] = J.T @ J

    return G


def riemannian_distance(
    x: np.ndarray,
    y: np.ndarray,
    G_x: np.ndarray,
    G_y: np.ndarray,
    method: str = "midpoint"
) -> float:
    """Approximate Riemannian distance using metric tensors.

    Parameters
    ----------
    x : np.ndarray, shape (d,)
        First point
    y : np.ndarray, shape (d,)
        Second point
    G_x : np.ndarray, shape (d, d)
        Metric tensor at x
    G_y : np.ndarray, shape (d, d)
        Metric tensor at y
    method : str
        Approximation method:
        - "midpoint": Use metric at midpoint
        - "average": Average of metrics

    Returns
    -------
    distance : float
        Approximate Riemannian distance

    Notes
    -----
    For small distances, the Riemannian distance is approximately:
    d(x,y) ≈ sqrt((x-y)^T G (x-y))
    where G is the metric tensor.
    """
    diff = y - x

    if method == "midpoint":
        # Linear interpolation of metrics (not geometrically correct but simple)
        G = 0.5 * (G_x + G_y)
    elif method == "average":
        G = 0.5 * (G_x + G_y)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Ensure positive definite
    evals = np.linalg.eigvals(G)
    if np.any(evals <= 0):
        G = G + 1e-6 * np.eye(len(diff))

    distance = np.sqrt(diff.T @ G @ diff)
    return distance


def fisher_rao_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Compute Fisher-Rao divergence between probability distributions.

    Parameters
    ----------
    p : np.ndarray, shape (k,)
        First probability distribution
    q : np.ndarray, shape (k,)
        Second probability distribution

    Returns
    -------
    divergence : float
        Fisher-Rao divergence

    Notes
    -----
    The Fisher-Rao divergence is the geodesic distance on the
    probability simplex with the Fisher-Rao metric.
    For discrete distributions: d_FR(p,q) = 2 * arccos(Σ sqrt(p_i q_i))
    """
    # Ensure probabilities are normalized and positive
    p = np.maximum(p / np.sum(p), 1e-12)
    q = np.maximum(q / np.sum(q), 1e-12)

    # Fisher-Rao distance
    inner_product = np.sum(np.sqrt(p * q))
    inner_product = np.clip(inner_product, 0, 1)  # Numerical safety

    divergence = 2 * np.arccos(inner_product)
    return divergence