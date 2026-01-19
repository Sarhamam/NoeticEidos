"""Fisher-Rao gradient flows for probabilistic models."""

import numpy as np
from scipy.special import softmax, logsumexp
from typing import Callable, Tuple, List, Dict, Any, Optional
import time

from geometry.fr_pullback import fisher_rao_metric, multinomial_fisher_info


def fr_gradient_flow(
    logits: np.ndarray,
    dlogits_dX: np.ndarray,
    F: Callable,
    steps: int = 50,
    eta: float = 0.01,
    adaptive_step: bool = False,
    verbose: bool = False
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """Evolve embeddings via Fisher-Rao gradient flow.

    Parameters
    ----------
    logits : np.ndarray, shape (n, k)
        Model logits (pre-softmax)
    dlogits_dX : np.ndarray, shape (n, k, d)
        Jacobian of logits w.r.t. input features X
    F : callable
        Functional F(logits) to minimize
    steps : int
        Number of flow steps
    eta : float
        Step size
    adaptive_step : bool
        Use adaptive step sizing
    verbose : bool
        Print progress

    Returns
    -------
    trajectory : list of np.ndarray
        Sequence of embedding points
    info : dict
        Flow information

    Notes
    -----
    Implements the Fisher-Rao gradient flow:
        dX/dt = -G^{-1} ∇_X F
    where G is the Fisher-Rao metric pullback.
    """
    n, k = logits.shape
    d = dlogits_dX.shape[2]

    # Initial embedding (will be updated implicitly through logits)
    X_current = np.zeros((n, d))  # Placeholder - in practice would be actual embeddings
    trajectory = [X_current.copy()]

    functional_values = []
    step_sizes = []

    logits_current = logits.copy()

    for iteration in range(steps):
        # Compute Fisher-Rao metric
        G = fisher_rao_metric(logits_current, dlogits_dX)

        # Compute gradient of functional w.r.t. logits
        grad_F_logits = finite_diff_gradient(F, logits_current)

        # Pullback to feature space: grad_F_X = (∂logits/∂X)^T grad_F_logits
        grad_F_X = np.zeros((n, d))
        for i in range(n):
            grad_F_X[i] = dlogits_dX[i].T @ grad_F_logits[i]

        # Apply Fisher-Rao metric (solve G @ direction = grad for direction)
        fr_direction = np.zeros((n, d))
        for i in range(n):
            try:
                # Solve G[i] @ direction = grad_F_X[i]
                fr_direction[i] = np.linalg.solve(G[i] + 1e-6 * np.eye(d), grad_F_X[i])
            except np.linalg.LinAlgError:
                # Fallback to pseudo-inverse
                fr_direction[i] = np.linalg.pinv(G[i]) @ grad_F_X[i]

        # Adaptive step size
        if adaptive_step:
            # Simple backtracking line search
            current_F = F(logits_current)
            eta_trial = eta
            for _ in range(5):  # Max 5 backtracks
                X_trial = X_current - eta_trial * fr_direction
                logits_trial = update_logits_from_features(X_trial, logits_current, dlogits_dX)
                trial_F = F(logits_trial)

                if trial_F < current_F:
                    eta_effective = eta_trial
                    break
                eta_trial *= 0.5
            else:
                eta_effective = eta_trial
        else:
            eta_effective = eta

        # Update embeddings
        X_current = X_current - eta_effective * fr_direction

        # Update logits (linear approximation)
        logits_current = update_logits_from_features(X_current, logits, dlogits_dX)

        # Store results
        trajectory.append(X_current.copy())
        functional_values.append(F(logits_current))
        step_sizes.append(eta_effective)

        if verbose and iteration % 10 == 0:
            print(f"FR flow iter {iteration:3d}: F={functional_values[-1]:.4f}, "
                  f"step={eta_effective:.4f}")

    info = {
        'functional_values': functional_values,
        'step_sizes': step_sizes,
        'converged': len(functional_values) > 1 and
                    abs(functional_values[-1] - functional_values[-2]) < 1e-6
    }

    return trajectory, info


def multinomial_nll_flow(
    logits: np.ndarray,
    dlogits_dX: np.ndarray,
    y_true: np.ndarray,
    steps: int = 50,
    eta: float = 0.01
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """Fisher-Rao flow for multinomial negative log-likelihood.

    Parameters
    ----------
    logits : np.ndarray, shape (n, k)
        Model logits
    dlogits_dX : np.ndarray, shape (n, k, d)
        Logit Jacobians
    y_true : np.ndarray, shape (n,) or (n, k)
        True labels (indices or one-hot)
    steps : int
        Number of steps
    eta : float
        Step size

    Returns
    -------
    trajectory : list
        Embedding trajectory
    info : dict
        Flow information
    """
    # Convert labels to one-hot if needed
    if y_true.ndim == 1:
        n, k = logits.shape
        y_onehot = np.zeros((n, k))
        y_onehot[np.arange(n), y_true] = 1.0
    else:
        y_onehot = y_true

    def nll_functional(logits_):
        """Negative log-likelihood functional."""
        log_probs = logits_ - logsumexp(logits_, axis=1, keepdims=True)
        return -np.sum(y_onehot * log_probs)

    return fr_gradient_flow(logits, dlogits_dX, nll_functional,
                           steps=steps, eta=eta)


def natural_gradient_descent(
    params: np.ndarray,
    grad_func: Callable,
    fisher_func: Callable,
    steps: int = 100,
    eta: float = 0.01,
    verbose: bool = False
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """Natural gradient descent using Fisher information.

    Parameters
    ----------
    params : np.ndarray
        Initial parameters
    grad_func : callable
        Function computing gradient
    fisher_func : callable
        Function computing Fisher information matrix
    steps : int
        Number of steps
    eta : float
        Step size
    verbose : bool
        Print progress

    Returns
    -------
    trajectory : list
        Parameter trajectory
    info : dict
        Optimization information
    """
    trajectory = [params.copy()]
    gradients = []

    params_current = params.copy()

    for iteration in range(steps):
        # Compute gradient and Fisher information
        grad = grad_func(params_current)
        F_info = fisher_func(params_current)

        # Natural gradient: F^{-1} grad
        try:
            nat_grad = np.linalg.solve(F_info + 1e-6 * np.eye(len(params_current)), grad)
        except np.linalg.LinAlgError:
            nat_grad = np.linalg.pinv(F_info) @ grad

        # Update parameters
        params_current = params_current - eta * nat_grad

        trajectory.append(params_current.copy())
        gradients.append(np.linalg.norm(grad))

        if verbose and iteration % 20 == 0:
            print(f"Natural GD iter {iteration:3d}: |grad|={gradients[-1]:.4f}")

    info = {
        'gradient_norms': gradients,
        'converged': len(gradients) > 1 and gradients[-1] < 1e-6
    }

    return trajectory, info


def exponential_family_flow(
    natural_params: np.ndarray,
    sufficient_stats: np.ndarray,
    observations: np.ndarray,
    steps: int = 50,
    eta: float = 0.01
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """Natural gradient flow for exponential family models.

    Parameters
    ----------
    natural_params : np.ndarray
        Natural parameters θ
    sufficient_stats : np.ndarray
        Sufficient statistics T(x)
    observations : np.ndarray
        Observed data
    steps : int
        Number of flow steps
    eta : float
        Step size

    Returns
    -------
    trajectory : list
        Natural parameter trajectory
    info : dict
        Flow information

    Notes
    -----
    For exponential families, the Fisher information matrix is
    the covariance matrix of sufficient statistics.
    """
    trajectory = [natural_params.copy()]
    log_likelihoods = []

    theta = natural_params.copy()

    for iteration in range(steps):
        # Compute expected sufficient statistics (model prediction)
        expected_stats = compute_expected_sufficient_stats(theta, sufficient_stats)

        # Compute empirical sufficient statistics
        empirical_stats = np.mean(sufficient_stats, axis=0)

        # Gradient of log-likelihood
        grad = empirical_stats - expected_stats

        # Fisher information (covariance of sufficient statistics)
        F_info = compute_fisher_information_exp_family(theta, sufficient_stats)

        # Natural gradient step
        try:
            nat_grad = np.linalg.solve(F_info, grad)
        except np.linalg.LinAlgError:
            nat_grad = np.linalg.pinv(F_info) @ grad

        theta = theta + eta * nat_grad

        trajectory.append(theta.copy())

        # Compute log-likelihood for monitoring
        ll = compute_log_likelihood_exp_family(theta, sufficient_stats, observations)
        log_likelihoods.append(ll)

    info = {
        'log_likelihoods': log_likelihoods,
        'converged': len(log_likelihoods) > 1 and
                    abs(log_likelihoods[-1] - log_likelihoods[-2]) < 1e-6
    }

    return trajectory, info


# Helper functions

def finite_diff_gradient(f: Callable, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Compute gradient via finite differences."""
    grad = np.zeros_like(x)
    f_x = f(x)

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x_plus = x.copy()
            x_plus[i, j] += eps
            grad[i, j] = (f(x_plus) - f_x) / eps

    return grad


def update_logits_from_features(X: np.ndarray, logits_base: np.ndarray,
                               dlogits_dX: np.ndarray) -> np.ndarray:
    """Update logits using linear approximation from feature changes."""
    n, k = logits_base.shape
    d = X.shape[1]

    logits_new = logits_base.copy()
    for i in range(n):
        # Linear update: logits_new[i] = logits_base[i] + J[i] @ (X[i] - X_base[i])
        # Since we don't have X_base, assume it's zero for simplicity
        logits_new[i] = logits_base[i] + dlogits_dX[i] @ X[i]

    return logits_new


def compute_expected_sufficient_stats(natural_params: np.ndarray,
                                    sufficient_stats: np.ndarray) -> np.ndarray:
    """Compute expected sufficient statistics for exponential family."""
    # Placeholder implementation - depends on specific exponential family
    # For now, return empirical mean (should be replaced with proper expectation)
    return np.mean(sufficient_stats, axis=0)


def compute_fisher_information_exp_family(natural_params: np.ndarray,
                                        sufficient_stats: np.ndarray) -> np.ndarray:
    """Compute Fisher information for exponential family."""
    # Placeholder - should compute covariance of sufficient statistics
    return np.cov(sufficient_stats.T) + 1e-6 * np.eye(sufficient_stats.shape[1])


def compute_log_likelihood_exp_family(natural_params: np.ndarray,
                                     sufficient_stats: np.ndarray,
                                     observations: np.ndarray) -> float:
    """Compute log-likelihood for exponential family."""
    # Placeholder implementation
    return np.sum(natural_params @ sufficient_stats.T)