"""
Expectation-Maximization (EM) Algorithm for Gaussian Mixture Models

A standalone implementation of the EM algorithm that fits a mixture of
Gaussian distributions to observed data. The algorithm iteratively:
  1. E-step: assigns soft responsibilities of each component for each data point
  2. M-step: updates component parameters to maximize expected log-likelihood

Usage:
    python em_algorithm.py
"""

import numpy as np
import scipy.stats


def expectation_maximization(data, n_components, max_iter=100, tol=1e-6):
    """Fit a Gaussian Mixture Model to 1-D data using the EM algorithm.

    Parameters
    ----------
    data : np.ndarray
        1-D array of observed values.
    n_components : int
        Number of Gaussian components (clusters) to fit.
    max_iter : int
        Maximum number of EM iterations.
    tol : float
        Convergence threshold on log-likelihood change.

    Returns
    -------
    weights : np.ndarray of shape (n_components,)
        Mixture weights (sum to 1).
    means : np.ndarray of shape (n_components,)
        Means of each Gaussian component.
    stds : np.ndarray of shape (n_components,)
        Standard deviations of each Gaussian component.
    log_likelihoods : list of float
        Log-likelihood at each iteration (useful for checking convergence).
    """
    n = len(data)

    # --- Initialisation -------------------------------------------------- #
    # Place initial means at evenly-spaced quantiles of the data so each
    # component starts near a different region of the distribution.
    quantiles = np.linspace(0, 1, n_components + 2)[1:-1]
    means = np.quantile(data, quantiles)

    stds = np.ones(n_components) * np.std(data) / n_components
    weights = np.ones(n_components) / n_components  # uniform start

    # Responsibility matrix: (n_components, n_data)
    resp = np.zeros((n_components, n))
    log_likelihoods = []

    for iteration in range(max_iter):
        # ---- E-step: compute responsibilities --------------------------- #
        for k in range(n_components):
            resp[k] = weights[k] * scipy.stats.norm.pdf(data, means[k], stds[k])
        total = resp.sum(axis=0)
        total = np.maximum(total, 1e-300)  # avoid division by zero
        resp /= total

        # ---- Log-likelihood --------------------------------------------- #
        ll = np.sum(np.log(np.maximum(
            sum(weights[k] * scipy.stats.norm.pdf(data, means[k], stds[k])
                for k in range(n_components)),
            1e-300)))
        log_likelihoods.append(ll)

        if len(log_likelihoods) > 1 and abs(log_likelihoods[-1] - log_likelihoods[-2]) < tol:
            break

        # ---- M-step: update parameters ---------------------------------- #
        Nk = resp.sum(axis=1)  # effective number of points per component

        weights = Nk / n
        means = (resp * data).sum(axis=1) / Nk
        stds = np.sqrt((resp * (data - means[:, np.newaxis]) ** 2).sum(axis=1) / Nk)
        stds = np.maximum(stds, 0.1)  # floor to prevent degenerate components

    return weights, means, stds, log_likelihoods


def generate_mixture_data(means, stds, weights, n_samples=2000, seed=None):
    """Generate samples from a 1-D Gaussian mixture.

    Parameters
    ----------
    means : list of float
        Mean of each component.
    stds : list of float
        Standard deviation of each component.
    weights : list of float
        Mixing proportions (must sum to 1).
    n_samples : int
        Total number of samples to draw.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    data : np.ndarray
        Generated samples.
    labels : np.ndarray
        Component index that generated each sample.
    """
    rng = np.random.default_rng(seed)
    labels = rng.choice(len(means), size=n_samples, p=weights)
    data = np.array([rng.normal(means[l], stds[l]) for l in labels])
    return data, labels


# --------------------------------------------------------------------------- #
#  Quick demo when run as a script                                             #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    print("=" * 60)
    print("  EM Algorithm — Gaussian Mixture Model Demo")
    print("=" * 60)

    # Ground truth
    true_means = [-8.0, 10.0]
    true_stds = [2.0, 2.0]
    true_weights = [0.6, 0.4]

    print("\nGround truth:")
    for i, (m, s, w) in enumerate(zip(true_means, true_stds, true_weights)):
        print(f"  Component {i+1}: mean={m:+.1f}, std={s:.1f}, weight={w:.2f}")

    # Generate data
    data, _ = generate_mixture_data(true_means, true_stds, true_weights,
                                    n_samples=2000, seed=42)

    # Fit
    weights, means, stds, lls = expectation_maximization(data, n_components=2)

    # Sort components by mean for consistent display
    order = np.argsort(means)
    means, stds, weights = means[order], stds[order], weights[order]

    print(f"\nEM converged after {len(lls)} iterations.")
    print("\nLearned parameters:")
    for i, (m, s, w) in enumerate(zip(means, stds, weights)):
        print(f"  Component {i+1}: mean={m:+.2f}, std={s:.2f}, weight={w:.2f}")

    print(f"\nFinal log-likelihood: {lls[-1]:.2f}")
    print("=" * 60)
