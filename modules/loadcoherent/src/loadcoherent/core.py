"""Small dense systems; Cholesky whitening followed by reduced QR.

All forecasts have nodes on their final axis. No explicit covariance inverse or
normal-equation matrix is formed. The API deliberately rejects singular weights.
"""

import numpy as np
from scipy.linalg import solve_triangular


def _array(value, name, ndim=None):
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    array = np.array(value, dtype=float, copy=True)
    if (ndim is not None and array.ndim != ndim) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite with ndim={ndim}")
    return array


def _covariance(value, size, name, positive_definite):
    cov = _array(value, name, 2)
    if cov.shape != (size, size):
        raise ValueError(f"{name} must have shape {(size, size)}")
    scale = np.max(np.abs(cov))
    if not np.allclose(cov, cov.T, rtol=0, atol=1e-12 * scale):
        raise ValueError(f"{name} must be symmetric")
    cov = 0.5 * (cov + cov.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    if positive_definite:
        if scale == 0 or eigenvalues[0] <= 1e-12 * eigenvalues[-1]:
            raise ValueError(f"{name} must be positive definite and well conditioned")
    elif eigenvalues[0] < -1e-12 * scale:
        raise ValueError(f"{name} must be positive semidefinite")
    return cov


class Reconciler:
    """Fit a projection onto range(S) using an SPD error weight W.

    Parameters
    ----------
    summing : array, shape (n, m)
        Full-column-rank aggregation matrix, with n >= m >= 1.
    covariance : array, shape (n, n)
        Symmetric positive-definite weight. It need not be the true covariance.

    Notes
    -----
    Rejects relative covariance eigenvalues <= 1e-12 and whitened design
    singular values <= 1e-12 of the largest. These are safeguards, not error
    bounds. Coherence does not guarantee positivity, unbiasedness of the base
    forecasts, or accuracy. The numerical core is O(n^3); intended for small
    dense hierarchies, not national-scale sparse networks.
    """

    def __init__(self, summing, covariance):
        s = _array(summing, "summing", 2)
        n, m = s.shape
        if m < 1 or n < m:
            raise ValueError("summing needs n >= m >= 1")
        w = _covariance(covariance, n, "covariance", True)
        lower = np.linalg.cholesky(w)
        a = solve_triangular(lower, s, lower=True)
        singular = np.linalg.svd(a, compute_uv=False)
        if singular[-1] <= 1e-12 * singular[0]:
            raise ValueError("whitened summing matrix must have full column rank")
        q, r = np.linalg.qr(a, mode="reduced")
        # Q^T L^{-1} = (L^{-T} Q)^T, evaluated by triangular solve.
        rhs = solve_triangular(lower.T, q, lower=False).T
        self._bottom_map = solve_triangular(r, rhs)
        self._projection = s @ self._bottom_map
        self._summing = s
        self._weight = w
        self.design_condition = float(singular[0] / singular[-1])

    @property
    def projection(self):
        """Return an independent copy of the n by n projection."""
        return self._projection.copy()

    @property
    def bottom_map(self):
        """Return an independent copy of the m by n coefficient map."""
        return self._bottom_map.copy()

    def reconcile(self, forecasts):
        """Return coherent forecasts for a vector or a batch (..., n)."""
        forecasts = _array(forecasts, "forecasts")
        if forecasts.ndim < 1 or forecasts.shape[-1] != self._projection.shape[0]:
            raise ValueError("forecasts must have nodes on the final axis")
        # Applying S last keeps the result in the represented coherent space.
        return (forecasts @ self._bottom_map.T) @ self._summing.T


def estimate_covariance(errors, shrinkage=0.5):
    """Centered sample covariance shrunk toward its diagonal.

    W_alpha = (1-alpha) sample_cov + alpha diag(sample_cov).
    Alpha is a user-specified constant in [0, 1], not an estimated optimum.
    Rows must be training forecast-error samples, columns are nodes. Requires
    at least two rows and positive variance at every node. Alpha=0 can return
    a singular covariance; Reconciler will reject it rather than silently jitter.
    Sample centering does not correct forecast bias.
    """
    errors = _array(errors, "errors", 2)
    if errors.shape[0] < 2 or errors.shape[1] < 1:
        raise ValueError("errors needs at least two rows and one column")
    if not np.isscalar(shrinkage) or not np.isfinite(shrinkage) or not 0 <= shrinkage <= 1:
        raise ValueError("shrinkage must be a finite scalar in [0, 1]")
    centered = errors - errors.mean(axis=0)
    sample = centered.T @ centered / (errors.shape[0] - 1)
    if np.any(np.diag(sample) <= 0) or not np.all(np.isfinite(sample)):
        raise ValueError("each node must have positive finite sample variance")
    return (1 - shrinkage) * sample + shrinkage * np.diag(np.diag(sample))


def prediction_mse(projection, true_covariance, bias=None):
    """Per-node expected squared error, conditional on a fixed projection.

    Returns diag(P V P^T) + (P bias)^2, assuming P preserves the true signal.
    This function cannot verify that signal assumption from covariance alone.
    Bias uses the sign convention base forecast minus truth.
    """
    p = _array(projection, "projection", 2)
    if p.shape[0] < 1 or p.shape[0] != p.shape[1]:
        raise ValueError("projection must be a nonempty square matrix")
    v = _covariance(true_covariance, p.shape[0], "true_covariance", False)
    mu = np.zeros(p.shape[0]) if bias is None else _array(bias, "bias", 1)
    if mu.shape != (p.shape[0],):
        raise ValueError("bias has incompatible shape")
    variance = np.einsum("ij,jk,ik->i", p, v, p)
    return np.maximum(variance, 0) + (p @ mu) ** 2
