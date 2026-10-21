"""Dense joint-Gaussian reference for short-record verification only."""

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from .model import prior, time_grid


def batch_condition(model, times, observations, variance, mean0, covariance0):
    """Independently condition the full joint state Gaussian; O(N^3) work.

    Return full posterior mean, covariance, and observation marginal NLL.
    Limit record size to 256 to avoid accidental quadratic allocation.
    """
    t = time_grid(times)
    if len(t) > 256:
        raise ValueError("batch reference limited to 256 times")
    y = np.asarray(observations, dtype=float)
    if y.shape != t.shape or np.isinf(y).any():
        raise ValueError("invalid observations")
    if not np.isfinite(variance) or variance <= 0:
        raise ValueError("variance must be finite and positive")
    m0, p0 = prior(mean0, covariance0)
    n = len(t)
    mean, cov = np.zeros(2 * n), np.zeros((2 * n, 2 * n))
    mean[:2], cov[:2, :2] = m0, p0
    for k in range(1, n):
        f, q = model.transition(t[k] - t[k - 1])
        now, prev = slice(2 * k, 2 * k + 2), slice(2 * k - 2, 2 * k)
        mean[now] = f @ mean[prev]
        cov[now, : 2 * k] = f @ cov[prev, : 2 * k]
        cov[: 2 * k, now] = cov[now, : 2 * k].T
        cov[now, now] = f @ cov[prev, prev] @ f.T + q
    observed = np.flatnonzero(~np.isnan(y))
    if not len(observed):
        return mean.reshape(n, 2), cov, 0.0
    ids = 2 * observed
    s = cov[np.ix_(ids, ids)] + variance * np.eye(len(ids))
    factor = cho_factor(s, lower=True)
    residual = y[observed] - mean[ids]
    cross = cov[:, ids]
    post_mean = mean + cross @ cho_solve(factor, residual)
    post_cov = cov - cross @ cho_solve(factor, cross.T)
    nll = 0.5 * (
        len(ids) * np.log(2 * np.pi)
        + 2 * np.log(np.diag(factor[0])).sum()
        + residual @ cho_solve(factor, residual)
    )
    return post_mean.reshape(n, 2), (post_cov + post_cov.T) / 2, float(nll)
