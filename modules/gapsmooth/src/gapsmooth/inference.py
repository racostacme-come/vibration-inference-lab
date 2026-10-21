"""Scalar displacement observations, Joseph filtering and RTS smoothing."""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve

from .model import Oscillator, prior, time_grid


@dataclass
class Estimate:
    """Moments at observation times (including unobserved NaN entries).

    innovations and innovation_variances are NaN at missing samples.
    nll sums only observed scalar Gaussian conditional log densities.
    """

    filtered_mean: np.ndarray
    filtered_covariance: np.ndarray
    smoothed_mean: np.ndarray
    smoothed_covariance: np.ndarray
    innovations: np.ndarray
    innovation_variances: np.ndarray
    nll: float


def infer(model: Oscillator, times, observations, variance, mean0, covariance0):
    """Filter and smooth; prior is at times[0], before its measurement.

    observations has shape (N,); NaN means missing, infinity is invalid.
    variance is a strictly positive scalar measurement variance. No implicit
    interpolation, parameter fitting, or zero-filling of missing values occurs.
    """
    t = time_grid(times)
    y = np.asarray(observations, dtype=float)
    if y.shape != t.shape or np.isinf(y).any():
        raise ValueError("observations must match times; only NaN may be missing")
    if not np.isscalar(variance) or not np.isfinite(variance) or variance <= 0:
        raise ValueError("measurement variance must be a finite positive scalar")
    m, p = prior(mean0, covariance0)
    n = len(t)
    mf, mp = np.empty((n, 2)), np.empty((n, 2))
    pf, pp = np.empty((n, 2, 2)), np.empty((n, 2, 2))
    transitions = np.empty((n, 2, 2))
    innovations, variances = np.full(n, np.nan), np.full(n, np.nan)
    nll = 0.0
    cache = {}
    for k in range(n):
        if k:
            dt = float(t[k] - t[k - 1])
            if dt not in cache:
                cache[dt] = model.transition(dt)
            f, q = cache[dt]
            transitions[k] = f
            m, p = f @ m, f @ p @ f.T + q
        mp[k], pp[k] = m, p
        if not np.isnan(y[k]):
            innovation, s = y[k] - m[0], p[0, 0] + variance
            gain = p[:, 0] / s
            m = m + gain * innovation
            j = np.eye(2)
            j[:, 0] -= gain
            p = j @ p @ j.T + variance * np.outer(gain, gain)
            innovations[k], variances[k] = innovation, s
            nll += 0.5 * (np.log(2 * np.pi * s) + innovation**2 / s)
        p = (p + p.T) / 2
        mf[k], pf[k] = m, p
    ms, ps = mf.copy(), pf.copy()
    for k in range(n - 2, -1, -1):
        gain = solve(pp[k + 1], transitions[k + 1] @ pf[k], assume_a="pos").T
        ms[k] += gain @ (ms[k + 1] - mp[k + 1])
        ps[k] += gain @ (ps[k + 1] - pp[k + 1]) @ gain.T
        ps[k] = (ps[k] + ps[k].T) / 2
    return Estimate(mf, pf, ms, ps, innovations, variances, float(nll))


def simulate(model, times, variance, mean0, covariance0, seed=0):
    """Generate exact-transition synthetic states and noisy displacement samples."""
    t = time_grid(times)
    m, p = prior(mean0, covariance0)
    if not np.isfinite(variance) or variance <= 0:
        raise ValueError("variance must be finite and positive")
    rng = np.random.default_rng(seed)
    states = np.empty((len(t), 2))
    states[0] = rng.multivariate_normal(m, p)
    cache = {}
    for k, dt in enumerate(np.diff(t), start=1):
        dt = float(dt)
        if dt not in cache:
            cache[dt] = model.transition(dt)
        f, q = cache[dt]
        states[k] = f @ states[k - 1] + rng.multivariate_normal(np.zeros(2), q)
    y = states[:, 0] + rng.normal(0, np.sqrt(variance), len(t))
    return states, y
