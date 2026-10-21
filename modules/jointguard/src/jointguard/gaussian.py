"""Gaussian forecasts, exact expectations, and linear elastic propagation.

Scores use a loss convention: lower is better. All dimensions should first be
expressed on a common, fixed scale; the experiment uses displacement / 1 mm.
"""

from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad
from scipy.linalg import cho_solve, solve_triangular
from scipy.special import ndtr


def _vector(value, name):
    a = np.array(value, dtype=float, copy=True)
    if a.ndim != 1 or not a.size or not np.isfinite(a).all():
        raise ValueError(f"{name} must be a finite nonempty vector")
    return a


def _spd(value, d, name):
    a = np.array(value, dtype=float, copy=True)
    if a.shape != (d, d) or not np.isfinite(a).all():
        raise ValueError(f"{name} must be a finite square matrix matching the mean")
    if not np.allclose(a, a.T, rtol=1e-12, atol=1e-14):
        raise ValueError(f"{name} must be symmetric")
    a = (a + a.T) / 2
    try:
        np.linalg.cholesky(a)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"{name} must be positive definite") from exc
    return a


@dataclass(frozen=True)
class Gaussian:
    """Nonsingular Gaussian in a fixed coordinate system; arrays are read-only."""

    mean: np.ndarray
    covariance: np.ndarray

    def __post_init__(self):
        mean = _vector(self.mean, "mean")
        covariance = _spd(self.covariance, mean.size, "covariance")
        mean.setflags(write=False)
        covariance.setflags(write=False)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "covariance", covariance)

    def _observations(self, y):
        y = np.asarray(y, dtype=float)
        if y.ndim != 2 or y.shape[1] != self.mean.size or not np.isfinite(y).all():
            raise ValueError("observations must be finite (cases, dimension)")
        return y

    def log_score(self, y):
        """Negative log density per case, using Cholesky rather than an inverse."""
        y = self._observations(y)
        chol = np.linalg.cholesky(self.covariance)
        z = solve_triangular(chol, (y - self.mean).T, lower=True)
        return (
            self.mean.size * np.log(2 * np.pi)
            + 2 * np.log(np.diag(chol)).sum()
            + (z * z).sum(axis=0)
        ) / 2

    def marginal_crps(self, y):
        """Gaussian CRPS averaged across coordinates per observation."""
        y = self._observations(y)
        sd = np.sqrt(np.diag(self.covariance))
        z = (y - self.mean) / sd
        return np.mean(
            sd
            * (
                z * (2 * ndtr(z) - 1)
                + 2 * np.exp(-z * z / 2) / np.sqrt(2 * np.pi)
                - 1 / np.sqrt(np.pi)
            ),
            axis=1,
        )

    def variogram_score(self, y):
        """p=1 score, average over unordered pairs, exact Gaussian moments.

        A uniform shift of all forecast coordinates is invisible to this score.
        """
        y = self._observations(y)
        i, j = _pairs(self.mean.size)
        moment = self.difference_moments()
        return np.mean((np.abs(y[:, i] - y[:, j]) - moment) ** 2, axis=1)

    def difference_moments(self):
        i, j = _pairs(self.mean.size)
        variance = self.covariance[i, i] + self.covariance[j, j] - 2 * self.covariance[i, j]
        return absolute_normal_mean(self.mean[i] - self.mean[j], variance)

    def exceedance(self, direction, threshold):
        """P(|direction @ X| > threshold), an absolute linear-response event."""
        a = _vector(direction, "direction")
        if a.shape != self.mean.shape or not np.any(a):
            raise ValueError("direction must be nonzero and match mean")
        if not np.isfinite(threshold) or threshold < 0:
            raise ValueError("threshold must be finite and nonnegative")
        mu, sd = a @ self.mean, np.sqrt(a @ self.covariance @ a)
        return float(ndtr((-threshold - mu) / sd) + ndtr((mu - threshold) / sd))


def _pairs(d):
    if d < 2:
        raise ValueError("variograms require at least two dimensions")
    return np.triu_indices(d, 1)


def absolute_normal_mean(mean, variance):
    """E|Z| for Gaussian Z, including zero variance."""
    mean, variance = np.broadcast_arrays(np.asarray(mean, float), np.asarray(variance, float))
    if not np.isfinite(mean).all() or not np.isfinite(variance).all() or np.any(variance < 0):
        raise ValueError("finite mean and nonnegative variance required")
    sd = np.sqrt(variance)
    z = np.divide(mean, sd, out=np.zeros_like(mean), where=sd > 0)
    value = sd * np.sqrt(2 / np.pi) * np.exp(-z * z / 2) + mean * (2 * ndtr(z) - 1)
    return np.where(sd == 0, np.abs(mean), value)


def expected_norm(distribution):
    """E||X|| via the Gaussian Laplace transform and adaptive quadrature.

    Returns (value, quadrature error estimate). The estimate is not a certified
    bound. Positive definite covariance is required by Gaussian.
    """
    scale = max(np.sqrt(np.diag(distribution.covariance).max()), np.abs(distribution.mean).max())
    lam, vectors = np.linalg.eigh(distribution.covariance / scale**2)
    coordinates = vectors.T @ (distribution.mean / scale)

    def integrand(x):
        if x == 0:
            return float(lam.sum() + coordinates @ coordinates) / np.sqrt(np.pi)
        t = x * x
        log_transform = -0.5 * np.log1p(2 * t * lam).sum()
        log_transform -= np.sum(t * coordinates**2 / (1 + 2 * t * lam))
        return -np.expm1(log_transform) / (t * np.sqrt(np.pi))

    value, error = quad(integrand, 0, np.inf, epsabs=2e-11, epsrel=2e-11)
    return scale * value, scale * error


def expected_scores(forecast, truth):
    """Population losses under a declared Gaussian truth, no Monte Carlo noise.

    Energy uses scalar numerical quadrature; other expectations are closed form.
    """
    if forecast.mean.shape != truth.mean.shape:
        raise ValueError("forecast and truth dimensions must match")
    delta = forecast.mean - truth.mean
    chol = np.linalg.cholesky(forecast.covariance)
    trace = np.trace(cho_solve((chol, True), truth.covariance))
    bias = delta @ cho_solve((chol, True), delta)
    log = (delta.size * np.log(2 * np.pi) + 2 * np.log(np.diag(chol)).sum() + trace + bias) / 2
    crps = np.mean(
        absolute_normal_mean(delta, np.diag(forecast.covariance + truth.covariance))
        - np.sqrt(np.diag(forecast.covariance) / np.pi)
    )
    i, j = _pairs(delta.size)
    truth_second = (truth.mean[i] - truth.mean[j]) ** 2 + truth.covariance[i, i]
    truth_second += truth.covariance[j, j] - 2 * truth.covariance[i, j]
    fm, tm = forecast.difference_moments(), truth.difference_moments()
    vs = np.mean(truth_second - 2 * fm * tm + fm * fm)
    cross, err_cross = expected_norm(Gaussian(delta, forecast.covariance + truth.covariance))
    within, err_within = expected_norm(Gaussian(np.zeros_like(delta), 2 * forecast.covariance))
    return {
        "marginal_crps": float(crps),
        "log_score": float(log),
        "variogram": float(vs),
        "energy": float(cross - within / 2),
        "energy_quad_error_estimate": float(err_cross + err_within / 2),
    }


def elastic_response(stiffness, force):
    """Propagate Gaussian forces through K u = f (SPD K); SI units recommended."""
    k = _spd(stiffness, force.mean.size, "stiffness")
    mean = np.linalg.solve(k, force.mean)
    left = np.linalg.solve(k, force.covariance)
    covariance = np.linalg.solve(k, left.T).T
    return Gaussian(mean, covariance)
