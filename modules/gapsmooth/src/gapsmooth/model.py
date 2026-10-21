"""Exact transition moments of a mass-normalized damped oscillator."""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm


@dataclass(frozen=True)
class Oscillator:
    """dx=v dt, dv=(-omega**2*x-2*zeta*omega*v)dt+sqrt(q)dW.

    omega is angular frequency (rad/s), zeta is damping ratio, and q is
    acceleration noise spectral intensity (length**2/time**3).
    """

    omega: float = 2 * np.pi
    zeta: float = 0.08
    q: float = 0.6

    def __post_init__(self):
        if not np.isfinite([self.omega, self.zeta, self.q]).all():
            raise ValueError("parameters must be finite")
        if self.omega <= 0 or self.zeta <= 0 or self.q < 0:
            raise ValueError("require omega>0, zeta>0, q>=0")
        if not np.isfinite(self.drift).all():
            raise ValueError("parameters overflow the drift matrix")

    @property
    def drift(self):
        return np.array([[0.0, 1.0], [-(self.omega**2), -2 * self.zeta * self.omega]])

    @property
    def stationary_covariance(self):
        """Analytical invariant covariance, including q=0."""
        v = self.q / (4 * self.zeta * self.omega)
        return np.diag([v / self.omega**2, v])

    def transition(self, dt):
        """Return exact F(dt), Q(dt), up to floating-point matrix algebra.

        Evaluate a small Van Loan block exponential, then compose transitions
        by doubling. This avoids exponential growth of the auxiliary block
        at long time intervals. Zero duration is the identity transition.
        """
        if not np.isfinite(dt) or dt < 0:
            raise ValueError("dt must be finite and nonnegative")
        if dt == 0:
            return np.eye(2), np.zeros((2, 2))
        a = self.drift
        scale = float(np.linalg.norm(a, ord=np.inf))
        # Logs avoid overflow in scale*dt.
        doublings = max(0, int(np.ceil(np.log2(scale) + np.log2(dt))))
        h = np.ldexp(float(dt), -doublings)
        diffusion = np.diag([0.0, self.q])
        block = np.block([[a, diffusion], [np.zeros((2, 2)), -a.T]])
        e = expm(block * h)
        f = e[:2, :2]
        q = e[:2, 2:] @ f.T
        for _ in range(doublings):
            q = f @ q @ f.T + q
            f = f @ f
        q = (q + q.T) / 2
        if not np.isfinite(f).all() or not np.isfinite(q).all():
            raise FloatingPointError("transition moments overflowed")
        return f, q


def time_grid(times):
    """Validate a nonempty, finite, strictly increasing one-dimensional grid."""
    t = np.asarray(times, dtype=float)
    if t.ndim != 1 or not t.size or not np.isfinite(t).all():
        raise ValueError("times must be a finite nonempty vector")
    if np.any(np.diff(t) <= 0):
        raise ValueError("times must be strictly increasing")
    return t


def prior(mean, covariance):
    """Validate a finite 2D Gaussian prior; require positive definiteness."""
    m, p = np.asarray(mean, dtype=float), np.asarray(covariance, dtype=float)
    if m.shape != (2,) or p.shape != (2, 2):
        raise ValueError("prior shapes must be (2,) and (2,2)")
    if not np.isfinite(m).all() or not np.isfinite(p).all():
        raise ValueError("prior must be finite")
    if not np.allclose(p, p.T, rtol=1e-12, atol=1e-14):
        raise ValueError("prior covariance must be symmetric")
    p = (p + p.T) / 2
    try:
        np.linalg.cholesky(p)
    except np.linalg.LinAlgError as e:
        raise ValueError("prior covariance must be positive definite") from e
    return m.copy(), p.copy()
