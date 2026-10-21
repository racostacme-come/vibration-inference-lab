"""Physical model, matrix-free operator, and checked iterative inversion."""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm
from scipy.sparse.linalg import LinearOperator, lsqr

from ._core import Recurrence


def _positive(value, name, zero=False):
    if not np.isscalar(value) or not np.isreal(value) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite real scalar")
    if value < 0 or (not zero and value == 0):
        raise ValueError(f"{name} must be {'nonnegative' if zero else 'positive'}")


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _vector(value, length):
    arr = np.asarray(value)
    if arr.shape not in [(length,), (length, 1)] or np.iscomplexobj(arr):
        raise ValueError(f"expected real vector of length {length}")
    arr = np.asarray(arr, dtype=float).reshape(length)
    if not np.all(np.isfinite(arr)):
        raise ValueError("vector must be finite")
    return arr


@dataclass(frozen=True)
class Oscillator:
    """SI parameters of m*q'' + c*q' + k*q = f, initially at rest."""

    mass: float = 1.0
    damping: float = 0.8
    stiffness: float = 100.0

    def __post_init__(self):
        _positive(self.mass, "mass")
        _positive(self.damping, "damping", zero=True)
        _positive(self.stiffness, "stiffness", zero=True)

    def discretize(self, dt):
        """Block exponential: valid also at zero stiffness or critical damping."""
        _positive(dt, "dt")
        block = np.array(
            [
                [0, 1, 0],
                [-self.stiffness / self.mass, -self.damping / self.mass, 1 / self.mass],
                [0, 0, 0],
            ],
            dtype=float,
        )
        transition = expm(dt * block)
        if not np.all(np.isfinite(transition)):
            raise ValueError("parameters exceed the numerical range")
        return transition[:2, :2], transition[:2, 2]


class Window(LinearOperator):
    """Map N held forces to displacement at k*dt, k=0,...,N+tail-1.

    Loads act on [j*dt,(j+1)*dt); tail assumes force is identically zero
    after N*dt. The final load is completely unobserved if tail=0.
    Zero initial state is required. Each matvec/rmatvec uses linear storage
    including its input/output; it never assembles the dense impulse matrix.
    """

    def __init__(self, oscillator, dt, loads, tail=0):
        _integer(loads, "loads", 1)
        _integer(tail, "tail")
        if loads + tail > 10_000_000:
            raise ValueError("loads + tail exceeds 10000000")
        self.a, self.b = oscillator.discretize(dt)
        self.core = Recurrence(self.a.ravel().tolist(), self.b.tolist(), loads, tail)
        super().__init__(dtype=np.dtype(float), shape=(loads + tail, loads))

    def _matvec(self, value):
        return np.asarray(self.core.forward(_vector(value, self.shape[1]).tolist()))

    def _rmatvec(self, value):
        return np.asarray(self.core.adjoint(_vector(value, self.shape[0]).tolist()))

    def dense(self):
        """Independent Markov-parameter assembly for small validation problems."""
        rows, cols = self.shape
        if rows * cols > 4_000_000:
            raise ValueError("dense validation limited to 4000000 entries")
        impulse = np.zeros(rows)
        state = self.b.copy()
        for lag in range(1, rows):
            impulse[lag] = state[0]
            state = self.a @ state
        i, j = np.indices((rows, cols))
        return impulse[np.maximum(i - j, 0)] * (i > j)


def reconstruct(window, displacement, regularization, *, tolerance=1e-11, maxiter=None):
    """Minimize ||H f - y||_2^2 + regularization^2 ||f||_2^2.

    regularization is strictly positive, in m/N for SI data; tolerance is
    LSQR atol=btol. Raises RuntimeError on iteration/condition-limit stops.
    Returns (force, diagnostics). Noise covariance is assumed proportional
    to identity. No automatic hyperparameter selection is performed.
    """
    _positive(regularization, "regularization")
    _positive(tolerance, "tolerance")
    if tolerance >= 1:
        raise ValueError("tolerance must be smaller than one")
    if maxiter is None:
        maxiter = 10 * window.shape[1]
    _integer(maxiter, "maxiter", 1)
    y = _vector(displacement, window.shape[0])
    out = lsqr(
        window,
        y,
        damp=regularization,
        atol=tolerance,
        btol=tolerance,
        iter_lim=maxiter,
        conlim=1e14,
    )
    if out[1] not in (0, 1, 2, 4, 5):
        raise RuntimeError(f"LSQR did not converge: stop={out[1]}, iterations={out[2]}")
    force = out[0]
    residual = window @ force - y
    gradient = window.rmatvec(residual) + regularization**2 * force
    return force, {
        "stop": int(out[1]),
        "iterations": int(out[2]),
        "residual_l2_m": float(np.linalg.norm(residual)),
        "normal_residual_l2": float(np.linalg.norm(gradient)),
    }
