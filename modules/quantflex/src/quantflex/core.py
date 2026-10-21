"""Small explicit numerical kernels; no hidden sampling or fitted noise model."""

from dataclasses import dataclass

import numpy as np


def _positive(value, name, *, zero=False):
    value = float(value)
    if not np.isfinite(value) or (value < 0 if zero else value <= 0):
        raise ValueError(f"{name} must be finite and {'nonnegative' if zero else 'positive'}")
    return value


def _array(value, name):
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    a = np.asarray(value, dtype=float)
    if not a.size or not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be nonempty and finite")
    return a


@dataclass(frozen=True)
class Quantized:
    """Reconstructed measurements and mask of actually clipped integer codes."""

    values: np.ndarray
    saturated: np.ndarray


def quantize(signal, step, *, dither=None, max_code=None):
    """Midtread rounding, ties toward +infinity; optional subtractive dither.

    Computes step*floor((signal+dither)/step+1/2)-dither. The caller supplies
    the dither; the uniform-error theorem requires independent Uniform(-step/2,
    step/2) samples, independent of the signal, and no code clipping. This
    function cannot certify independence. max_code clips codes to [-M,M],
    a symmetric pedagogical converter, not a particular signed-bit ADC.
    """
    x = _array(signal, "signal")
    step = _positive(step, "step")
    d = np.zeros_like(x) if dither is None else _array(dither, "dither")
    if d.shape != x.shape:
        raise ValueError("dither and signal must have identical shapes")
    if np.any(np.abs(d) > step / 2):
        raise ValueError("dither must lie within one quantization interval")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        scaled = (x + d) / step
    if not np.all(np.isfinite(scaled)) or np.any(np.abs(scaled) > 2**40):
        raise ValueError("scaled input exceeds supported floating-point range (2**40)")
    codes = np.floor(scaled + 0.5)
    clipped = np.zeros(x.shape, dtype=bool)
    if max_code is not None:
        if isinstance(max_code, (bool, np.bool_)) or not isinstance(max_code, (int, np.integer)):
            raise ValueError("max_code must be a positive integer")
        if max_code < 1 or max_code > 2**40:
            raise ValueError("max_code must lie in [1, 2**40]")
        clipped = np.abs(codes) > max_code
        codes = np.clip(codes, -max_code, max_code)
    with np.errstate(over="ignore", invalid="ignore"):
        values = step * codes - d
    if not np.all(np.isfinite(values)):
        raise ValueError("reconstructed output overflows")
    return Quantized(values, clipped)


def steady_coefficients(mass, damping, stiffness, omega, force):
    """[DC, cosine, sine] for m*u''+c*u'+k*u=force*cos(omega*t).

    The steady particular solution is used; no initial transient is included.
    A positive damping coefficient is required to exclude undamped resonance.
    """
    m = _positive(mass, "mass")
    c = _positive(damping, "damping")
    k = _positive(stiffness, "stiffness")
    w = _positive(omega, "omega")
    f = float(force)
    if not np.isfinite(f):
        raise ValueError("force must be finite")
    with np.errstate(over="ignore", invalid="ignore"):
        z = complex(k - m * w * w, c * w)
    if not np.all(np.isfinite([z.real, z.imag])) or abs(z) == 0:
        raise ValueError("oscillator parameters exceed numerical range")
    with np.errstate(over="ignore", invalid="ignore"):
        h = f / z
    beta = np.array([0.0, h.real, -h.imag])
    if not np.all(np.isfinite([z.real, z.imag, *beta])) or abs(z) == 0:
        raise ValueError("oscillator parameters exceed numerical range")
    return beta


class HarmonicModel:
    """Fixed-frequency least squares via reduced QR, with a fitted DC offset.

    times is a 1-D sequence of seconds; omega is radians/second. The design
    [1,cos(omega*t),sin(omega*t)] must have condition number below 1e8.
    No frequency estimation, anti-alias filtering or confidence intervals.
    """

    def __init__(self, times, omega):
        t = _array(times, "times")
        w = _positive(omega, "omega")
        if t.ndim != 1 or t.size < 3:
            raise ValueError("times must be 1-D with at least three entries")
        with np.errstate(over="ignore"):
            phase = w * t
        if not np.all(np.isfinite(phase)) or np.max(np.abs(phase)) > 2**30:
            raise ValueError("phase exceeds supported trigonometric range")
        design = np.column_stack((np.ones_like(t), np.cos(phase), np.sin(phase)))
        if np.linalg.cond(design) >= 1e8:
            raise ValueError("harmonic design is rank deficient or ill-conditioned")
        q, r = np.linalg.qr(design, mode="reduced")
        self.design = design
        self.operator = np.linalg.solve(r, q.T)
        self.design.flags.writeable = False
        self.operator.flags.writeable = False

    def fit(self, observations):
        """Fit one vector or batches (..., n); return (..., 3) coefficients."""
        y = _array(observations, "observations")
        if y.ndim < 1 or y.shape[-1] != self.design.shape[0]:
            raise ValueError("last observation dimension must match the sampling times")
        beta = y @ self.operator.T
        if not np.all(np.isfinite(beta)):
            raise ValueError("fitted coefficients overflow")
        return beta

    def covariance(self, noise_variance):
        """Coefficient covariance for iid zero-mean errors of known variance.

        Use step**2/12 only for the declared unclipped iid subtractive-dither
        experiment. Ordinary quantization generally does not satisfy this model.
        """
        variance = _positive(noise_variance, "noise_variance", zero=True)
        return variance * (self.operator @ self.operator.T)


def elastic_energy(coefficients, stiffness, *, covariance=None):
    """Cycle-average oscillatory spring energy k*(a**2+b**2)/4 in joules.

    DC is excluded. Optionally subtract k*trace(C[1:,1:])/4, unbiased when
    coefficients are unbiased with covariance C. Negative values are retained;
    clipping to zero would reintroduce positive bias.
    """
    b = _array(coefficients, "coefficients")
    k = _positive(stiffness, "stiffness")
    if b.ndim < 1 or b.shape[-1] != 3:
        raise ValueError("coefficients must end in [DC, cosine, sine]")
    correction = 0.0
    if covariance is not None:
        cov = _array(covariance, "covariance")
        if cov.shape != (3, 3) or not np.allclose(cov, cov.T, rtol=1e-12, atol=0):
            raise ValueError("covariance must be symmetric 3x3")
        eigenvalues = np.linalg.eigvalsh(cov)
        if eigenvalues[0] < -1e-12 * max(float(np.max(np.abs(cov))), 1e-300):
            raise ValueError("covariance must be positive semidefinite")
        correction = np.trace(cov[1:, 1:])
    with np.errstate(over="ignore", invalid="ignore"):
        energy = k / 4 * (np.sum(b[..., 1:] ** 2, axis=-1) - correction)
    if not np.all(np.isfinite(energy)):
        raise ValueError("energy overflows")
    return energy
