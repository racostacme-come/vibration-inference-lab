"""Filter construction and a validated NumPy-facing streaming interface."""

import operator

import numpy as np

from . import _core


def _positive(value, name):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer, not bool")
    try:
        return operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _vector(value, name, *, empty=False):
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    result = np.asarray(value, dtype=float)
    if result.ndim != 1 or (not empty and result.size == 0) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite one-dimensional vector")
    return result


def lowpass(taps, cutoff, sample_rate):
    """Odd symmetric Blackman-windowed sinc, normalized to unit DC gain.

    Frequencies are in Hz. Cutoff is the ideal sinc cutoff (approximately
    half amplitude), not a guaranteed passband edge. No automatic filter sizing.
    """
    taps = _integer(taps, "taps")
    sample_rate = _positive(sample_rate, "sample_rate")
    cutoff = _positive(cutoff, "cutoff")
    if taps < 3 or taps % 2 != 1 or cutoff >= sample_rate / 2:
        raise ValueError("odd taps >= 3 and cutoff < sample_rate/2 required")
    offset = np.arange(taps) - (taps - 1) / 2
    h = 2 * cutoff / sample_rate * np.sinc(2 * cutoff / sample_rate * offset)
    h *= np.blackman(taps)
    return h / h.sum()


def alias_frequency(frequency, sample_rate):
    """Nonnegative alias in [0, sample_rate/2]; phase is not returned."""
    sample_rate = _positive(sample_rate, "sample_rate")
    if np.iscomplexobj(frequency):
        raise ValueError("frequency must be real")
    f = np.asarray(frequency, dtype=float)
    if not np.isfinite(f).all():
        raise ValueError("frequency must be finite")
    return abs((f + sample_rate / 2) % sample_rate - sample_rate / 2)


def frequency_response(coefficients, frequency, sample_rate):
    """Causal complex response sum h[k] exp(-2 pi i f k / sample_rate)."""
    h = _vector(coefficients, "coefficients")
    sample_rate = _positive(sample_rate, "sample_rate")
    f = _vector(np.atleast_1d(frequency), "frequency")
    return np.exp(-2j * np.pi * np.outer(f / sample_rate, np.arange(h.size))) @ h


class Decimator:
    """Causal FIR, emitting input indices 0, R, 2R, ... across chunk boundaries.

    Initial history is zero. Empty input is a no-op. Invalid chunks are rejected
    before state changes. No implicit tail flush or startup discard occurs.
    For symmetric odd L taps, the group delay is (L-1)/2 input samples.
    Instances hold mutable state and must not be shared concurrently.
    """

    def __init__(self, coefficients, factor):
        h = _vector(coefficients, "coefficients")
        factor = _integer(factor, "factor")
        if factor < 1 or factor > np.iinfo(np.uintp).max:
            raise ValueError("factor must fit a positive size_t")
        self._implementation = _core.Decimator(h.tolist(), factor)

    def process(self, samples):
        """Return a float64 array; accepted finite real 1-D inputs are copied."""
        samples = _vector(samples, "samples", empty=True)
        return np.asarray(self._implementation.process(samples.tolist()), dtype=float)

    def reset(self):
        """Clear history and restart output phase at input index zero."""
        self._implementation.reset()
