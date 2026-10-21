import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.signal import firwin, freqz, lfilter

from aliasguard import Decimator, alias_frequency, frequency_response, lowpass


@pytest.mark.parametrize("factor", [1, 2, 8, 1000])
@pytest.mark.parametrize("taps", [1, 8, 65])
def test_random_stream_matches_independent_convolution(factor, taps):
    rng = np.random.default_rng(551)
    h, x = rng.normal(size=taps), rng.normal(size=527)
    expected = lfilter(h, [1.0], x)[::factor]
    fir = Decimator(h, factor)
    chunks = [x[:0], x[:1], x[1:49], x[49:50], x[50:100], x[100:]]
    actual = np.concatenate([fir.process(chunk) for chunk in chunks])
    assert_allclose(actual, expected, atol=2e-14, rtol=2e-14)
    fir.reset()
    assert_array_equal(actual, fir.process(x))


@pytest.mark.parametrize("taps", [3, 65, 129, 257, 513])
def test_designer_and_response_independent_reference(taps):
    h = lowpass(taps, 96, 2048)
    assert_allclose(h, firwin(taps, 96, window="blackman", fs=2048), atol=1e-16)
    assert_allclose(h, h[::-1], atol=1e-16)
    assert_allclose(h.sum(), 1, atol=4e-16)
    f = np.array([0, 23, 71, 96, 128, 185, 1024])
    _, expected = freqz(h, worN=f, fs=2048)
    assert_allclose(frequency_response(h, f, 2048), expected, atol=5e-15)


def test_sampled_collision_includes_phase_reversal():
    t = np.arange(1000) / 256
    assert_allclose(
        np.cos(2 * np.pi * 185 * t + 0.4), np.cos(2 * np.pi * 71 * t - 0.4), atol=8e-13
    )
    assert_array_equal(
        alias_frequency([23, 71, 185, -185, 256, 128], 256), [23, 71, 71, 71, 0, 128]
    )


def test_causal_response_and_delay():
    h = lowpass(129, 96, 2048)
    n = np.arange(4096)
    f = 71.3
    y = Decimator(h, 8).process(np.cos(2 * np.pi * f * n / 2048 + 0.2))
    response = frequency_response(h, f, 2048)[0]
    exact = np.real(response * np.exp(1j * (2 * np.pi * f * n[::8] / 2048 + 0.2)))
    assert_allclose(y[16:], exact[16:], atol=4e-14)


@pytest.mark.parametrize("invalid", [[1, np.nan], [np.inf], [[1, 2]], [1 + 2j]])
def test_invalid_chunk_is_transactional(invalid):
    fir = Decimator([1, 2, 3], 2)
    fir.process([1, 2])
    with pytest.raises(ValueError):
        fir.process(invalid)
    assert_array_equal(fir.process([3, 4, 5]), [10, 22])


@pytest.mark.parametrize("factor", [0, -1, 1.5, True, 2**100])
def test_invalid_factor(factor):
    with pytest.raises(ValueError):
        Decimator([1], factor)


@pytest.mark.parametrize("h", [[], [np.nan], [[1]], [1j]])
def test_invalid_coefficients(h):
    with pytest.raises(ValueError):
        Decimator(h, 1)


@pytest.mark.parametrize(
    "args",
    [(2, 10, 100), (0, 10, 100), (5, 50, 100), (5, 0, 100), (5, 10, np.inf), (3.5, 10, 100)],
)
def test_invalid_design(args):
    with pytest.raises(ValueError):
        lowpass(*args)


def test_startup_tail_and_empty_conventions():
    fir = Decimator([0.25, 0.5, 0.25], 2)
    assert_array_equal(fir.process([1, 0, 0, 0, 0]), [0.25, 0.25, 0])
    assert fir.process([]).shape == (0,)


def test_dc_after_warmup():
    h = lowpass(65, 96, 2048)
    y = Decimator(h, 8).process(np.ones(500))
    assert_allclose(y[8:], 1, atol=5e-16)
