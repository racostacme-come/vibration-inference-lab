import numpy as np
import pytest

from quantflex import HarmonicModel, elastic_energy, quantize, steady_coefficients


def test_ties_and_saturation():
    x = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5])
    np.testing.assert_array_equal(quantize(x, 1).values, [-2, -1, 0, 1, 2, 3])
    result = quantize(x, 1, max_code=1)
    np.testing.assert_array_equal(result.values, [-1, -1, 0, 1, 1, 1])
    np.testing.assert_array_equal(result.saturated, [True, False, False, False, True, True])


@pytest.mark.parametrize("x", [-3.21, -0.5, -0.123, 0, 0.2, 0.5, 8.43])
def test_dither_integral(x):
    # Deterministic midpoint integration, independent of the study PRNG.
    n = 100_000
    d = (np.arange(n) + 0.5) / n - 0.5
    error = quantize(np.full(n, x), 1, dither=d).values - x
    assert abs(error.mean()) < 1 / n
    assert abs(np.mean(error**2) - 1 / 12) < 1e-9
    assert np.max(np.abs(error)) <= 0.5 + 1e-14


def test_steady_equilibrium_and_energy():
    m, c, k, w, f = 1.3, 0.7, 110, 7.1, 0.2
    beta = steady_coefficients(m, c, k, w, f)
    # Independent real 2x2 dynamic equilibrium, not complex division.
    z = np.array([[k - m * w**2, c * w], [-c * w, k - m * w**2]])
    np.testing.assert_allclose(z @ beta[1:], [f, 0], atol=1e-16)
    t = np.arange(2000) * (2 * np.pi / w / 2000)
    u = beta[1] * np.cos(w * t) + beta[2] * np.sin(w * t)
    np.testing.assert_allclose(elastic_energy(beta, k), (k * u**2 / 2).mean())


def test_qr_against_lstsq_nonuniform():
    rng = np.random.default_rng(9)
    model = HarmonicModel(np.sort(rng.uniform(0, 3, 81)), 5.2)
    y = rng.normal(size=(4, 81))
    oracle = np.linalg.lstsq(model.design, y.T, rcond=None)[0].T
    np.testing.assert_allclose(model.fit(y), oracle, atol=2e-15)
    expected = np.linalg.inv(model.design.T @ model.design) * 0.2
    np.testing.assert_allclose(model.covariance(0.2), expected, atol=2e-17)


def test_exact_recovery_and_integer_cycle_covariance():
    n = 200
    model = HarmonicModel(np.arange(n) / 40, 2 * np.pi)
    beta = np.array([0.1, 0.2, -0.3])
    np.testing.assert_allclose(model.fit(model.design @ beta), beta, atol=1e-15)
    np.testing.assert_allclose(model.covariance(1), np.diag([1, 2, 2]) / n, atol=1e-17)


def test_energy_correction_and_negative_retention():
    covariance = np.diag([0.5, 0.2, 0.3])
    assert elastic_energy([0, 0, 0], 4, covariance=covariance) == -0.5
    assert elastic_energy([100, 3, 4], 4, covariance=covariance) == 24.5


def test_low_amplitude_rounding_does_not_average_away():
    for n in [40, 80, 200, 800]:
        model = HarmonicModel(np.arange(n) / 40, 2 * np.pi)
        signal = model.design @ [0, 0.2, 0.1]
        np.testing.assert_array_equal(model.fit(quantize(signal, 1).values), 0)


def test_finite_sample_dither_covariance():
    rng = np.random.default_rng(739)
    model = HarmonicModel(np.arange(80) / 40, 2 * np.pi)
    beta = np.array([0.1, 0.2, 0.1])
    x = np.broadcast_to(model.design @ beta, (20000, 80))
    result = quantize(x, 1, dither=rng.uniform(-0.5, 0.5, x.shape))
    estimates = model.fit(result.values)
    expected = model.covariance(1 / 12)
    np.testing.assert_allclose(np.cov(estimates.T), expected, rtol=0.04, atol=4e-5)
    assert np.max(np.abs(estimates.mean(axis=0) - beta) / np.sqrt(np.diag(expected) / len(x))) < 5
    energies = elastic_energy(estimates, 4, covariance=expected)
    assert abs(energies.mean() - elastic_energy(beta, 4)) < 5 * energies.std() / np.sqrt(len(x))


@pytest.mark.parametrize("step", [0, -1, np.nan, np.inf])
def test_bad_step(step):
    with pytest.raises(ValueError):
        quantize([1], step)


@pytest.mark.parametrize("value", [[], [np.nan], [np.inf], [1j], [2**42]])
def test_bad_signal(value):
    with pytest.raises(ValueError):
        quantize(value, 1)


@pytest.mark.parametrize("code", [0, -1, 1.5, True, 2**41])
def test_bad_code(code):
    with pytest.raises(ValueError):
        quantize([1], 1, max_code=code)


def test_bad_dither():
    for d in [[0], [0.6, 0], [np.inf, 0]]:
        with pytest.raises(ValueError):
            quantize([0, 0], 1, dither=d)


@pytest.mark.parametrize("times", [[0, 0, 0], [0, 1], [[0, 1, 2]], [0, np.nan, 2]])
def test_bad_design(times):
    with pytest.raises(ValueError):
        HarmonicModel(times, 1)


def test_bad_fit_and_covariance():
    model = HarmonicModel([0, 1, 2, 3], 1)
    with pytest.raises(ValueError):
        model.fit([1, 2, 3])
    with pytest.raises(ValueError):
        model.covariance(-1)
    with pytest.raises(ValueError):
        elastic_energy([1, 2], 1)
    for cov in [np.eye(2), np.diag([1, -1, 1]), np.arange(9).reshape(3, 3)]:
        with pytest.raises(ValueError):
            elastic_energy([1, 2, 3], 1, covariance=cov)


@pytest.mark.parametrize(
    "params",
    [(0, 1, 1, 1, 1), (1, 0, 1, 1, 1), (1, 1, -1, 1, 1), (1, 1, 1, 0, 1), (1, 1, 1, 1, np.nan)],
)
def test_bad_mechanics(params):
    with pytest.raises(ValueError):
        steady_coefficients(*params)


def test_clipping_breaks_unbiasedness():
    d = (np.arange(10000) + 0.5) / 10000 - 0.5
    result = quantize(np.full_like(d, 2.0), 1, dither=d, max_code=1)
    assert result.saturated.all()
    assert abs(result.values.mean() - 1) < 1e-12


def test_extreme_mechanics_range_is_rejected():
    for params in [(1e308, 1, 1, 1e308, 1), (1e-300, 1e-300, 1e-300, 1, 1e308)]:
        with pytest.raises(ValueError):
            steady_coefficients(*params)
