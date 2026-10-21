import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.integrate import quad_vec
from scipy.linalg import expm

from gapsmooth.inference import infer, simulate
from gapsmooth.model import Oscillator
from gapsmooth.reference import batch_condition


@pytest.mark.parametrize("dt", [0, 1e-7, 0.07, 1, 50])
def test_transition_against_independent_quadrature_and_exponential(dt):
    model = Oscillator()
    f, q = model.transition(dt)
    a = model.drift
    ref, _ = quad_vec(
        lambda s: expm(a * s) @ np.diag([0, model.q]) @ expm(a.T * s),
        0,
        dt,
        epsabs=1e-14,
        epsrel=1e-13,
    )
    assert_allclose(f, expm(a * dt), atol=2e-13, rtol=2e-12)
    assert_allclose(q, ref, atol=2e-13, rtol=2e-12)
    assert np.linalg.eigvalsh(q).min() >= -1e-14


def test_semigroup_and_stationary_covariance():
    model = Oscillator()
    f, q = model.transition(0.4)
    g, r = model.transition(0.7)
    h, s = model.transition(1.1)
    assert_allclose(g @ f, h, atol=1e-13)
    assert_allclose(g @ q @ g.T + r, s, atol=1e-13)
    p = model.stationary_covariance
    assert_allclose(f @ p @ f.T + q, p, atol=1e-13)


@pytest.mark.parametrize("pattern", ["full", "gap", "none", "single"])
def test_batch_posterior_likelihood_and_prefix_filter(pattern):
    model = Oscillator()
    t = np.array([0, 0.03, 0.16, 0.21, 0.4, 0.71])
    m, p, r = np.array([0.1, -0.2]), np.diag([0.02, 0.3]), 0.003
    _, y = simulate(model, t, r, m, p, seed=19)
    if pattern == "gap":
        y[2:5] = np.nan
    elif pattern == "none":
        y[:] = np.nan
    elif pattern == "single":
        y[:-1] = np.nan
    estimate = infer(model, t, y, r, m, p)
    bm, bp, bnll = batch_condition(model, t, y, r, m, p)
    assert_allclose(estimate.smoothed_mean, bm, atol=2e-13)
    for k in range(len(t)):
        assert_allclose(
            estimate.smoothed_covariance[k],
            bp[2 * k : 2 * k + 2, 2 * k : 2 * k + 2],
            atol=2e-13,
        )
        prefix, pc, _ = batch_condition(model, t[: k + 1], y[: k + 1], r, m, p)
        assert_allclose(estimate.filtered_mean[k], prefix[-1], atol=2e-13)
        assert_allclose(estimate.filtered_covariance[k], pc[-2:, -2:], atol=2e-13)
        assert (
            np.linalg.eigvalsh(
                estimate.filtered_covariance[k] - estimate.smoothed_covariance[k]
            ).min()
            > -1e-13
        )
    assert_allclose(estimate.nll, bnll, atol=2e-12)
    assert np.array_equal(np.isnan(estimate.innovations), np.isnan(y))


def test_zero_process_noise_and_one_observation():
    model = Oscillator(q=0)
    assert_allclose(model.transition(1)[1], 0)
    result = infer(model, [10], [1], 0.1, [0, 0], np.eye(2))
    assert_allclose(result.filtered_mean[0], [1 / 1.1, 0])
    assert_allclose(result.filtered_covariance[0], np.diag([0.1 / 1.1, 1]))


@pytest.mark.parametrize("kwargs", [{"omega": 0}, {"zeta": -1}, {"q": -1}, {"q": np.nan}])
def test_invalid_model(kwargs):
    with pytest.raises(ValueError):
        Oscillator(**kwargs)


@pytest.mark.parametrize(
    "times,y,r,p",
    [
        ([0, 0], [1, 2], 1, np.eye(2)),
        ([], [], 1, np.eye(2)),
        ([0], [np.inf], 1, np.eye(2)),
        ([0], [1, 2], 1, np.eye(2)),
        ([0], [1], 0, np.eye(2)),
        ([0], [1], 1, np.zeros((2, 2))),
        ([0], [1], 1, [[1, 0.1], [0, 1]]),
    ],
)
def test_invalid_inference(times, y, r, p):
    with pytest.raises(ValueError):
        infer(Oscillator(), times, y, r, [0, 0], p)


@pytest.mark.parametrize("dt", [-1, np.nan, np.inf])
def test_invalid_interval(dt):
    with pytest.raises(ValueError):
        Oscillator().transition(dt)
