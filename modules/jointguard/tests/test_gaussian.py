"""Closed-form identities, independent density checks, and mechanical invariants."""

import numpy as np
import pytest
from scipy.special import gamma
from scipy.stats import multivariate_normal, norm

from jointguard import Gaussian, elastic_response, expected_norm, expected_scores
from jointguard.gaussian import absolute_normal_mean


@pytest.mark.parametrize("d", [1, 2, 5])
def test_chi_norm(d):
    result, estimate = expected_norm(Gaussian(np.zeros(d), 4 * np.eye(d)))
    exact = 2 * np.sqrt(2) * gamma((d + 1) / 2) / gamma(d / 2)
    assert result == pytest.approx(exact, rel=1e-11)
    assert estimate < 1e-8


def test_folded_normal_and_noncentral_quadrature():
    m, v = 1.3, 0.7
    exact = norm.expect(lambda x: abs(x), loc=m, scale=np.sqrt(v))
    assert absolute_normal_mean(m, v) == pytest.approx(exact, abs=1e-8)
    assert expected_norm(Gaussian([m], [[v]]))[0] == pytest.approx(exact, abs=1e-8)
    assert absolute_normal_mean(-2, 0) == 2


def test_log_density_oracle():
    g = Gaussian([1, -0.5], [[2, 0.3], [0.3, 1]])
    y = np.random.default_rng(8).normal(size=(20, 2))
    np.testing.assert_allclose(g.log_score(y), -multivariate_normal.logpdf(y, g.mean, g.covariance))


def test_calibrated_population_scores():
    g = Gaussian([0, 0], np.eye(2))
    s = expected_scores(g, g)
    assert s["marginal_crps"] == pytest.approx(1 / np.sqrt(np.pi))
    assert s["energy"] == pytest.approx(np.sqrt(np.pi) / 2)
    assert s["variogram"] == pytest.approx(2 * (1 - 2 / np.pi))
    assert s["log_score"] == pytest.approx(np.log(2 * np.pi) + 1)


def test_correlation_blind_marginals_and_shift_blind_variograms():
    truth = Gaussian([0, 0], [[1, 0.9], [0.9, 1]])
    independent = Gaussian([0, 0], np.eye(2))
    shifted = Gaussian([1, 1], truth.covariance)
    y = np.random.default_rng(91).normal(size=(60, 2))
    np.testing.assert_array_equal(truth.marginal_crps(y), independent.marginal_crps(y))
    np.testing.assert_array_equal(truth.variogram_score(y), shifted.variogram_score(y))
    best = expected_scores(truth, truth)
    for g in [independent, shifted]:
        score = expected_scores(g, truth)
        assert score["log_score"] > best["log_score"]
        assert score["energy"] > best["energy"]
    assert expected_scores(independent, truth)["variogram"] > best["variogram"]


def test_exceedance_closed_form_and_shift_invariance():
    g = Gaussian([0, 0], [[1, 0.9], [0.9, 1]])
    p = g.exceedance([-1, 1], 1)
    assert p == pytest.approx(2 * norm.sf(1 / np.sqrt(0.2)))
    assert Gaussian([2, 2], g.covariance).exceedance([-1, 1], 1) == p
    assert g.exceedance([-1, 1], 0) == pytest.approx(1)


def test_force_response_equilibrium():
    k = 1e6 * np.array([[2, -1], [-1, 2]])
    f = Gaussian([1200, -300], [[9e5, 2e5], [2e5, 7e5]])
    u = elastic_response(k, f)
    np.testing.assert_allclose(k @ u.mean, f.mean, rtol=1e-14)
    np.testing.assert_allclose(k @ u.covariance @ k.T, f.covariance, rtol=1e-14)
    assert np.linalg.eigvalsh(u.covariance).min() > 0


@pytest.mark.parametrize(
    "mean,cov",
    [
        ([], []),
        ([0], [[0]]),
        ([0, 0], [[1, 2], [2, 1]]),
        ([0, 0], [[1, 0.1], [0.2, 1]]),
        ([np.inf], [[1]]),
        ([0], [[np.nan]]),
        ([0, 0], [[1]]),
    ],
)
def test_gaussian_rejects_invalid(mean, cov):
    with pytest.raises(ValueError):
        Gaussian(mean, cov)


def test_invalid_diagnostics_and_copy_ownership():
    mean, cov = np.zeros(2), np.eye(2)
    g = Gaussian(mean, cov)
    mean[:] = 8
    cov[:] = 0
    np.testing.assert_array_equal(g.mean, [0, 0])
    for direction, threshold in [([0, 0], 1), ([1], 1), ([1, 0], -1), ([1, 0], np.nan)]:
        with pytest.raises(ValueError):
            g.exceedance(direction, threshold)
    with pytest.raises(ValueError):
        g.log_score([[np.nan, 0]])
    with pytest.raises(ValueError):
        Gaussian([0], [[1]]).variogram_score([[1]])
    with pytest.raises(ValueError):
        elastic_response([[0, 0], [0, 1]], g)
