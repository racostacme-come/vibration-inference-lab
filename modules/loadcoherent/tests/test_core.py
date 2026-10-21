"""Independent constrained solves, analytic examples, and input contracts."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from loadcoherent import Reconciler, estimate_covariance, prediction_mse


@pytest.fixture
def problem():
    s = np.vstack(([1, 1, 1], [1, 1, 0], np.eye(3)))
    rng = np.random.default_rng(42)
    a = rng.normal(size=(5, 5))
    w = a @ a.T + np.eye(5)
    return s, w


def test_scalar_analytic_inverse_variance_weighting():
    model = Reconciler([[1], [1]], np.diag([1, 4]))
    assert_allclose(model.bottom_map, [[0.8, 0.2]], atol=1e-14)
    assert_allclose(model.reconcile([10, 20]), [12, 12])
    assert_allclose(prediction_mse(model.projection, np.diag([1, 4])), [0.8, 0.8])


def test_against_independent_kkt(problem):
    s, w = problem
    # Independent saddle-point formulation: W lambda + S b = forecast; S^T lambda=0.
    kkt = np.block([[w, s], [s.T, np.zeros((3, 3))]])
    forecast = np.array([4, 9, -2, 1, 8])
    solution = np.linalg.solve(kkt, np.r_[forecast, np.zeros(3)])
    assert_allclose(Reconciler(s, w).reconcile(forecast), s @ solution[5:], atol=1e-13)


def test_projection_identities(problem):
    s, w = problem
    model = Reconciler(s, w)
    p = model.projection
    assert_allclose(p @ p, p, atol=1e-14)
    assert_allclose(p @ s, s, atol=1e-14)
    assert_allclose(model.bottom_map @ s, np.eye(3), atol=1e-14)
    assert_allclose(np.linalg.solve(w, p), np.linalg.solve(w, p).T, atol=1e-14)
    forecast = np.arange(5)
    residual = forecast - model.reconcile(forecast)
    assert_allclose(s.T @ np.linalg.solve(w, residual), 0, atol=1e-14)


def test_batch_and_scale_invariance(problem):
    s, w = problem
    batch = np.arange(30).reshape(2, 3, 5)
    model = Reconciler(s, w)
    result = model.reconcile(batch)
    assert result.shape == batch.shape
    assert_allclose(result, Reconciler(s, 1e-8 * w).reconcile(batch), atol=1e-12)
    assert_allclose(result[1, 2], model.reconcile(batch[1, 2]))


def test_mutation_isolation(problem):
    s, w = problem
    model = Reconciler(s, w)
    expected = model.projection
    s[:] = 0
    w[:] = 0
    p = model.projection
    p[:] = 0
    assert_allclose(model.projection, expected)


def test_oracle_variance_dominance(problem):
    s, w = problem
    oracle = Reconciler(s, w).projection
    ols = Reconciler(s, np.eye(5)).projection
    difference = ols @ w @ ols.T - oracle @ w @ oracle.T
    assert np.linalg.eigvalsh(difference)[0] > -1e-12
    assert np.all(prediction_mse(oracle, w) <= np.diag(w) + 1e-12)


def test_coherence_does_not_guarantee_accuracy_or_positivity():
    model = Reconciler([[1, 1], [1, 0], [0, 1]], np.eye(3))
    assert model.reconcile([1, 10, 1])[2] < 0
    wrong = Reconciler([[1], [1]], np.diag([1e-3, 1])).projection
    assert prediction_mse(wrong, np.diag([100, 1])).sum() > 101


def test_covariance_centering_endpoints():
    errors = np.array([[1, 4], [3, 1], [7, 2], [5, 9]])
    expected = np.cov(errors, rowvar=False)
    assert_allclose(estimate_covariance(errors, 0), expected)
    assert_allclose(estimate_covariance(errors, 1), np.diag(np.diag(expected)))
    assert_allclose(estimate_covariance(errors + 200, 0.5), estimate_covariance(errors))


def test_singular_sample_is_explicit():
    sample = estimate_covariance([[1, 2], [3, 4]], 0)
    with pytest.raises(ValueError, match="positive definite"):
        Reconciler([[1], [1]], sample)
    Reconciler([[1], [1]], estimate_covariance([[1, 2], [3, 4]], 0.5))


def test_bias_and_semidefinite_risk():
    assert_allclose(prediction_mse(np.eye(2), np.zeros((2, 2)), [2, -3]), [4, 9])


@pytest.mark.parametrize(
    "s,w",
    [
        ([[1, 1]], [[1]]),
        ([[1], [1]], np.eye(3)),
        ([[1, 1], [1, 1]], np.eye(2)),
        ([[1], [1]], [[1, 2], [0, 1]]),
        ([[1], [1]], [[1, 2], [2, 1]]),
        ([[1]], [[0]]),
        ([[np.nan]], [[1]]),
        ([[1]], [[np.inf]]),
        ([[1j]], [[1]]),
        ([[1], [1]], np.diag([1, 1e-14])),
        (np.empty((0, 0)), np.empty((0, 0))),
    ],
)
def test_invalid_model(s, w):
    with pytest.raises(ValueError):
        Reconciler(s, w)


@pytest.mark.parametrize("values", [1, [1], [1, np.nan], [[1, 2, 3]], [1j, 2]])
def test_invalid_forecasts(values):
    with pytest.raises(ValueError):
        Reconciler([[1], [1]], np.eye(2)).reconcile(values)


@pytest.mark.parametrize(
    "errors,alpha",
    [
        ([[1, 2]], 0.5),
        ([[1, 2], [1, 3]], 0.5),
        ([[1, 2], [3, np.nan]], 0.5),
        ([[1, 2], [3, 4]], -1),
        ([[1, 2], [3, 4]], 2),
        ([[1, 2], [3, 4]], np.nan),
    ],
)
def test_invalid_estimation(errors, alpha):
    with pytest.raises(ValueError):
        estimate_covariance(errors, alpha)


@pytest.mark.parametrize(
    "p,v,bias",
    [
        (np.ones((2, 1)), np.eye(2), None),
        (np.eye(2), np.eye(3), None),
        (np.eye(2), -np.eye(2), None),
        (np.eye(2), np.eye(2), [1]),
    ],
)
def test_invalid_risk(p, v, bias):
    with pytest.raises(ValueError):
        prediction_mse(p, v, bias)
