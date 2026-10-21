"""Independent distance-matrix oracle and native boundary tests."""

import numpy as np
import pytest
from scipy.spatial.distance import cdist, pdist

from jointguard import energy_score


@pytest.mark.parametrize("unbiased", [True, False])
@pytest.mark.parametrize("d", [1, 2, 7])
def test_distance_matrix_oracle(unbiased, d):
    rng = np.random.default_rng(73)
    x, y = rng.normal(size=(19, d)), rng.normal(size=(11, d))
    expected = cdist(y, x).mean(axis=1) - pdist(x).sum() / (19 * (18 if unbiased else 19))
    np.testing.assert_allclose(energy_score(x, y, unbiased), expected, rtol=2e-14, atol=2e-14)


def test_ensemble_correction():
    x = [[0.0], [2.0]]
    np.testing.assert_allclose(energy_score(x, [[1.0]], True), [0])
    np.testing.assert_allclose(energy_score(x, [[1.0]], False), [0.5])
    expected = 1 - (2 + 2 * np.sqrt(2)) / 6
    assert energy_score([[-1, 0], [1, 0], [0, 1]], [[0, 0]])[0] == pytest.approx(expected)


def test_permutation_translation_rotation_and_scale():
    rng = np.random.default_rng(9)
    x, y = rng.normal(size=(13, 2)), rng.normal(size=(17, 2))
    q = np.array([[0.6, -0.8], [0.8, 0.6]])
    baseline = energy_score(x, y)
    np.testing.assert_allclose(energy_score(3 * x[::-1] @ q + 7, 3 * y @ q + 7), 3 * baseline)


def test_strided_and_single_empirical_member():
    np.testing.assert_allclose(
        energy_score(np.arange(12).reshape(3, 4)[:, ::2], [[1, 2]], False),
        energy_score([[0, 2], [4, 6], [8, 10]], [[1, 2]], False),
    )
    np.testing.assert_allclose(energy_score([[0, 0]], [[3, 4]], False), [5])


@pytest.mark.parametrize(
    "x,y",
    [
        ([], [[0]]),
        ([[1]], [[0]]),
        ([[1], [2]], [[1, 2]]),
        ([[1], [np.nan]], [[0]]),
        ([[1], [2]], [[np.inf]]),
        ([[], []], [[]]),
        ([[1], [2]], []),
    ],
)
def test_bad_inputs(x, y):
    with pytest.raises(ValueError):
        energy_score(x, y)


def test_overflow_rejected():
    with pytest.raises(OverflowError):
        energy_score([[-1e308], [-1e308]], [[1e308]])
