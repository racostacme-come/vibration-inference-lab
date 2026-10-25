import numpy as np
import pytest
from gapsmooth.model import Oscillator
from quantflex.core import quantize
from verification import transition, process_covariance, experiment


@pytest.fixture(scope='module')
def evidence():
    return experiment()


def test_closed_form_and_covariance_quadrature(evidence):
    assert evidence['transition_max_error'] < 2e-12
    assert evidence['response_max_error'] < 2e-13
    assert evidence['process_covariance_max_error'] < 2e-13


@pytest.mark.parametrize('zeta', [0.05, 1.0, 1.7])
def test_stationary_balance_and_covariance_composition(zeta):
    model = Oscillator(3.0, zeta, 0.7)
    f, q = model.transition(0.3)
    np.testing.assert_allclose(f @ model.stationary_covariance @ f.T + q,
                               model.stationary_covariance, atol=2e-13)
    f1, q1 = model.transition(0.1)
    f2, q2 = model.transition(0.2)
    np.testing.assert_allclose(f2 @ f1, f, atol=2e-13)
    np.testing.assert_allclose(f2 @ q1 @ f2.T + q2, q, atol=2e-13)
    np.testing.assert_allclose(process_covariance(3.0, zeta, 0.7, 0.3, 48), q, atol=2e-13)


def test_dense_conditioning_all_masks_and_process_noise(evidence):
    assert len(evidence['inference_cases']) == 8
    for row in evidence['inference_cases']:
        assert row['max_mean_error'] < 5e-11
        assert row['max_covariance_error'] < 5e-12
        assert row['min_covariance_eigenvalue'] > -1e-12
        assert row['max_smoothing_variance_increment'] < 1e-12


def test_generalized_alpha_second_order_against_continuous_solution(evidence):
    finest = [r for r in evidence['time_refinement'] if r['steps'] == 400]
    assert len(finest) == 2
    for row in finest:
        assert 1.95 < row['observed_order'] < 2.05
        assert row['scaled_state_error'] < 3e-5


def test_unclipped_dither_moments_and_bound(evidence):
    d = evidence['dither']
    assert abs(d['normalized_mean']) < 0.005
    assert abs(d['variance_ratio']-1) < 0.02
    assert d['max_absolute_error_over_step'] <= 0.5+1e-13


def test_clipping_invalidates_uniform_error_model():
    signal = np.full(1001, 0.1)
    dither = np.linspace(-0.0099, 0.0099, 1001)
    result = quantize(signal, 0.02, dither=dither, max_code=1)
    assert result.saturated.all()
    assert abs(np.mean(result.values-signal)) > 0.07


def test_reference_initial_value_and_drift():
    for zeta in (0.05, 1.0, 1.7):
        np.testing.assert_array_equal(transition(3.0, zeta, 0), np.eye(2))
        h = 1e-5
        derivative = (transition(3.0, zeta, h)-transition(3.0, zeta, -h))/(2*h)
        np.testing.assert_allclose(derivative, [[0, 1], [-9, -6*zeta]], atol=2e-8)
