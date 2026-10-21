import numpy as np
import pytest
from forcewindow._core import Recurrence
from numpy.testing import assert_allclose
from scipy.integrate import solve_ivp

from forcewindow import Oscillator, Window, reconstruct


@pytest.mark.parametrize("tail", [0, 1, 17])
@pytest.mark.parametrize("loads", [1, 8, 53])
def test_forward_adjoint_dense_and_inner_product(loads, tail):
    w = Window(Oscillator(), 0.013, loads, tail)
    rng = np.random.default_rng(317)
    f, r = rng.normal(size=loads), rng.normal(size=loads + tail)
    h = w.dense()
    assert_allclose(w @ f, h @ f, atol=1e-16, rtol=1e-12)
    assert_allclose(w.rmatvec(r), h.T @ r, atol=1e-16, rtol=1e-12)
    assert_allclose(r @ (w @ f), f @ w.rmatvec(r), atol=1e-16, rtol=1e-12)


def test_terminal_load_is_exact_null_and_tail_exposes_it():
    f = np.eye(12)[-1]
    assert np.array_equal(Window(Oscillator(), 0.01, 12) @ f, np.zeros(12))
    assert np.linalg.norm(Window(Oscillator(), 0.01, 12, 20) @ f) > 0.002


@pytest.mark.parametrize(
    "osc",
    [Oscillator(2, 0, 50), Oscillator(2, 20, 50), Oscillator(2, 40, 50), Oscillator(2, 0, 0)],
)
def test_zoh_against_independent_continuous_ode(osc):
    w = Window(osc, 0.01, 101)
    times = np.arange(101) * 0.01
    ref = solve_ivp(
        lambda t, x: [x[1], (1 - osc.damping * x[1] - osc.stiffness * x[0]) / osc.mass],
        (0, 1),
        [0, 0],
        t_eval=times,
        rtol=2e-12,
        atol=2e-14,
        method="DOP853",
    )
    assert ref.success
    assert_allclose(w @ np.ones(101), ref.y[0], atol=2e-12, rtol=2e-10)


def test_free_mass_constant_force_exact_solution():
    w = Window(Oscillator(2, 0, 0), 0.017, 101)
    assert_allclose(w @ np.ones(101), (np.arange(101) * 0.017) ** 2 / 4, atol=2e-15)


@pytest.mark.parametrize("tail", [0, 1, 15])
def test_ridge_against_augmented_lstsq(tail):
    w = Window(Oscillator(), 0.02, 35, tail)
    h = w.dense()
    y = np.random.default_rng(4).normal(size=w.shape[0]) * 1e-3
    lam = 3e-4
    actual, info = reconstruct(w, y, lam)
    ref = np.linalg.lstsq(np.vstack([h, lam * np.eye(35)]), np.r_[y, np.zeros(35)], rcond=None)[
        0
    ]
    assert_allclose(actual, ref, atol=2e-8, rtol=2e-8)
    assert info["normal_residual_l2"] < 1e-13
    if tail == 0:
        assert actual[-1] == 0


def test_failure_is_reported_and_zero_data_works():
    w = Window(Oscillator(), 0.01, 80, 10)
    with pytest.raises(RuntimeError, match="did not converge"):
        reconstruct(w, np.ones(90), 1e-6, maxiter=1)
    f, info = reconstruct(w, np.zeros(90), 0.001)
    assert np.count_nonzero(f) == 0 and info["stop"] == 0


@pytest.mark.parametrize(
    "args",
    [{"mass": 0}, {"mass": -1}, {"mass": np.nan}, {"damping": -1}, {"stiffness": np.inf}],
)
def test_bad_physical_parameters(args):
    with pytest.raises(ValueError):
        Oscillator(**args)


@pytest.mark.parametrize(
    "dt,n,tail",
    [
        (0, 2, 0),
        (np.nan, 2, 0),
        (0.1, 0, 0),
        (0.1, 2.5, 0),
        (0.1, 2, -1),
        (0.1, True, 0),
        (0.1, 10_000_001, 0),
    ],
)
def test_bad_window(dt, n, tail):
    with pytest.raises(ValueError):
        Window(Oscillator(), dt, n, tail)


@pytest.mark.parametrize("lam", [0, -1, np.nan, np.inf])
def test_bad_regularization(lam):
    with pytest.raises(ValueError):
        reconstruct(Window(Oscillator(), 0.1, 3), np.ones(3), lam)


def test_raw_core_checks_lengths_and_finiteness():
    r = Recurrence([1, 0, 0, 1], [0, 1], 3, 1)
    for value in ([1, 2], [1, 2, np.nan]):
        with pytest.raises(ValueError):
            r.forward(value)
    with pytest.raises(ValueError):
        r.adjoint([1, 2, 3])
    with pytest.raises(ValueError):
        Recurrence([1, np.nan, 0, 1], [0, 1], 3, 0)


def test_complex_input_is_not_silently_discarded():
    with pytest.raises(ValueError):
        Window(Oscillator(), 0.1, 3) @ np.ones(3, dtype=complex)
