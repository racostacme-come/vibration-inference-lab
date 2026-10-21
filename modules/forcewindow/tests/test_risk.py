import numpy as np
from numpy.testing import assert_allclose

from forcewindow import Oscillator, Window


def test_bias_variance_formula_against_deterministic_noise_quadrature():
    h = Window(Oscillator(), 0.02, 8, 5).dense()
    u, s, vt = np.linalg.svd(h, full_matrices=False)
    inv = (vt.T * (s / (s * s + 1e-6))) @ u.T
    f = np.linspace(-0.5, 1, 8)
    sigma = 1e-4
    # Equal +/- coordinate samples reproduce zero mean and sigma^2 I exactly.
    noise = sigma * np.sqrt(13) * np.vstack([np.eye(13), -np.eye(13)])
    errors = (h @ f + noise) @ inv.T - f
    analytic = (inv @ h @ f - f) ** 2 + sigma**2 * np.sum(inv**2, axis=1)
    assert_allclose(np.mean(errors**2, axis=0), analytic, atol=1e-16)


def test_sampled_smooth_input_has_second_order_displacement_error():
    from scipy.integrate import solve_ivp

    errors = []
    for dt in (0.04, 0.02, 0.01):
        n = round(1.2 / dt)
        t = np.arange(n + 1) * dt
        ref = solve_ivp(
            lambda t, x: [x[1], np.sin(7 * t) - 0.8 * x[1] - 100 * x[0]],
            (0, 1.2),
            [0, 0],
            t_eval=t,
            method="DOP853",
            rtol=2e-12,
            atol=2e-14,
        )
        out = Window(Oscillator(), dt, n, 1) @ np.sin(7 * (t[:-1] + dt / 2))
        errors.append(np.max(np.abs(out - ref.y[0])))
    assert np.all(np.array(errors[:-1]) / errors[1:] > 3.9)
    assert np.all(np.array(errors[:-1]) / errors[1:] < 4.1)
