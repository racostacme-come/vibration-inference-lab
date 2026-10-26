"""Connect deterministic dynamics, quantization and missing-observation inference."""
import numpy as np
from dampwatch.dynamics import System, exact_response
from quantflex.core import quantize
from gapsmooth.model import Oscillator
from gapsmooth.inference import infer

def ensemble_experiment(trials=256):
    """Prior-matched repeated trials; report Monte Carlo standard errors."""
    times = np.linspace(0, 4, 81)
    missing = (times >= 1.5) & (times <= 2.5)
    omega, step = 2*np.pi, 0.02
    model = Oscillator(omega=omega, zeta=0.05, q=0.0)
    prior = np.diag([0.04, 1.0])
    fs = np.array([model.transition(float(t))[0] for t in times])
    rng = np.random.default_rng(20261025)
    errors = []
    for _ in range(trials):
        initial = rng.multivariate_normal(np.zeros(2), prior)
        truth = np.einsum('nij,j->ni', fs, initial)[:, 0]
        y = quantize(truth, step, dither=rng.uniform(-step/2, step/2, len(times))).values
        y[missing] = np.nan
        estimate = infer(model, times, y, step**2/12, np.zeros(2), prior)
        errors.append([np.mean((mean[missing, 0]-truth[missing])**2)
                       for mean in (estimate.filtered_mean, estimate.smoothed_mean)])
    errors = np.array(errors)
    expected = [float(np.mean(cov[missing, 0, 0])) for cov in
                (estimate.filtered_covariance, estimate.smoothed_covariance)]
    return {'seed': 20261025, 'trials': trials, 'samples_per_trial': len(times),
            'missing_per_trial': int(missing.sum()),
            'filtered_mse': float(np.mean(errors[:, 0])),
            'smoothed_mse': float(np.mean(errors[:, 1])),
            'filtered_expected_mse': expected[0], 'smoothed_expected_mse': expected[1],
            'filtered_mse_standard_error': float(np.std(errors[:, 0], ddof=1)/np.sqrt(trials)),
            'smoothed_mse_standard_error': float(np.std(errors[:, 1], ddof=1)/np.sqrt(trials)),
            'paired_improvement_standard_error': float(np.std(errors[:, 0]-errors[:, 1], ddof=1)/np.sqrt(trials)),
            'scope': 'Independent prior draws and unclipped uniform dither; average linear-estimator risk, not Gaussian posterior coverage.'}


def experiment():
    times = np.linspace(0, 4, 201)
    omega, zeta, step = 2*np.pi, 0.05, 0.02
    model = Oscillator(omega=omega, zeta=zeta, q=0.0)
    system = System([[1.0]], [[2*zeta*omega]], [[omega**2]])
    u, v = exact_response(system, [0.1], [0.0], times)
    transition = np.array([model.transition(float(t))[0] @ np.array([0.1, 0.0]) for t in times])
    rng = np.random.default_rng(20261021)
    measurement = quantize(u[:, 0], step, dither=rng.uniform(-step/2, step/2, len(times)))
    y = measurement.values.copy()
    missing = (times >= 1.5) & (times <= 2.5)
    y[missing] = np.nan
    estimate = infer(model, times, y, step**2/12, np.zeros(2), np.diag([0.04, 1.0]))
    return {"experiment": "quantized-vibration-with-outage", "seed": 20261021,
        "samples": len(times), "missing_samples": int(missing.sum()),
        "transition_max_difference": float(np.max(abs(transition-np.column_stack((u[:,0],v[:,0]))))),
        "observed_measurement_rmse": float(np.sqrt(np.mean((y[~missing]-u[~missing,0])**2))),
        "outage_filtered_rmse": float(np.sqrt(np.mean((estimate.filtered_mean[missing,0]-u[missing,0])**2))),
        "outage_smoothed_rmse": float(np.sqrt(np.mean((estimate.smoothed_mean[missing,0]-u[missing,0])**2))),
        "max_smoothed_minus_filtered_variance": float(np.max(estimate.smoothed_covariance[:,0,0]-estimate.filtered_covariance[:,0,0])),
        "ensemble": ensemble_experiment(),
        "scope": "Uniform subtractive dither; linear minimum-variance inference uses correct second moments, but Gaussian likelihood/posterior coverage is not exact. Fixed-state realization and separate prior-matched ensemble."}
