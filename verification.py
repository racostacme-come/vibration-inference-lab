"""Independent scalar dynamics and dense second-moment references.

No matrix exponential, module transition, or recursive filter is used by the
reference routines. Dense conditioning is deliberately restricted to small
verification grids; it is a linear estimator, not a uniform-noise posterior.
"""
import numpy as np
from numpy.polynomial.legendre import leggauss


def transition(omega, zeta, time):
    """Closed-form F(t), including under-, critical and overdamped regimes."""
    alpha = zeta * omega
    if zeta < 1:
        beta = omega * np.sqrt(1 - zeta*zeta)
        c, s = np.cos(beta*time), np.sin(beta*time)/beta
    elif zeta > 1:
        beta = omega * np.sqrt(zeta*zeta - 1)
        c, s = np.cosh(beta*time), np.sinh(beta*time)/beta
    else:
        c, s = 1.0, time
    return np.exp(-alpha*time) * np.array(
        [[c+alpha*s, s], [-omega*omega*s, c-alpha*s]])


def process_covariance(omega, zeta, intensity, time, order=96):
    """Gauss-Legendre integral of q F(s)e_v e_v^T F(s)^T."""
    nodes, weights = leggauss(order)
    columns = np.array([transition(omega, zeta, time*(x+1)/2)[:, 1]
                        for x in nodes])
    return intensity*time/2 * np.einsum('n,ni,nj->ij', weights, columns, columns)


def batch_estimate(omega, zeta, intensity, times, observations, variance,
                   mean0, covariance0):
    """Dense joint covariance; condition separately on prefixes and all data."""
    times, y = np.asarray(times), np.asarray(observations)
    n = len(times)
    fs = np.array([transition(omega, zeta, t-times[0]) for t in times])
    mean = np.einsum('nij,j->ni', fs, mean0)
    cov = np.empty((n, n, 2, 2))
    for i in range(n):
        cov[i, i] = (fs[i] @ covariance0 @ fs[i].T +
                     process_covariance(omega, zeta, intensity, times[i]-times[0]))
        for j in range(i):
            cov[i, j] = transition(omega, zeta, times[i]-times[j]) @ cov[j, j]
            cov[j, i] = cov[i, j].T
    observed = np.flatnonzero(~np.isnan(y))

    def condition(k, indices):
        if not len(indices):
            return mean[k].copy(), cov[k, k].copy()
        innovation_cov = cov[indices[:, None], indices[None, :], 0, 0] + variance*np.eye(len(indices))
        cross = cov[k, indices, :, 0].T
        gain = np.linalg.solve(innovation_cov, cross.T).T
        return mean[k] + gain @ (y[indices]-mean[indices, 0]), cov[k, k] - gain @ cross.T

    fm, fp, sm, sp = [], [], [], []
    for k in range(n):
        m, p = condition(k, observed[observed <= k]); fm.append(m); fp.append(p)
        m, p = condition(k, observed); sm.append(m); sp.append(p)
    return tuple(np.array(a) for a in (fm, fp, sm, sp))


def experiment():
    from dampwatch.dynamics import System, exact_response, integrate
    from gapsmooth.model import Oscillator
    from gapsmooth.inference import infer
    from quantflex.core import quantize

    omega = 2*np.pi
    initial = np.array([0.1, -0.03])
    transition_errors, covariance_errors, response_errors = [], [], []
    for zeta in (0.05, 1.0, 1.7):
        model = Oscillator(omega=omega, zeta=zeta, q=0.6)
        system = System([[2.0]], [[4*zeta*omega]], [[2*omega**2]])
        for t in (0.0, 1e-4, 0.07, 0.6, 2.0):
            f, q = model.transition(t)
            reference = transition(omega, zeta, t)
            transition_errors.append(np.max(abs(f-reference)))
            covariance_errors.append(np.max(abs(q-process_covariance(omega, zeta, 0.6, t))))
            u, v = exact_response(system, initial[:1], initial[1:], [t])
            response_errors.append(np.max(abs(np.array([u[0, 0], v[0, 0]])-reference @ initial)))

    times = np.array([2.0, 2.013, 2.06, 2.19, 2.31, 2.55, 2.8, 3.11])
    mean0, p0 = np.array([0.03, -0.1]), np.array([[0.04, 0.01], [0.01, 0.3]])
    inference_rows = []
    for intensity in (0.0, 0.6):
        for mask in ('interior', 'leading-trailing', 'all-missing', 'all-observed'):
            y = 0.1*np.cos(omega*(times-times[0]))
            if mask == 'interior': y[2:6] = np.nan
            if mask == 'leading-trailing': y[[0, 1, 6, 7]] = np.nan
            if mask == 'all-missing': y[:] = np.nan
            estimate = infer(Oscillator(omega, 0.05, intensity), times, y, 0.02**2/12, mean0, p0)
            reference = batch_estimate(omega, 0.05, intensity, times, y, 0.02**2/12, mean0, p0)
            arrays = (estimate.filtered_mean, estimate.filtered_covariance,
                      estimate.smoothed_mean, estimate.smoothed_covariance)
            inference_rows.append({'intensity': intensity, 'mask': mask,
                                   'max_mean_error': float(max(np.max(abs(arrays[i]-reference[i])) for i in (0, 2))),
                                   'max_covariance_error': float(max(np.max(abs(arrays[i]-reference[i])) for i in (1, 3))),
                                   'min_covariance_eigenvalue': float(np.min(np.linalg.eigvalsh(estimate.smoothed_covariance))),
                                   'max_smoothing_variance_increment': float(np.max(np.linalg.eigvalsh(estimate.smoothed_covariance-estimate.filtered_covariance)))})

    refinement = []
    system = System([[1.0]], [[0.1*omega]], [[omega**2]])
    for rho in (0.5, 1.0):
        previous = None
        for steps in (50, 100, 200, 400):
            h = integrate(system, initial[:1], initial[1:], dt=1.3/steps, steps=steps, rho_inf=rho)
            exact = np.array([transition(omega, 0.05, t) @ initial for t in h.time])
            error = float(np.max(np.linalg.norm((np.column_stack((h.displacement[:, 0], h.velocity[:, 0]))-exact)/[1, omega], axis=1)))
            refinement.append({'rho_inf': rho, 'steps': steps, 'scaled_state_error': error,
                               'observed_order': None if previous is None else float(np.log2(previous/error))})
            previous = error

    rng = np.random.default_rng(20261025)
    step, n = 0.02, 120000
    signal = rng.uniform(-0.1, 0.1, n)
    dither = rng.uniform(-step/2, step/2, n)
    error = quantize(signal, step, dither=dither).values-signal
    return {'experiment': 'independent-vibration-verification',
            'transition_max_error': float(max(transition_errors)),
            'response_max_error': float(max(response_errors)),
            'process_covariance_max_error': float(max(covariance_errors)),
            'inference_cases': inference_rows, 'time_refinement': refinement,
            'dither': {'seed': 20261025, 'samples': n, 'step': step,
                       'normalized_mean': float(np.mean(error)/step),
                       'variance_ratio': float(np.var(error)/(step**2/12)),
                       'max_absolute_error_over_step': float(np.max(abs(error))/step)},
            'scope': 'Synthetic verification only. Dense conditioning checks linear second-moment estimators; it does not validate Gaussian likelihood or physical sensor data.'}


def write_table(data, path):
    rows = [r'\begin{center}\small', r'\begin{tabular}{lr}\toprule',
            r'Independent check & Maximum absolute difference\\\midrule']
    for label, key in [('Closed-form transition', 'transition_max_error'),
                       ('Dimensional free response', 'response_max_error'),
                       ('Process covariance quadrature', 'process_covariance_max_error')]:
        rows.append(f'{label} & {data[key]:.3e}'+r'\\')
    for label, key in [('Batch vs. recursive means', 'max_mean_error'),
                       ('Batch vs. recursive covariances', 'max_covariance_error')]:
        rows.append(f'{label} & {max(r[key] for r in data["inference_cases"]):.3e}'+r'\\')
    rows += [r'\bottomrule\end{tabular}', r'\par\medskip',
             r'\begin{tabular}{rrrr}\toprule',
             r'$\rho_\infty$ & Steps & Scaled state error & Observed order\\\midrule']
    for r in data['time_refinement']:
        order = '--' if r['observed_order'] is None else f'{r["observed_order"]:.4f}'
        rows.append(f'{r["rho_inf"]} & {r["steps"]} & {r["scaled_state_error"]:.3e} & {order}'+r'\\')
    rows += [r'\bottomrule\end{tabular}',
             f'\\par Dither variance ratio: {data["dither"]["variance_ratio"]:.6f}; normalized mean: {data["dither"]["normalized_mean"]:.3e}.',
             r'\end{center}']
    path.write_text('\n'.join(rows)+'\n', encoding='utf-8', newline='\n')
