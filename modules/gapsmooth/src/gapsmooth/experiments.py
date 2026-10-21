"""Reproducible synthetic experiments; all reported metrics originate here."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullLocator
from scipy.integrate import quad_vec
from scipy.linalg import expm
from scipy.stats import chi2

from .inference import infer, simulate
from .model import Oscillator
from .reference import batch_condition


def run_study(output, trials=256):
    """Write metrics, trace and two figures; return JSON-compatible metrics."""
    if not isinstance(trials, int) or trials < 2:
        raise ValueError("trials must be an integer >=2")
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    model = Oscillator()
    t = np.arange(129) / 16
    p0, m0, r = model.stationary_covariance, np.zeros(2), 0.02**2
    gap = (t >= 3) & (t <= 5)
    states, y = simulate(model, t, r, m0, p0, seed=2718)
    y[gap] = np.nan
    est = infer(model, t, y, r, m0, p0)
    center = 64
    trace = np.column_stack(
        [
            t,
            states,
            y,
            est.filtered_mean,
            est.smoothed_mean,
            np.sqrt(est.filtered_covariance[:, 0, 0]),
            np.sqrt(est.smoothed_covariance[:, 0, 0]),
        ]
    )
    with (out / "trace.csv").open("w", encoding="utf-8", newline="\n") as stream:
        np.savetxt(
            stream,
            trace,
            delimiter=",",
            fmt="%.17g",
            comments="",
            header="t,x_true,v_true,y,x_filter,v_filter,x_smooth,v_smooth,sd_filter,sd_smooth",
        )

    # Independent trajectories at ONE fixed time: no false independence of time samples.
    nees_f, nees_s, covered_f, covered_s = [], [], [], []
    for seed in range(10000, 10000 + trials):
        x, obs = simulate(model, t, r, m0, p0, seed)
        obs[gap] = np.nan
        e = infer(model, t, obs, r, m0, p0)
        for means, covs, nees, covered in [
            (e.filtered_mean, e.filtered_covariance, nees_f, covered_f),
            (e.smoothed_mean, e.smoothed_covariance, nees_s, covered_s),
        ]:
            error = x[center] - means[center]
            nees.append(float(error @ np.linalg.solve(covs[center], error)))
            covered.append(bool(abs(error[0]) <= 1.95996398454 * np.sqrt(covs[center, 0, 0])))

    # Short irregular grid avoids a reference check that only covers uniform sampling.
    bt = np.array([0, 0.04, 0.13, 0.2, 0.37, 0.59, 0.62, 0.91, 1.2])
    _, by = simulate(model, bt, r, m0, p0, seed=17)
    by[2:6] = np.nan
    rec = infer(model, bt, by, r, m0, p0)
    bm, bc, bnll = batch_condition(model, bt, by, r, m0, p0)
    diagonal = np.array([bc[2 * k : 2 * k + 2, 2 * k : 2 * k + 2] for k in range(len(bt))])
    a, diffusion = model.drift, np.diag([0.0, model.q])
    refinements = []
    for steps in [16, 32, 64, 128, 256]:
        dt = 1 / steps
        fe, qe = np.eye(2) + dt * a, dt * diffusion
        f, q = np.eye(2), np.zeros((2, 2))
        for _ in range(steps):
            q, f = fe @ q @ fe.T + qe, fe @ f
        exact_f, exact_q = model.transition(1)
        refinements.append(
            {
                "steps": steps,
                "dt": dt,
                "transition_error_fro": float(np.linalg.norm(f - exact_f)),
                "covariance_error_fro": float(np.linalg.norm(q - exact_q)),
            }
        )
    reference_q, _ = quad_vec(
        lambda s: expm(a * s) @ diffusion @ expm(a.T * s),
        0,
        0.3,
        epsabs=1e-14,
        epsrel=1e-13,
    )
    f, q = model.transition(0.3)
    metrics = {
        "setup": {
            "omega": model.omega,
            "zeta": model.zeta,
            "q": model.q,
            "measurement_variance": r,
            "seed_trace": 2718,
            "seed_mc_start": 10000,
            "trials": trials,
            "samples": len(t),
            "dt": 1 / 16,
            "gap_start": 3,
            "gap_end": 5,
            "evaluation_time": float(t[center]),
        },
        "verification": {
            "batch_mean_max_abs": float(np.max(abs(rec.smoothed_mean - bm))),
            "batch_covariance_max_abs": float(np.max(abs(rec.smoothed_covariance - diagonal))),
            "batch_nll_abs": float(abs(rec.nll - bnll)),
            "quadrature_covariance_max_abs": float(np.max(abs(q - reference_q))),
            "stationary_covariance_max_abs": float(np.max(abs(f @ p0 @ f.T + q - p0))),
        },
        "gap_center": {
            "filter_sd": float(np.sqrt(est.filtered_covariance[center, 0, 0])),
            "smooth_sd": float(np.sqrt(est.smoothed_covariance[center, 0, 0])),
            "variance_reduction_fraction": float(
                1
                - est.smoothed_covariance[center, 0, 0] / est.filtered_covariance[center, 0, 0]
            ),
        },
        "monte_carlo": {
            "filter_mean_nees": float(np.mean(nees_f)),
            "smooth_mean_nees": float(np.mean(nees_s)),
            "filter_displacement_coverage": float(np.mean(covered_f)),
            "smooth_displacement_coverage": float(np.mean(covered_s)),
            "mean_nees_95_interval": (chi2.ppf([0.025, 0.975], 2 * trials) / trials).tolist(),
        },
        "euler_refinement": refinements,
    }
    (out / "metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False) + "\n", newline="\n"
    )
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 5.1), sharex=True, layout="constrained")
    axes[0].plot(t, states[:, 0], color="#252a34", lw=1, label="Synthetic truth")
    axes[0].plot(t, est.smoothed_mean[:, 0], color="#126782", label="RTS mean")
    sd = np.sqrt(est.smoothed_covariance[:, 0, 0])
    axes[0].fill_between(
        t,
        est.smoothed_mean[:, 0] - 1.96 * sd,
        est.smoothed_mean[:, 0] + 1.96 * sd,
        color="#126782",
        alpha=0.2,
        label="Pointwise 95% band",
    )
    axes[0].scatter(t, y, s=7, color="#d97732", label="Observations", zorder=3)
    axes[0].set_ylabel("Displacement [m]")
    axes[0].legend(ncol=2, fontsize=8, loc="upper left")
    axes[1].plot(
        t, np.sqrt(est.filtered_covariance[:, 0, 0]), label="Causal filter", color="#d97732"
    )
    axes[1].plot(t, sd, label="Offline smoother", color="#126782")
    axes[1].set(xlabel="Time [s]", ylabel="Posterior standard deviation [m]")
    axes[1].legend(fontsize=9)
    for ax in axes:
        ax.axvspan(3, 5, color="#9aa4af", alpha=0.15)
        ax.grid(alpha=0.18)
    fig.savefig(out / "outage.pdf")
    fig.savefig(out / "outage.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.2, 2.8), layout="constrained")
    ax.loglog(
        [v["dt"] for v in refinements],
        [v["transition_error_fro"] for v in refinements],
        "o-",
        label="Transition F",
    )
    ax.loglog(
        [v["dt"] for v in refinements],
        [v["covariance_error_fro"] for v in refinements],
        "s-",
        label="Process covariance Q",
    )
    ax.set(xlabel="Euler step [s]", ylabel="Frobenius error at 1 s")
    ax.set_xticks(
        [1 / n for n in [256, 128, 64, 32, 16]],
        labels=["1/256", "1/128", "1/64", "1/32", "1/16"],
    )
    ax.xaxis.set_minor_locator(NullLocator())
    ax.legend()
    ax.grid(alpha=0.2, which="both")
    fig.savefig(out / "refinement.pdf")
    plt.close(fig)
    return metrics
