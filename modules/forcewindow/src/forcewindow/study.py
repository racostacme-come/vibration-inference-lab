"""Reproduce deterministic synthetic experiments; no external datasets."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import svd

from .model import Oscillator, Window, reconstruct


def run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    n, dt, sigma, lam = 120, 0.01, 2e-5, 2e-4
    oscillator = Oscillator()
    t = (np.arange(n) + 0.5) * dt
    force = 0.8 * np.sin(2 * np.pi * 1.3 * t) + 1.5 * np.exp(-(((t - 1.14) / 0.035) ** 2))
    noise = np.random.default_rng(20261013).normal(0, sigma, n + 80)
    records, paths, curves = [], [], []
    max_solver_error = 0.0
    for tail in (0, 1, 20, 80):
        w = Window(oscillator, dt, n, tail)
        h = w.dense()
        u, s, vt = svd(h, full_matrices=False)
        y = w @ force + noise[: n + tail]
        estimate, info = reconstruct(w, y, lam)
        # Independent SVD ridge solution and exact Gaussian expectation.
        inverse = (vt.T * (s / (s * s + lam * lam))) @ u.T
        max_solver_error = max(max_solver_error, float(np.max(np.abs(estimate - inverse @ y))))
        bias = inverse @ (h @ force) - force
        variance = sigma**2 * np.sum(inverse**2, axis=1)
        records.append(
            {
                "tail": tail,
                "observations": n + tail,
                "terminal_sensitivity_m_per_N": float(np.linalg.norm(h[:, -1])),
                "expected_rmse_N": float(np.sqrt(np.mean(bias**2 + variance))),
                "expected_last10_rmse_N": float(np.sqrt(np.mean((bias**2 + variance)[-10:]))),
                "realized_rmse_N": float(np.sqrt(np.mean((estimate - force) ** 2))),
                "estimated_last_N": float(estimate[-1]),
                "last_resolution_diagonal": float((inverse @ h)[-1, -1]),
                "smallest_singular_value": float(s[-1]),
                **info,
            }
        )
        for j in range(n):
            paths.append(
                {
                    "tail": tail,
                    "time_s": float(t[j]),
                    "true_force_N": float(force[j]),
                    "estimated_force_N": float(estimate[j]),
                }
            )
        for damping in np.logspace(-6, -2, 41):
            inv = (vt.T * (s / (s * s + damping * damping))) @ u.T
            bias2 = np.mean((inv @ (h @ force) - force) ** 2)
            var = sigma**2 * np.sum(inv**2) / n
            curves.append(
                {
                    "tail": tail,
                    "lambda_m_per_N": float(damping),
                    "expected_rmse_N": float(np.sqrt(bias2 + var)),
                }
            )

    # Continuum input approximation: smooth sine versus midpoint held loads.
    continuum = []
    for step in (0.04, 0.02, 0.01, 0.005):
        count = round(1.2 / step)
        grid = np.arange(count + 1) * step
        reference = solve_ivp(
            lambda time, x: [x[1], np.sin(7 * time) - 0.8 * x[1] - 100 * x[0]],
            (0, 1.2),
            [0, 0],
            t_eval=grid,
            method="DOP853",
            rtol=2e-12,
            atol=2e-14,
        )
        if not reference.success:
            raise RuntimeError(reference.message)
        response = Window(oscillator, step, count, 1) @ np.sin(7 * (grid[:-1] + step / 2))
        continuum.append(
            {
                "dt_s": step,
                "max_displacement_error_m": float(np.max(np.abs(response - reference.y[0]))),
            }
        )
    constant = Window(oscillator, dt, n, 1) @ np.ones(n)
    grid = np.arange(n + 1) * dt
    decay, wd = 0.4, np.sqrt(100 - 0.4**2)
    exact = (
        1 - np.exp(-decay * grid) * (np.cos(wd * grid) + decay / wd * np.sin(wd * grid))
    ) / 100
    metrics = {
        "parameters": {
            "loads": n,
            "dt_s": dt,
            "noise_std_m": sigma,
            "lambda_m_per_N": lam,
            "seed": 20261013,
            "mass_kg": 1,
            "damping_Ns_per_m": 0.8,
            "stiffness_N_per_m": 100,
        },
        "windows": records,
        "continuum": continuum,
        "max_lsqr_svd_difference_N": max_solver_error,
        "constant_load_max_error_m": float(np.max(np.abs(constant - exact))),
        "true_last_force_N": float(force[-1]),
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    for name, rows in [("reconstruction", paths), ("regularization", curves)]:
        with (output / f"{name}.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.3), layout="constrained")
    axes[0].plot(t, force, color="black", lw=2, label="prescribed load")
    for tail, color in [(0, "#b54434"), (20, "#167d9a"), (80, "#6655a4")]:
        data = [r["estimated_force_N"] for r in paths if r["tail"] == tail]
        axes[0].plot(t, data, color=color, label=f"tail={tail}")
        curve = [r for r in curves if r["tail"] == tail]
        axes[1].loglog(
            [r["lambda_m_per_N"] for r in curve],
            [r["expected_rmse_N"] for r in curve],
            color=color,
            label=f"tail={tail}",
        )
    axes[0].set(
        xlim=(0.9, 1.2),
        xlabel="load midpoint (s)",
        ylabel="force (N)",
        title="A. End of the reconstructed load",
    )
    axes[1].axvline(lam, color="0.5", ls=":", label="fixed study lambda")
    axes[1].set(
        xlabel="regularization (m/N)",
        ylabel="expected force RMSE (N)",
        title="B. Exact bias-variance sensitivity",
    )
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.savefig(output / "window-study.pdf")
    fig.savefig(output / "window-study.png")
    plt.close(fig)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("examples/results"))
    args = parser.parse_args()
    result = run(args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
