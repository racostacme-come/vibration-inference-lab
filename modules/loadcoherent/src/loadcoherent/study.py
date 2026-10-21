"""Seeded synthetic heat-load experiment; no fitted production forecasting model."""

import csv
import json
from pathlib import Path

import numpy as np

from .core import Reconciler, estimate_covariance, prediction_mse

NAMES = ["Plant", "North", "South", "N1", "N2", "S1", "S2"]


def district():
    """Four lossless building branches; return S and error covariance in kW^2."""
    s = np.vstack(([1, 1, 1, 1], [1, 1, 0, 0], [0, 0, 1, 1], np.eye(4)))
    # Common weather and two local forecast-model factors, deliberately not
    # coherent across levels. Independent residuals ensure positive definiteness.
    factors = np.array(
        [
            [7, 0, 0],
            [2, 4, 0],
            [3, 0, 3],
            [1, 2, 0],
            [1.5, -1, 0],
            [1, 0, 1.5],
            [2, 0, -1],
        ]
    )
    independent_sd = np.array([3, 2, 2.5, 1, 1.2, 1.1, 1.5])
    return s, factors @ factors.T + np.diag(independent_sd**2)


def thermal_truth(hours):
    """Quasi-steady heat losses U_i(21 - T_out), in kW; no pipe losses/storage."""
    hours = np.asarray(hours, dtype=float)
    if hours.ndim != 1 or not np.all(np.isfinite(hours)):
        raise ValueError("hours must be a finite vector")
    outdoor = 4 + 5 * np.sin(2 * np.pi * hours / 24)
    bottom = (21 - outdoor[:, None]) * np.array([1.2, 0.8, 1.5, 1.0])
    return bottom @ district()[0].T


def _csv(path, header, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def run_study(output, *, plots=True):
    """Write fixed-seed metrics and every calibration risk; return metrics.

    80 independent calibration sets per size (10, 20, 50, 200). Their exact
    conditional risks use the known synthetic covariance, so test Monte Carlo
    noise does not obscure covariance-estimation effects. A separate 60,000-row
    test sample checks population risk for fixed reference methods only.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    s, v = district()
    n = len(NAMES)
    base_risk = float(np.trace(v))
    c = np.c_[np.eye(3), -s[:3]]
    fixed = {
        "base": np.eye(n),
        "bottom_up": np.c_[np.zeros((n, 3)), s],
        "ols": Reconciler(s, np.eye(n)).projection,
        "oracle": Reconciler(s, v).projection,
        "wrong_weights": Reconciler(s, np.diag([0.01, 1, 1, 100, 100, 100, 100])).projection,
    }
    seed = 20261008
    rng_test, rng_train, rng_path = [
        np.random.default_rng(child) for child in np.random.SeedSequence(seed).spawn(3)
    ]
    errors = rng_test.standard_normal((60000, n)) @ np.linalg.cholesky(v).T
    fixed_metrics = {}
    for name, p in fixed.items():
        mse = prediction_mse(p, v)
        sample_loss = np.sum((errors @ p.T) ** 2, axis=1)
        # Exact standard error for the mean squared norm of a zero-mean Gaussian.
        reconciled_cov = p @ v @ p.T
        analytic_se = np.sqrt(2 * np.trace(reconciled_cov @ reconciled_cov) / len(errors))
        fixed_metrics[name] = {
            "mse_per_node": mse.tolist(),
            "total_mse": float(mse.sum()),
            "risk_ratio": float(mse.sum() / base_risk),
            "monte_carlo_mse": float(sample_loss.mean()),
            "monte_carlo_standard_error": float(analytic_se),
            "monte_carlo_z": float((sample_loss.mean() - mse.sum()) / analytic_se),
            "constraint_operator_max": float(np.max(np.abs(c @ p))),
            "preservation_max": float(np.max(np.abs(p @ s - s))),
        }
    rows = []
    calibration = []
    for count in (10, 20, 50, 200):
        risks = {name: [] for name in ("sample", "shrink_half", "diagonal")}
        for replicate in range(80):
            training = rng_train.standard_normal((count, n)) @ np.linalg.cholesky(v).T
            for name, alpha in (("sample", 0.0), ("shrink_half", 0.5), ("diagonal", 1.0)):
                w = estimate_covariance(training, alpha)
                p = Reconciler(s, w).projection
                ratio = float(prediction_mse(p, v).sum() / base_risk)
                rows.append([count, replicate, name, ratio, float(np.linalg.cond(w))])
                risks[name].append(ratio)
        for name, values in risks.items():
            calibration.append(
                {
                    "samples": count,
                    "method": name,
                    "mean_risk_ratio": float(np.mean(values)),
                    "mean_standard_error": float(np.std(values, ddof=1) / np.sqrt(len(values))),
                    "median_risk_ratio": float(np.median(values)),
                    "q10": float(np.quantile(values, 0.1)),
                    "q90": float(np.quantile(values, 0.9)),
                    "fraction_worse_than_base": float(np.mean(np.array(values) > 1)),
                }
            )
    # A coherent bias is invisible to any reconciliation preserving range(S).
    coherent_bias = s @ np.array([2.0, 2.0, 2.0, 2.0])
    biased_mse = prediction_mse(fixed["oracle"], v, coherent_bias)
    hours = np.arange(48)
    truth = thermal_truth(hours)
    base = truth + rng_path.standard_normal(truth.shape) @ np.linalg.cholesky(v).T
    reconciled = base @ fixed["oracle"].T
    p = fixed["oracle"]
    kkt = np.block([[v, s], [s.T, np.zeros((4, 4))]])
    independent = s @ np.linalg.solve(kkt, np.r_[base[0], np.zeros(4)])[n:]
    metrics = {
        "seed": seed,
        "nodes": NAMES,
        "covariance_kw2": v.tolist(),
        "summing_matrix": s.tolist(),
        "replicates": 80,
        "test_samples": len(errors),
        "fixed": fixed_metrics,
        "calibration": calibration,
        "coherent_bias_kw": coherent_bias.tolist(),
        "coherent_bias_preservation_max": float(
            np.max(np.abs(p @ coherent_bias - coherent_bias))
        ),
        "oracle_biased_total_mse": float(biased_mse.sum()),
        "oracle_bias_penalty": float(np.sum(coherent_bias**2)),
        "kkt_max_error": float(np.max(np.abs(independent - reconciled[0]))),
        "idempotence_max": float(np.max(np.abs(p @ p - p))),
        "path_base_max_imbalance_kw": float(np.max(np.abs(base @ c.T))),
        "path_reconciled_max_imbalance_kw": float(np.max(np.abs(reconciled @ c.T))),
    }
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n"
    )
    _csv(
        output / "calibration.csv",
        ["samples", "replicate", "method", "risk_ratio", "weight_condition"],
        rows,
    )
    path_rows = [
        [hour, node, truth[t, j], base[t, j], reconciled[t, j]]
        for t, hour in enumerate(hours)
        for j, node in enumerate(NAMES)
    ]
    _csv(
        output / "forecast_path.csv",
        ["hour", "node", "truth_kw", "base_kw", "oracle_kw"],
        path_rows,
    )
    if plots:
        _plots(output, metrics, hours, truth, base, reconciled)
    return metrics


def _plots(output, metrics, hours, truth, base, reconciled):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    colors = ["#006d77", "#e29578", "#334e68"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), constrained_layout=True)
    ax = axes[0]
    names = list(metrics["fixed"])
    ax.bar(
        np.arange(len(names)),
        [metrics["fixed"][k]["risk_ratio"] for k in names],
        color="#006d77",
    )
    ax.set_xticks(
        np.arange(len(names)), ["Base", "Bottom\nup", "OLS", "Oracle", "Wrong\nweights"]
    )
    ax.axhline(1, color="#555555", ls="--", lw=1)
    ax.set_ylabel("Total expected MSE / base MSE")
    ax.set_title("Exact risk, fixed weights")
    ax = axes[1]
    for name, label, color in zip(
        ("sample", "shrink_half", "diagonal"),
        ("Sample", "Half shrinkage", "Diagonal"),
        colors,
        strict=True,
    ):
        records = [r for r in metrics["calibration"] if r["method"] == name]
        x = [r["samples"] for r in records]
        ax.plot(x, [r["mean_risk_ratio"] for r in records], "o-", label=label, color=color)
        ax.fill_between(
            x, [r["q10"] for r in records], [r["q90"] for r in records], color=color, alpha=0.13
        )
    ax.axhline(1, color="#555555", ls="--", lw=1)
    ax.axhline(
        metrics["fixed"]["oracle"]["risk_ratio"], color="#999999", ls=":", label="Oracle"
    )
    ax.set_xscale("log")
    ax.set_xticks([10, 20, 50, 200], ["10", "20", "50", "200"])
    ax.set_xlabel("Independent calibration errors")
    ax.set_ylabel("Conditional expected MSE / base MSE")
    ax.set_title("Means and 10–90% replicate bands")
    ax.legend(fontsize=8)
    fig.savefig(output / "risk.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3), constrained_layout=True)
    for ax, j in zip(axes, [0, 3], strict=True):
        ax.plot(hours, truth[:, j], color="#334e68", label="Synthetic truth", lw=2)
        ax.plot(hours, base[:, j], color="#e29578", alpha=0.8, label="Base")
        ax.plot(hours, reconciled[:, j], color="#006d77", label="Oracle")
        ax.set(xlabel="Hour (independent error draws)", ylabel="Heat load [kW]", title=NAMES[j])
    axes[0].legend(fontsize=8)
    fig.savefig(output / "path.png", dpi=180)
    plt.close(fig)
