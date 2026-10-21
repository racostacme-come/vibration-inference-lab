"""Reproduce population scoring and elastic extension-risk experiments."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist, pdist

from . import Gaussian, elastic_response, energy_score, expected_norm, expected_scores

NAMES = ["Calibrated", "Independent", "Anticorrelated", "Common bias"]


def experiment():
    """Deterministic population results plus an explicitly finite-sample check."""
    scale = 1e-3
    k = 1e6 * np.array([[2.0, -1.0], [-1.0, 2.0]])
    target = np.array([[1.0, 0.9], [0.9, 1.0]])
    force = Gaussian([0, 0], k @ (scale**2 * target) @ k.T)
    response = elastic_response(k, force)
    truth = Gaussian(response.mean / scale, response.covariance / scale**2)
    models = [
        truth,
        Gaussian([0, 0], np.eye(2)),
        Gaussian([0, 0], [[1, -0.5], [-0.5, 1]]),
        Gaussian([1, 1], truth.covariance),
    ]
    rows = []
    for name, g in zip(NAMES, models, strict=True):
        rows.append(
            {
                "model": name,
                **expected_scores(g, truth),
                "extension_exceedance": g.exceedance([-1, 1], 1),
            }
        )
    sweep = []
    for rho in np.linspace(-0.9, 0.99, 64):
        g = Gaussian([0, 0], [[1, rho], [rho, 1]])
        sweep.append(
            {
                "rho": float(rho),
                **expected_scores(g, truth),
                "extension_exceedance": g.exceedance([-1, 1], 1),
            }
        )
    # Each replicate has independent observations and ensembles. All candidate
    # models share latent normal draws within a replicate to enable paired checks.
    rng = np.random.default_rng(20261020)
    finite = []
    for rep in range(24):
        y = rng.standard_normal((256, 2)) @ np.linalg.cholesky(truth.covariance).T
        latent = rng.standard_normal((128, 2))
        for name, g in zip(NAMES, models, strict=True):
            x = latent @ np.linalg.cholesky(g.covariance).T + g.mean
            finite.append(
                {
                    "replicate": rep,
                    "model": name,
                    "energy_unbiased": float(energy_score(x, y).mean()),
                    "energy_empirical": float(energy_score(x, y, False).mean()),
                }
            )
    oracle_x = rng.normal(size=(37, 2))
    oracle_y = rng.normal(size=(53, 2))
    oracle = cdist(oracle_y, oracle_x).mean(axis=1) - pdist(oracle_x).sum() / (37 * 36)
    native_error = float(np.max(np.abs(energy_score(oracle_x, oracle_y) - oracle)))
    relative_equilibrium = float(
        np.linalg.norm(k @ response.covariance @ k.T - force.covariance)
        / np.linalg.norm(force.covariance)
    )
    rayleigh_error = float(abs(expected_norm(Gaussian([0, 0], np.eye(2)))[0] - np.sqrt(np.pi / 2)))
    summary = {}
    for row in rows:
        name = row["model"]
        scores = np.array([r["energy_unbiased"] for r in finite if r["model"] == name])
        empirical = np.array([r["energy_empirical"] for r in finite if r["model"] == name])
        expected_within, _ = expected_norm(
            Gaussian([0, 0], 2 * models[NAMES.index(name)].covariance)
        )
        summary[name] = {
            "energy_mean": float(scores.mean()),
            "energy_replicate_se": float(scores.std(ddof=1) / np.sqrt(scores.size)),
            "observed_empirical_minus_unbiased": float(np.mean(empirical - scores)),
            "expected_empirical_bias": float(expected_within / (2 * 128)),
        }
    metrics = {
        "setup": {
            "seed": 20261020,
            "replicates": 24,
            "cases_per_replicate": 256,
            "ensemble_members": 128,
            "displacement_scale_m": scale,
            "spring_stiffness_N_per_m": 1e6,
            "area_m2": 1e-4,
            "stress_threshold_Pa": 1e7,
            "truth_correlation": 0.9,
        },
        "population": rows,
        "finite_ensemble": summary,
        "validation": {
            "cpp_distance_oracle_max_abs": native_error,
            "equilibrium_covariance_relative": relative_equilibrium,
            "isotropic_norm_absolute_error": rayleigh_error,
        },
        "risk_ratio_independent_to_truth": rows[1]["extension_exceedance"]
        / rows[0]["extension_exceedance"],
    }
    return metrics, sweep, finite


def _write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_results(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    metrics, sweep, finite = experiment()
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    _write_csv(output / "population.csv", metrics["population"])
    _write_csv(output / "correlation.csv", sweep)
    _write_csv(output / "replicates.csv", finite)
    # Machine-generated table ensures paper values have a direct data lineage.
    table = (
        "\\begin{tabular}{lrrrrr}\n\\toprule\n"
        "Forecast & CRPS & LogS & ES & VS & Risk (\\%) \\\\\n\\midrule\n"
    )
    for row in metrics["population"]:
        table += (
            f"{row['model']} & {row['marginal_crps']:.4f} & {row['log_score']:.4f} & "
            f"{row['energy']:.4f} & {row['variogram']:.4f} & "
            f"{100 * row['extension_exceedance']:.3f} \\\\\n"
        )
    table += "\\bottomrule\n\\end{tabular}\n"
    (output / "table.tex").write_text(table, encoding="utf-8", newline="\n")
    finite_calibrated = metrics["finite_ensemble"]["Calibrated"]
    values = {
        "RiskRatio": f"{metrics['risk_ratio_independent_to_truth']:.2f}",
        "NativeError": f"{metrics['validation']['cpp_distance_oracle_max_abs']:.2e}",
        "EquilibriumError": f"{metrics['validation']['equilibrium_covariance_relative']:.2e}",
        "NormError": f"{metrics['validation']['isotropic_norm_absolute_error']:.2e}",
        "CalibratedMC": f"{metrics['finite_ensemble']['Calibrated']['energy_mean']:.5f}",
        "CalibratedSE": f"{metrics['finite_ensemble']['Calibrated']['energy_replicate_se']:.5f}",
        "BiasObserved": f"{finite_calibrated['observed_empirical_minus_unbiased']:.5f}",
        "BiasExpected": f"{finite_calibrated['expected_empirical_bias']:.5f}",
    }
    (output / "numbers.tex").write_text(
        "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in values.items()),
        encoding="utf-8",
        newline="\n",
    )
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.35), layout="constrained")
    rho = [r["rho"] for r in sweep]
    best = metrics["population"][0]
    for key, label, color in [
        ("marginal_crps", "Marginal CRPS", "#707070"),
        ("energy", "Energy", "#007d91"),
        ("variogram", "Variogram", "#c87520"),
    ]:
        ax[0].plot(rho, [r[key] - best[key] for r in sweep], label=label, color=color, lw=2)
    ax[0].axvline(0.9, color="#333333", ls=":", lw=1)
    ax[0].set(
        xlabel="Forecast correlation",
        ylabel="Excess population loss",
        title="(a) Dependence diagnostics",
    )
    ax[0].legend(frameon=False, fontsize=9)
    ax[1].plot(rho, [100 * r["extension_exceedance"] for r in sweep], color="#813c69", lw=2)
    for r, color in [(metrics["population"][0], "#007d91"), (metrics["population"][1], "#c87520")]:
        x = 0.9 if r["model"] == "Calibrated" else 0
        ax[1].plot(x, 100 * r["extension_exceedance"], "o", color=color)
    ax[1].set(
        xlabel="Forecast correlation",
        ylabel="Exceedance probability (%)",
        title="(b) Absolute spring stress > 10 MPa",
    )
    fig.savefig(output / "joint-study.png", dpi=220)
    plt.close(fig)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    metrics = write_results(args.output)
    print(json.dumps(metrics["validation"], indent=2))
    ratio = metrics["risk_ratio_independent_to_truth"]
    print(f"Independent/truth stress exceedance ratio: {ratio:.4f}")


if __name__ == "__main__":
    main()
