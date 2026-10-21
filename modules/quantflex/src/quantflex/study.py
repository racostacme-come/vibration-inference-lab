"""Reproduce the synthetic study: quantflex --output results."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import NullLocator  # noqa: E402

from .core import HarmonicModel, elastic_energy, quantize, steady_coefficients  # noqa: E402

SEED = 20261020
REPLICATES = 10000


def experiment(amplitude_steps, n, *, seed, max_code=None, replicates=REPLICATES):
    """One fixed signal, independent dither for every replicate and sample."""
    m, c, k, w, step = 1.0, 0.4, 100.0, 2 * np.pi, 1e-3
    force = amplitude_steps * step * abs(complex(k - m * w**2, c * w))
    beta = steady_coefficients(m, c, k, w, force)
    t = np.arange(n) / 40.0
    model = HarmonicModel(t, w)
    signal = model.design @ beta
    raw = model.fit(quantize(signal, step, max_code=max_code).values)
    rng = np.random.default_rng(seed)
    x = np.broadcast_to(signal, (replicates, n))
    observations = quantize(
        x, step, dither=rng.uniform(-step / 2, step / 2, x.shape), max_code=max_code
    )
    estimates = model.fit(observations.values)
    covariance = model.covariance(step**2 / 12)
    squared_errors = np.sum((estimates[:, 1:] - beta[1:]) ** 2, axis=1)
    energy = elastic_energy(estimates, k)
    corrected = elastic_energy(estimates, k, covariance=covariance)
    true_energy = float(elastic_energy(beta, k))
    energy_se = float(energy.std(ddof=1) / np.sqrt(replicates))
    row = {
        "amplitude_steps": amplitude_steps,
        "samples": n,
        "max_code": 0 if max_code is None else max_code,
        "force_N": force,
        "true_energy_J": true_energy,
        "raw_harmonic_mse_m2": float(np.sum((raw[1:] - beta[1:]) ** 2)),
        "dither_harmonic_mse_m2": float(squared_errors.mean()),
        "mse_se_m2": float(squared_errors.std(ddof=1) / np.sqrt(replicates)),
        "theory_harmonic_mse_m2": float(np.trace(covariance[1:, 1:])),
        "raw_energy_J": float(elastic_energy(raw, k)),
        "dither_energy_mean_J": float(energy.mean()),
        "corrected_energy_mean_J": float(corrected.mean()),
        "energy_se_J": energy_se,
        "theory_energy_bias_J": float(k / 4 * np.trace(covariance[1:, 1:])),
        "negative_corrected_fraction": float(np.mean(corrected < 0)),
        "saturated_fraction": float(observations.saturated.mean()),
        "coefficient_mean_max_z": float(
            np.max(
                np.abs(estimates.mean(axis=0) - beta) / np.sqrt(np.diag(covariance) / replicates)
            )
        ),
        "mse_z": float(
            (squared_errors.mean() - np.trace(covariance[1:, 1:]))
            / (squared_errors.std(ddof=1) / np.sqrt(replicates))
        ),
        "corrected_energy_z": float((corrected.mean() - true_energy) / energy_se),
    }
    # This diagnostic measures implementation roundoff, not continuum error.
    oracle = np.linalg.lstsq(model.design, observations.values[:5].T, rcond=None)[0].T
    row["qr_oracle_max_error_m"] = float(np.max(np.abs(estimates[:5] - oracle)))
    return row, t, signal, observations.values[0], raw, corrected


def run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    configs = [(a, n, None) for a in [0.05, 0.2, 0.8, 2.0] for n in [40, 160, 640]]
    configs += [(0.0, 160, None), (2.0, 160, 1)]
    for i, (amplitude, n, code) in enumerate(configs):
        row, t, signal, first, raw, corrected = experiment(
            amplitude, n, seed=SEED + i, max_code=code
        )
        rows.append(row)
        if amplitude == 0.2 and n == 160:
            history = np.column_stack((t, signal, first, quantize(signal, 1e-3).values))
        if amplitude == 0:
            zero_energies = corrected
    with (output / "study.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    with (output / "history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["time_s", "exact_m", "dithered_m", "rounded_m"])
        writer.writerows(history)
    safe = rows[:-1]
    metrics = {
        "seed": SEED,
        "replicates_per_case": REPLICATES,
        "cases": len(rows),
        "model": {
            "mass_kg": 1,
            "damping_Ns_m": 0.4,
            "stiffness_N_m": 100,
            "frequency_Hz": 1,
            "sample_rate_Hz": 40,
            "step_m": 0.001,
        },
        "max_unclipped_coefficient_z": max(r["coefficient_mean_max_z"] for r in safe),
        "max_unclipped_mse_abs_z": max(abs(r["mse_z"]) for r in safe),
        "max_unclipped_energy_abs_z": max(abs(r["corrected_energy_z"]) for r in safe),
        "qr_oracle_max_error_m": max(r["qr_oracle_max_error_m"] for r in rows),
        "zero_energy_negative_fraction": rows[-2]["negative_corrected_fraction"],
        "clipped_fraction": rows[-1]["saturated_fraction"],
        "clipped_corrected_energy_ratio": rows[-1]["corrected_energy_mean_J"]
        / rows[-1]["true_energy_J"],
        "physical_validation": False,
    }
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    table = [
        r"\begin{tabular}{rrrrrr}",
        r"\toprule",
        r"$A/\Delta$ & $N$ & Rounded MSE & Dither MSE & Theory MSE & $\overline E_c/E$\\",
        r"\midrule",
    ]
    for r in rows:
        if r["amplitude_steps"] in [0.05, 0.2] and r["samples"] in [40, 640]:
            table.append(
                f"{r['amplitude_steps']:.2f} & {r['samples']} & "
                f"{r['raw_harmonic_mse_m2'] / 1e-6:.5f} & "
                f"{r['dither_harmonic_mse_m2'] / 1e-6:.5f} & "
                f"{r['theory_harmonic_mse_m2'] / 1e-6:.5f} & "
                f"{r['corrected_energy_mean_J'] / r['true_energy_J']:.3f} " + r"\\"
            )
    table.extend([r"\bottomrule", r"\end{tabular}"])
    (output / "table.tex").write_text("\n".join(table) + "\n", encoding="utf-8", newline="\n")
    macros = {
        "CoefZ": f"{metrics['max_unclipped_coefficient_z']:.3f}",
        "MseZ": f"{metrics['max_unclipped_mse_abs_z']:.3f}",
        "EnergyZ": f"{metrics['max_unclipped_energy_abs_z']:.3f}",
        "OracleError": f"{metrics['qr_oracle_max_error_m']:.3g}",
        "NegativePercent": f"{100 * metrics['zero_energy_negative_fraction']:.2f}",
        "ClippedPercent": f"{100 * metrics['clipped_fraction']:.2f}",
        "ClippedRatio": f"{metrics['clipped_corrected_energy_ratio']:.4f}",
    }
    (output / "numbers.tex").write_text(
        "\n".join(rf"\newcommand{{\{key}}}{{{value}}}" for key, value in macros.items()) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(9, 5.8), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(history[:80, 0], history[:80, 1] * 1000, label="exact")
    ax.plot(
        history[:80, 0], history[:80, 2] * 1000, ".", ms=3, alpha=0.6, label="one dither record"
    )
    ax.plot(history[:80, 0], history[:80, 3] * 1000, "--", label="rounded")
    ax.set(
        xlabel="Time (s)", ylabel="Displacement (mm)", title="A. Sub-step vibration, A/step = 0.2"
    )
    ax.legend(fontsize=7, loc="lower right")
    ax = axes[0, 1]
    for a, marker in [(0.05, "o"), (0.2, "s")]:
        rs = [r for r in safe if r["amplitude_steps"] == a]
        ns = [r["samples"] for r in rs]
        ax.loglog(
            ns, [r["dither_harmonic_mse_m2"] / 1e-6 for r in rs], marker, label=f"dither A/step={a}"
        )
        ax.loglog(ns, [r["raw_harmonic_mse_m2"] / 1e-6 for r in rs], "--", label=f"rounded {a}")
    ax.loglog([40, 160, 640], 1 / (3 * np.array([40, 160, 640])), "k-", label="1/(3N)")
    ax.set(
        xlabel="Number of samples N",
        ylabel="Harmonic MSE / step squared",
        title="B. Dither has a variance cost",
    )
    ax.legend(fontsize=7)
    ax = axes[1, 0]
    rs = [r for r in safe if r["amplitude_steps"] == 0.05]
    ns = np.array([r["samples"] for r in rs])
    for key, label, marker in [
        ("dither_energy_mean_J", "uncorrected", "o-"),
        ("corrected_energy_mean_J", "bias corrected", "s-"),
    ]:
        ax.errorbar(
            ns,
            [r[key] / r["true_energy_J"] for r in rs],
            yerr=[2 * r["energy_se_J"] / r["true_energy_J"] for r in rs],
            fmt=marker,
            capsize=3,
            label=label,
        )
    ax.axhline(1, color="black", lw=1)
    ax.set(
        xscale="log",
        xlabel="Number of samples N",
        ylabel="Mean energy / true energy",
        title="C. A/step = 0.05; bars: two MC SE",
    )
    ax.legend(fontsize=8)
    ax = axes[1, 1]
    ax.hist(zero_energies * 1e6, bins=55, color="#336c92", alpha=0.85)
    ax.axvline(0, color="black", lw=1)
    ax.set(
        xlabel="Corrected energy (microjoules)",
        ylabel="Replicates",
        title="D. Zero signal: unbiased can be negative",
    )
    for ax in [axes[0, 1], axes[1, 0]]:
        ax.set_xticks([40, 160, 640], ["40", "160", "640"])
        ax.xaxis.set_minor_locator(NullLocator())
    fig.savefig(output / "study.png", dpi=190)
    plt.close(fig)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2))


if __name__ == "__main__":
    main()
