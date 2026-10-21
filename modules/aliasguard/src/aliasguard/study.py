"""Deterministic experiments: analytical modes, measured FIR outputs, alias budgets."""

import csv
import json
from pathlib import Path

import numpy as np
from scipy.signal import firwin, freqz, lfilter

from .signal import Decimator, frequency_response, lowpass

FS = 2048.0
FACTOR = 8
FREQUENCIES = np.array([23.0, 71.0, 185.0])
AMPLITUDES = np.array([1.0, 0.35, 0.6])
TAPS = (65, 129, 257, 513)
COUNT = 2048
CUTOFF = 96.0


def modal_signal(times, phase=0.0):
    """Sensor displacement in mm from three independent undamped modal coordinates."""
    phases = np.array([0.0, 0.0, phase])
    return np.cos(2 * np.pi * np.outer(times, FREQUENCIES) + phases) @ AMPLITUDES


def measure_amplitude(times, values, frequency=71.0):
    """Known-frequency harmonic least squares with DC and the 23 Hz nuisance mode."""
    angle = 2 * np.pi * times
    design = np.column_stack(
        [
            np.ones(times.size),
            np.cos(23 * angle),
            np.sin(23 * angle),
            np.cos(frequency * angle),
            np.sin(frequency * angle),
        ]
    )
    coef, _, rank, _ = np.linalg.lstsq(design, values, rcond=None)
    if rank != 5:
        raise ValueError("harmonic regression is rank deficient")
    return float(np.hypot(coef[-2], coef[-1]))


def aligned_filtered(h, phase=0.0):
    """Use true modal prehistory; discard startup and label by group-delay centers."""
    delay = (h.size - 1) // 2
    if 2 * delay % FACTOR:
        raise ValueError("study requires startup endpoint aligned with decimation phase")
    t_input = (np.arange(COUNT * FACTOR + 2 * delay) - delay) / FS
    x = modal_signal(t_input, phase)
    stream = Decimator(h, FACTOR)
    # Deliberately irregular chunks exercise state carried between calls.
    y = np.concatenate(
        [
            stream.process(x[:37]),
            stream.process(x[37:1082]),
            stream.process([]),
            stream.process(x[1082:]),
        ]
    )
    return x, y, y[2 * delay // FACTOR :][:COUNT]


def csv_write(path, header, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def run_study(output, *, plots=True):
    """Write JSON, CSV, and optional figures; return the measured metrics."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    times = np.arange(COUNT) / (FS / FACTOR)
    target = np.cos(2 * np.pi * np.outer(times, FREQUENCIES[:2])) @ AMPLITUDES[:2]
    raw = modal_signal(times)
    records, filtered = [], {}
    errors = {
        "convolution_max_abs_mm": 0.0,
        "harmonic_prediction_max_abs_mm": 0.0,
        "designer_max_abs": 0.0,
        "response_max_abs": 0.0,
    }
    for taps in TAPS:
        h = lowpass(taps, CUTOFF, FS)
        delay = (taps - 1) // 2
        x, y, aligned = aligned_filtered(h)
        response = frequency_response(h, FREQUENCIES, FS)
        centered = response * np.exp(2j * np.pi * FREQUENCIES * delay / FS)
        predicted = np.real(
            np.exp(2j * np.pi * np.outer(times, FREQUENCIES)) @ (AMPLITUDES * centered)
        )
        _, independent = freqz(h, worN=FREQUENCIES, fs=FS)
        measures = [
            np.max(abs(y - lfilter(h, [1.0], x)[::FACTOR])),
            np.max(abs(aligned - predicted)),
            np.max(abs(h - firwin(taps, CUTOFF, window="blackman", fs=FS))),
            np.max(abs(response - independent)),
        ]
        errors = {
            key: max(errors[key], float(val)) for key, val in zip(errors, measures, strict=True)
        }
        gain = centered.real
        amplitude = measure_amplitude(times, aligned)
        record = {
            "taps": taps,
            "delay_ms": 1000 * delay / FS,
            "gain_23": float(gain[0]),
            "gain_71": float(gain[1]),
            "gain_185": float(gain[2]),
            "amplitude_71_mm": amplitude,
            "relative_amplitude_error": abs(amplitude - 0.35) / 0.35,
            "lowband_rmse_mm": float(np.sqrt(np.mean((aligned - target) ** 2))),
            "alias_bound_mm": float(0.6 * abs(gain[2])),
            "passband_bias_bound_mm": float(0.35 * abs(gain[1] - 1)),
        }
        records.append(record)
        filtered[taps] = aligned
    phase_rows = []
    h = lowpass(257, CUTOFF, FS)
    gain = np.array([records[2]["gain_71"], records[2]["gain_185"]])
    phase_error = 0.0
    for phase in np.linspace(0, 2 * np.pi, 65):
        _, _, aligned = aligned_filtered(h, phase)
        measured = measure_amplitude(times, aligned)
        raw_amplitude = measure_amplitude(times, modal_signal(times, phase))
        prediction = abs(0.35 * gain[0] + 0.6 * gain[1] * np.exp(-1j * phase))
        phase_error = max(
            phase_error,
            abs(measured - prediction),
            abs(raw_amplitude - abs(0.35 + 0.6 * np.exp(-1j * phase))),
        )
        phase_rows.append([float(phase), raw_amplitude, measured, float(prediction)])
    errors["phase_amplitude_prediction_max_abs_mm"] = float(phase_error)
    metrics = {
        "setup": {
            "sample_rate_hz": FS,
            "factor": FACTOR,
            "output_rate_hz": FS / FACTOR,
            "output_samples": COUNT,
            "duration_s": COUNT / (FS / FACTOR),
            "frequencies_hz": FREQUENCIES.tolist(),
            "amplitudes_mm": AMPLITUDES.tolist(),
            "cutoff_hz": CUTOFF,
            "phase_samples": 65,
        },
        "raw_amplitude_71_mm": measure_amplitude(times, raw),
        "raw_lowband_rmse_mm": float(np.sqrt(np.mean((raw - target) ** 2))),
        "filters": records,
        "verification": errors,
        "phase_sweep": {
            "raw_min_mm": min(row[1] for row in phase_rows),
            "raw_max_mm": max(row[1] for row in phase_rows),
            "filtered_min_mm": min(row[2] for row in phase_rows),
            "filtered_max_mm": max(row[2] for row in phase_rows),
        },
        "physical_validation": False,
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", newline="\n")
    csv_write(
        output / "filter_sweep.csv", list(records[0]), [list(r.values()) for r in records]
    )
    csv_write(
        output / "phase_sweep.csv",
        ["phase_rad", "raw_mm", "filtered_mm", "predicted_mm"],
        phase_rows,
    )
    csv_write(
        output / "timeseries.csv",
        ["time_s", "target_mm", "raw_mm", "filtered_257_mm"],
        zip(times, target, raw, filtered[257], strict=True),
    )
    if plots:
        make_figure(output, records, np.array(phase_rows))
    return metrics


def make_figure(output, records, phases):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": 180,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), layout="constrained")
    grid = np.linspace(0, 256, 2049)
    for record in records:
        h = lowpass(record["taps"], CUTOFF, FS)
        gain = abs(frequency_response(h, grid, FS))
        axes[0].plot(
            grid, 20 * np.log10(np.maximum(gain, 1e-7)), label=f"{record['taps']} taps"
        )
    axes[0].axvline(128, color="0.4", linestyle=":", label="Output Nyquist")
    axes[0].set(
        xlabel="Input frequency (Hz)",
        ylabel="Amplitude gain (dB)",
        xlim=(0, 256),
        ylim=(-120, 3),
        title="(a) Finite-filter response",
    )
    axes[0].legend(fontsize=8, loc="lower left", ncol=2)
    axes[1].plot(
        phases[:, 0] / np.pi, phases[:, 1], label="Unfiltered samples", color="#c5503c"
    )
    axes[1].plot(phases[:, 0] / np.pi, phases[:, 2], label="257-tap FIR", color="#237e8a")
    axes[1].axhline(0.35, color="0.25", linestyle=":", label="True 71 Hz amplitude")
    axes[1].set(
        xlabel="185 Hz phase / pi",
        ylabel="Measured 71 Hz amplitude (mm)",
        title="(b) Aliased modes interfere",
        xlim=(0, 2),
        ylim=(0.2, 1.02),
    )
    axes[1].legend(fontsize=8, loc="upper center")
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.savefig(output / "alias_budget.pdf", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(output / "alias_budget.png")
    plt.close(fig)
