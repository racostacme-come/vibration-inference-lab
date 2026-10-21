"""Recompute every recorded numeric result and enforce analytical validation."""

import csv
import json
import tempfile
from pathlib import Path

import numpy as np

from quantflex.study import run

ROOT = Path(__file__).resolve().parents[1]


def compare(expected, actual):
    if isinstance(expected, dict):
        assert expected.keys() == actual.keys()
        for key in expected:
            compare(expected[key], actual[key])
    elif isinstance(expected, bool):
        assert expected is actual
    else:
        # The near-zero oracle residual varies with BLAS; absolute tolerance is
        # only for implementation roundoff, not for the physical-scale metrics.
        np.testing.assert_allclose(actual, expected, rtol=2e-7, atol=5e-17)


def main():
    build = ROOT / "build"
    build.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=build) as tmp:
        output = Path(tmp)
        metrics = run(output)
        compare(json.loads((ROOT / "results/metrics.json").read_text()), metrics)
        count = 0
        for filename in ["study.csv", "history.csv"]:
            with (ROOT / "results" / filename).open(newline="") as f:
                expected = list(csv.DictReader(f))
            with (output / filename).open(newline="") as f:
                actual = list(csv.DictReader(f))
            assert len(expected) == len(actual)
            for a, b in zip(expected, actual, strict=True):
                compare({k: float(v) for k, v in a.items()}, {k: float(v) for k, v in b.items()})
                count += 1
        assert (ROOT / "results/table.tex").read_text() == (output / "table.tex").read_text()
        # Formatted oracle roundoff is platform dependent; other paper macros
        # are checked numerically above and the compiled paper inputs are hashed.
        for key in [
            "max_unclipped_coefficient_z",
            "max_unclipped_mse_abs_z",
            "max_unclipped_energy_abs_z",
        ]:
            assert metrics[key] < 5, (key, metrics[key])
        assert metrics["qr_oracle_max_error_m"] < 1e-15
        assert 0.4 < metrics["clipped_fraction"] < 0.5
        assert metrics["clipped_corrected_energy_ratio"] < 0.5
        print(f"All metrics and {count} CSV rows reproduced; analytical checks passed.")


if __name__ == "__main__":
    main()
