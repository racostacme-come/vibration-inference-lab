"""Recompute and compare every JSON/CSV sample, with floating-point tolerance."""

import csv
import json
import tempfile
from pathlib import Path

import numpy as np

from loadcoherent.study import run_study


def compare(actual, expected, path="root"):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys(), path
        for key in expected:
            compare(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, list):
        assert len(actual) == len(expected), path
        for i, value in enumerate(expected):
            compare(actual[i], value, f"{path}[{i}]")
    elif isinstance(expected, (float, int)):
        np.testing.assert_allclose(actual, expected, rtol=2e-8, atol=2e-11, err_msg=path)
    else:
        assert actual == expected, path


def main():
    root = Path(__file__).resolve().parents[1]
    samples = root / "examples/results"
    # Temporary reproduction stays within this repository.
    (root / "build").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root / "build") as folder:
        data = run_study(folder, plots=False)
        compare(data, json.loads((samples / "metrics.json").read_text()))
        for name in ("calibration.csv", "forecast_path.csv"):
            with (samples / name).open(newline="") as f:
                expected = list(csv.reader(f))
            with (Path(folder) / name).open(newline="") as f:
                actual = list(csv.reader(f))
            assert actual[0] == expected[0]
            assert len(actual) == len(expected)
            for i, (a, e) in enumerate(zip(actual[1:], expected[1:], strict=True)):
                assert len(a) == len(e)
                for av, ev in zip(a, e, strict=True):
                    try:
                        ef = float(ev)
                    except ValueError:
                        assert av == ev
                    else:
                        compare(float(av), ef, f"{name}:{i}")
    print("Every recorded JSON and CSV value reproduced within tolerance.")


if __name__ == "__main__":
    main()
