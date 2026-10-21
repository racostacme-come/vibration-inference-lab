"""Recompute the complete numerical experiment and compare committed outputs."""

import json
import tempfile
from pathlib import Path

import numpy as np

from aliasguard.study import run_study

ROOT = Path(__file__).resolve().parents[1]


def compare(reference, actual, path="root"):
    if isinstance(reference, dict):
        assert reference.keys() == actual.keys(), path
        for key in reference:
            compare(reference[key], actual[key], path + "." + key)
    elif isinstance(reference, list):
        assert len(reference) == len(actual), path
        for index, (left, right) in enumerate(zip(reference, actual, strict=True)):
            compare(left, right, f"{path}[{index}]")
    elif isinstance(reference, (float, int)) and not isinstance(reference, bool):
        np.testing.assert_allclose(actual, reference, rtol=2e-7, atol=5e-12, err_msg=path)
    else:
        assert reference == actual, path


def main():
    reference = ROOT / "examples/results"
    with tempfile.TemporaryDirectory(prefix="reproduce-", dir=ROOT / "build") as name:
        output = Path(name)
        actual = run_study(output, plots=False)
        compare(json.loads((reference / "metrics.json").read_text()), actual)
        for filename in ("filter_sweep.csv", "phase_sweep.csv", "timeseries.csv"):
            expected = np.loadtxt(reference / filename, delimiter=",", skiprows=1)
            measured = np.loadtxt(output / filename, delimiter=",", skiprows=1)
            np.testing.assert_allclose(measured, expected, rtol=2e-7, atol=5e-12)
    print("Full JSON and all three CSV files reproduced (rtol=2e-7, atol=5e-12).")


if __name__ == "__main__":
    (ROOT / "build").mkdir(exist_ok=True)
    main()
