"""Reproduce numerical outputs; tolerate platform rounding, never byte-compare plots."""

import json
from pathlib import Path

import numpy as np

from gapsmooth.experiments import run_study

root = Path(__file__).resolve().parents[1]
expected = json.loads((root / "examples/results/metrics.json").read_text())
actual = run_study(root / "reproduced")


def compare(a, b):
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            compare(a[key], b[key])
    elif isinstance(a, list):
        assert len(a) == len(b)
        for x, y in zip(a, b, strict=True):
            compare(x, y)
    else:
        np.testing.assert_allclose(a, b, rtol=1e-7, atol=2e-12)


compare(expected, actual)
trace = np.genfromtxt(root / "examples/results/trace.csv", delimiter=",", skip_header=1)
reproduced = np.genfromtxt(root / "reproduced/trace.csv", delimiter=",", skip_header=1)
np.testing.assert_allclose(trace, reproduced, rtol=1e-7, atol=2e-12, equal_nan=True)
assert max(actual["verification"].values()) < 1e-10
lo, hi = actual["monte_carlo"]["mean_nees_95_interval"]
for method in ["filter", "smooth"]:
    assert lo < actual["monte_carlo"][f"{method}_mean_nees"] < hi
assert 0 < actual["gap_center"]["variance_reduction_fraction"] < 1
print("Recorded outputs reproduced; deterministic and fixed-time calibration checks passed.")
