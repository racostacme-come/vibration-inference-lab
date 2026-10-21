"""Numerically reproduce all recorded tables; graphics are rebuilt, not byte-compared."""

import csv
import json
from pathlib import Path

import numpy as np

from forcewindow.study import run

root = Path(__file__).resolve().parents[1]
actual = run(root / "build/reproduced")
expected = json.loads((root / "examples/results/metrics.json").read_text())


def compare(a, b, path=""):
    if isinstance(b, dict):
        assert a.keys() == b.keys(), path
        for key in b:
            if key not in {"iterations", "normal_residual_l2", "max_lsqr_svd_difference_N"}:
                compare(a[key], b[key], path + "/" + key)
    elif isinstance(b, list):
        assert len(a) == len(b)
        for x, y in zip(a, b, strict=True):
            compare(x, y, path)
    else:
        np.testing.assert_allclose(a, b, rtol=2e-6, atol=1e-8, err_msg=path)


compare(actual, expected)
assert actual["max_lsqr_svd_difference_N"] < 1e-7
assert actual["constant_load_max_error_m"] < 1e-13
assert all(x["normal_residual_l2"] < 1e-12 for x in actual["windows"])
for name in ("reconstruction", "regularization"):
    with (root / f"examples/results/{name}.csv").open() as f:
        expected_rows = list(csv.DictReader(f))
    with (root / f"build/reproduced/{name}.csv").open() as f:
        actual_rows = list(csv.DictReader(f))
    assert len(actual_rows) == len(expected_rows)
    for a, b in zip(actual_rows, expected_rows, strict=True):
        assert a.keys() == b.keys()
        for key in b:
            np.testing.assert_allclose(float(a[key]), float(b[key]), rtol=2e-6, atol=1e-8)
print("All JSON metrics, 480 reconstruction rows, and 164 risk rows reproduced.")
