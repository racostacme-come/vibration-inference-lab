"""Regenerate all numerical artifacts and compare every recorded value."""

import csv
import json
import tempfile
from pathlib import Path

import numpy as np

from jointguard.study import write_results

ROOT = Path(__file__).resolve().parents[1]


def compare(expected, actual, key="root"):
    if isinstance(expected, dict):
        assert expected.keys() == actual.keys(), key
        for k in expected:
            compare(expected[k], actual[k], f"{key}.{k}")
    elif isinstance(expected, list):
        assert len(expected) == len(actual), key
        for i, (a, b) in enumerate(zip(expected, actual, strict=True)):
            compare(a, b, f"{key}[{i}]")
    elif isinstance(expected, (float, int)):
        np.testing.assert_allclose(actual, expected, rtol=2e-9, atol=2e-12, err_msg=key)
    else:
        assert expected == actual, key


def main():
    (ROOT / "build").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=ROOT / "build") as temp:
        output = Path(temp)
        actual = write_results(output)
        compare(json.loads((ROOT / "results/metrics.json").read_text()), actual)
        count = 0
        for name in ["population.csv", "correlation.csv", "replicates.csv"]:
            with (ROOT / "results" / name).open() as a, (output / name).open() as b:
                expected, generated = list(csv.DictReader(a)), list(csv.DictReader(b))
            assert len(expected) == len(generated)
            for old, new in zip(expected, generated, strict=True):
                for key in old:
                    if key == "model":
                        assert old[key] == new[key]
                    else:
                        np.testing.assert_allclose(
                            float(new[key]), float(old[key]), rtol=2e-9, atol=2e-12
                        )
            count += len(expected)
        # Rounded diagnostic errors may legitimately differ across BLAS platforms;
        # table (substantive results) must still match exactly.
        assert (output / "table.tex").read_text() == (ROOT / "results/table.tex").read_text()
        v = actual["validation"]
        assert max(v.values()) < 1e-12
        for row in actual["population"]:
            assert row["energy_quad_error_estimate"] < 1e-8
            finite = actual["finite_ensemble"][row["model"]]
            assert abs(finite["energy_mean"] - row["energy"]) < 4 * finite["energy_replicate_se"]
        assert actual["risk_ratio_independent_to_truth"] > 18
        print(f"All JSON values and {count} CSV rows reproduced; analytical checks passed.")


if __name__ == "__main__":
    main()
