import numpy as np
from numpy.testing import assert_allclose

from aliasguard.study import run_study


def test_analytical_study_and_two_part_error_budget(tmp_path):
    result = run_study(tmp_path, plots=False)
    assert_allclose(result["raw_amplitude_71_mm"], 0.95, atol=1e-12)
    assert_allclose(result["raw_lowband_rmse_mm"], 0.6 / np.sqrt(2), atol=1e-12)
    assert_allclose(result["phase_sweep"]["raw_min_mm"], 0.25, atol=1e-12)
    for value in result["verification"].values():
        assert value < 1e-11
    for record in result["filters"]:
        error = abs(record["amplitude_71_mm"] - 0.35)
        bound = record["alias_bound_mm"] + record["passband_bias_bound_mm"]
        assert error <= bound + 1e-12
    fine = result["filters"][2]
    assert fine["relative_amplitude_error"] < 2e-4
    # Increasing length need not monotonically reduce gain at one stopband frequency.
    assert abs(result["filters"][1]["gain_185"]) < abs(fine["gain_185"])


def test_cli_writes_portable_outputs(tmp_path):
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "aliasguard.cli", "--output", str(tmp_path), "--no-plots"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert '"physical_validation": false' in result.stdout
    assert (tmp_path / "metrics.json").is_file()
    assert (tmp_path / "timeseries.csv").is_file()
