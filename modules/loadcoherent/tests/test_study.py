"""Check research invariants without assuming Monte Carlo equals expectation."""

import json
import subprocess
import sys

import numpy as np
import pytest
from numpy.testing import assert_allclose

from loadcoherent.study import district, run_study, thermal_truth


@pytest.fixture(scope="module")
def study(tmp_path_factory):
    return run_study(tmp_path_factory.mktemp("study"), plots=False)


def test_thermal_balance():
    s, _ = district()
    q = thermal_truth(np.arange(48))
    assert_allclose(q[:, 0], q[:, 1] + q[:, 2])
    assert_allclose(q, q[:, 3:] @ s.T)
    assert_allclose(q[0, 3:], 17 * np.array([1.2, 0.8, 1.5, 1]))
    assert np.min(q) > 0


def test_fixed_risk_and_monte_carlo(study):
    methods = study["fixed"]
    assert methods["oracle"]["total_mse"] < methods["bottom_up"]["total_mse"]
    assert methods["wrong_weights"]["total_mse"] > methods["base"]["total_mse"]
    for name, result in methods.items():
        assert abs(result["monte_carlo_z"]) < 5
        assert result["preservation_max"] < 2e-13
        if name != "base":
            assert result["constraint_operator_max"] < 2e-13


def test_independent_validation_and_bias(study):
    assert study["kkt_max_error"] < 2e-12
    assert study["idempotence_max"] < 2e-13
    assert study["path_reconciled_max_imbalance_kw"] < 2e-12
    assert study["coherent_bias_preservation_max"] < 2e-13
    assert_allclose(
        study["oracle_biased_total_mse"] - study["fixed"]["oracle"]["total_mse"], 112
    )


def test_calibration_diagnostics(study):
    records = {(r["samples"], r["method"]): r for r in study["calibration"]}
    assert (
        records[10, "sample"]["mean_risk_ratio"] > records[10, "shrink_half"]["mean_risk_ratio"]
    )
    assert (
        records[200, "sample"]["mean_risk_ratio"]
        < records[200, "shrink_half"]["mean_risk_ratio"]
    )
    for r in records.values():
        assert r["q10"] <= r["median_risk_ratio"] <= r["q90"]
        assert r["q10"] >= study["fixed"]["oracle"]["risk_ratio"] - 1e-12


def test_cli(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "loadcoherent.cli", "--output", str(tmp_path), "--no-plots"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Oracle/base" in result.stdout
    assert json.loads((tmp_path / "metrics.json").read_text())["seed"] == 20261008
    assert len((tmp_path / "calibration.csv").read_text().splitlines()) == 961
    assert len((tmp_path / "forecast_path.csv").read_text().splitlines()) == 337
    assert b"\r\n" not in (tmp_path / "calibration.csv").read_bytes()


@pytest.mark.parametrize("hours", [[[1, 2]], [np.inf]])
def test_invalid_thermal_hours(hours):
    with pytest.raises(ValueError):
        thermal_truth(hours)
