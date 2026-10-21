import subprocess
import sys


def test_cli_rejects_invalid_trials():
    result = subprocess.run(
        [sys.executable, "-m", "gapsmooth.cli", "--trials", "1"], capture_output=True
    )
    assert result.returncode == 2
    assert b"must be >=2" in result.stderr
