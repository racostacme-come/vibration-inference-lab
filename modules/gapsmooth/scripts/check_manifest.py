"""Check that committed paper provenance matches numerical and source inputs."""

import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
report = json.loads((root / "paper/validation.json").read_text())
for relative, expected in report["inputs_sha256"].items():
    actual = hashlib.sha256((root / relative).read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f"Manuscript input changed: {relative}; rebuild paper")
actual_pdf = hashlib.sha256((root / "paper/paper.pdf").read_bytes()).hexdigest()
if actual_pdf != report["pdf_sha256"]:
    raise SystemExit("PDF hash mismatch")
print(f"Verified {len(report['inputs_sha256'])} manuscript inputs and PDF hash.")
