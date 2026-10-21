"""Verify the paper's recorded source/data hashes and the delivered PDF hash."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
report = json.loads((ROOT / "paper/validation.json").read_text())
for name, digest in report["inputs_sha256"].items():
    assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
assert (
    hashlib.sha256((ROOT / "paper/paper.pdf").read_bytes()).hexdigest() == report["pdf_sha256"]
)
assert 1 <= report["pages"] <= 5
print(f"Verified {len(report['inputs_sha256'])} input hashes and PDF hash.")
