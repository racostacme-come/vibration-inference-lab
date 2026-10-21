"""Check delivered manuscript hashes against the current source and results."""

import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    report = json.loads((root / "paper/validation.json").read_text())
    entries = {**report["inputs_sha256"], "paper/paper.pdf": report["pdf_sha256"]}
    for name, expected in entries.items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
    print(f"Verified {len(entries)} manuscript input/PDF hashes.")


if __name__ == "__main__":
    main()
