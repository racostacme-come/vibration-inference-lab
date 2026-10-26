"""Reject incomplete or stale manuscript evidence without rebuilding the PDF.

An external CI report/PDF may be checked against this checkout. Hash equality
binds bytes; it does not establish numerical correctness or visual quality.
"""
import argparse
import hashlib
import json
from pathlib import Path

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent
INPUTS = (
    "paper/paper.tex", "results/integration.json", "results/table.tex",
    "integration.py", "suite.json", "verification.py",
    "results/independent.json", "results/independent-table.tex", "paper/paper.pdf",
)


def verify(root=ROOT, report_path=None, pdf_path=None):
    root = Path(root)
    report_path = Path(report_path) if report_path else root / "paper/build-report.json"
    pdf_path = Path(pdf_path) if pdf_path else root / "paper/paper.pdf"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if set(report.get("sha256", {})) != set(INPUTS):
        raise ValueError("Manuscript manifest must contain exactly the required nine files")
    for name in INPUTS:
        path = pdf_path if name == "paper/paper.pdf" else root / name
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != report["sha256"][name]:
            raise ValueError(f"Stale manuscript evidence: {name}")
    pages = len(PdfReader(pdf_path).pages)
    if not 1 <= pages <= 5 or report.get("pages") != pages:
        raise ValueError(f"Manuscript page count mismatch or outside 1-5: {pages}")
    return pages


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--pdf", type=Path)
    args = parser.parse_args()
    print(f"Verified all nine manuscript hashes and {verify(report_path=args.report, pdf_path=args.pdf)} pages.")
