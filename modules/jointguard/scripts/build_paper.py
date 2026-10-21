"""Compile LaTeX, reject unresolved diagnostics, enforce <=5 pages and hash inputs."""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
INPUTS = [
    "paper/paper.tex",
    "results/numbers.tex",
    "results/table.tex",
    "results/joint-study.png",
    "results/metrics.json",
    "results/population.csv",
    "results/correlation.csv",
    "results/replicates.csv",
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", default=None, help="Path to tectonic or pdflatex")
    parser.add_argument(
        "--verify", action="store_true", help="Verify recorded hashes without compiling"
    )
    args = parser.parse_args()
    report_path = ROOT / "paper/build-report.json"
    if args.verify:
        report = json.loads(report_path.read_text())
        for path, expected in report["sha256"].items():
            assert digest(ROOT / path) == expected, f"Stale paper input: {path}"
        assert len(PdfReader(ROOT / "paper/paper.pdf").pages) == report["pages"] <= 5
        print("Recorded manuscript inputs and PDF verified.")
        return
    engine = args.engine or shutil.which("pdflatex") or shutil.which("tectonic")
    if not engine:
        raise SystemExit("Install TeX Live (pdflatex) or Tectonic, or pass --engine PATH")
    # Preserve the executable name: pdflatex may be a symlink to pdftex, whose
    # argv[0] selects the LaTeX format. Resolving that link selects plain TeX.
    engine = str(Path(engine).absolute())
    if "tectonic" in Path(engine).name.lower():
        commands = [[engine, "--keep-logs", "paper.tex"]]
    else:
        commands = [[engine, "-interaction=nonstopmode", "-halt-on-error", "paper.tex"]] * 2
    for command in commands:
        result = subprocess.run(command, cwd=ROOT / "paper", text=True, capture_output=True)
        if result.returncode:
            raise SystemExit(result.stdout + result.stderr)
    log = (ROOT / "paper/paper.log").read_text(encoding="utf-8", errors="replace")
    forbidden = [
        "undefined references",
        "undefined citations",
        "Overfull",
        "Rerun to get",
        "LaTeX Error",
        "There were undefined",
        "multiply defined",
    ]
    for marker in forbidden:
        if marker in log:
            raise SystemExit(f"Manuscript diagnostic: {marker}\n{log}")
    pages = len(PdfReader(ROOT / "paper/paper.pdf").pages)
    if not 1 <= pages <= 5:
        raise SystemExit(f"Paper has {pages} pages; required 1-5")
    report = {
        "pages": pages,
        "engine": Path(engine).name,
        "sha256": {p: digest(ROOT / p) for p in [*INPUTS, "paper/paper.pdf"]},
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"Compiled {pages} pages; references and layout diagnostics passed.")


if __name__ == "__main__":
    main()
