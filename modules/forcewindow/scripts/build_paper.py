"""Compile the manuscript, reject LaTeX defects, and enforce the five-page cap."""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from pypdf import PdfReader

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--engine", default="pdflatex", help="pdflatex or a path to tectonic")
args = parser.parse_args()
paper = root / "paper"
build = paper / "build"
build.mkdir(parents=True, exist_ok=True)
if "tectonic" in Path(args.engine).stem:
    command = [
        args.engine,
        "--keep-logs",
        "--keep-intermediates",
        "--outdir",
        str(build),
        str(paper / "paper.tex"),
    ]
    subprocess.run(command, cwd=paper, check=True)
else:
    for _ in range(2):
        subprocess.run(
            [
                args.engine,
                "-interaction=nonstopmode",
                "-halt-on-error",
                f"-output-directory={build}",
                "paper.tex",
            ],
            cwd=paper,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
log = (build / "paper.log").read_text(encoding="utf-8", errors="replace")
for defect in (
    "Undefined control sequence",
    "undefined references",
    "undefined citations",
    "Overfull",
    "LaTeX Error",
):
    if defect.lower() in log.lower():
        raise RuntimeError(f"LaTeX defect: {defect}")
if "Rerun to get cross-references right" in log:
    raise RuntimeError("unresolved cross references")
pdf = build / "paper.pdf"
pages = len(PdfReader(pdf).pages)
if not 1 <= pages <= 5:
    raise RuntimeError(f"Manuscript has {pages} pages, cap is five")
shutil.copyfile(pdf, paper / "paper.pdf")
files = [
    paper / "paper.tex",
    paper / "paper.pdf",
    root / "examples/results/metrics.json",
    root / "examples/results/window-study.pdf",
    root / "src/forcewindow/study.py",
    root / "src/forcewindow/model.py",
    root / "cpp/state.hpp",
]
report = {
    "pages": pages,
    "engine": Path(args.engine).name,
    "checks": "compiled; no undefined references/citations or overfull boxes",
    "sha256": {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in files
    },
}
(paper / "validation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2))
