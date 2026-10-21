"""Generate data-bound tables, compile LaTeX and enforce manuscript quality gates."""

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

from pypdf import PdfReader


def sci(value):
    mantissa, exponent = f"{value:.3e}".split("e")
    return rf"{mantissa}\times10^{{{int(exponent)}}}"


def numbers(root):
    data = json.loads((root / "examples/results/metrics.json").read_text())
    v, g, mc = data["verification"], data["gap_center"], data["monte_carlo"]
    commands = {
        "BatchMean": sci(v["batch_mean_max_abs"]),
        "BatchCov": sci(v["batch_covariance_max_abs"]),
        "BatchNll": sci(v["batch_nll_abs"]),
        "QuadError": sci(v["quadrature_covariance_max_abs"]),
        "StationaryError": sci(v["stationary_covariance_max_abs"]),
        "FilterSd": f"{g['filter_sd']:.5f}",
        "SmoothSd": f"{g['smooth_sd']:.5f}",
        "Reduction": f"{100 * g['variance_reduction_fraction']:.2f}",
        "FilterNees": f"{mc['filter_mean_nees']:.3f}",
        "SmoothNees": f"{mc['smooth_mean_nees']:.3f}",
        "FilterCoverage": f"{100 * mc['filter_displacement_coverage']:.2f}",
        "SmoothCoverage": f"{100 * mc['smooth_displacement_coverage']:.2f}",
        "NeesLow": f"{mc['mean_nees_95_interval'][0]:.3f}",
        "NeesHigh": f"{mc['mean_nees_95_interval'][1]:.3f}",
    }
    rows = []
    for r in data["euler_refinement"]:
        rows.append(
            f"{r['steps']} & {r['dt']:.6f} & "
            f"{r['transition_error_fro']:.5f} & {r['covariance_error_fro']:.5f} " + r"\\"
        )
    commands["EulerRows"] = "\n".join(rows)
    return (
        "\n".join(rf"\newcommand{{\{key}}}{{{value}}}" for key, value in commands.items())
        + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", help="path to tectonic or pdflatex")
    args = parser.parse_args()
    folder = Path(__file__).resolve().parent
    root = folder.parent
    (folder / "numbers.tex").write_text(numbers(root), encoding="utf-8", newline="\n")
    engine = args.engine or shutil.which("tectonic") or shutil.which("pdflatex")
    if not engine:
        raise SystemExit("Install Tectonic or TeX Live; pass --engine if not on PATH")
    if Path(engine).exists():
        engine = str(Path(engine).resolve())
    name = Path(engine).stem.lower()
    build = folder / "build"
    build.mkdir(exist_ok=True)
    if name == "tectonic":
        command = [
            engine,
            "--keep-logs",
            "--keep-intermediates",
            "--outdir",
            str(build),
            "paper.tex",
        ]
        passes = 1
    elif name == "pdflatex":
        command = [
            engine,
            "-no-shell-escape",
            "-halt-on-error",
            "-interaction=nonstopmode",
            "-output-directory",
            str(build),
            "paper.tex",
        ]
        passes = 2
    else:
        raise SystemExit("Supported engines: tectonic, pdflatex")
    for _ in range(passes):
        result = subprocess.run(
            command, cwd=folder, capture_output=True, text=True, errors="replace", timeout=180
        )
        if result.returncode:
            raise SystemExit(result.stdout + result.stderr)
    log = (build / "paper.log").read_text(errors="replace")
    issues = re.findall(
        r"Overfull \\[hv]box[^\n]*|[^\n]*(?:undefined references|"
        r"Citation .* undefined|Reference .* undefined)[^\n]*",
        log,
    )
    if issues:
        raise SystemExit("Layout/reference errors:\n" + "\n".join(issues))
    pdf = build / "paper.pdf"
    pages = PdfReader(pdf).pages
    if not 1 <= len(pages) <= 5:
        raise SystemExit(f"Page limit failed: {len(pages)}")
    if any(not (p.extract_text() or "").strip() for p in pages):
        raise SystemExit("Empty manuscript page")
    inputs = [
        folder / "paper.tex",
        folder / "numbers.tex",
        folder / "build.py",
        root / "pyproject.toml",
    ]
    inputs += sorted((root / "src").rglob("*.py"))
    inputs += sorted((root / "tests").glob("*.py"))
    inputs += sorted((root / "scripts").glob("*.py"))
    inputs += sorted((root / "examples/results").glob("*"))
    report = {
        "pages": len(pages),
        "page_limit": 5,
        "engine": name,
        "overfull_boxes": 0,
        "undefined_references": 0,
        "inputs_sha256": {
            p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in inputs
        },
        "pdf_sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
    }
    shutil.copyfile(pdf, folder / "paper.pdf")
    (folder / "validation.json").write_text(json.dumps(report, indent=2) + "\n", newline="\n")
    print(f"Compiled {len(pages)} pages; no overfull boxes or undefined references.")


if __name__ == "__main__":
    main()
