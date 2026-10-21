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
    v, phases = data["verification"], data["phase_sweep"]
    commands = {
        "ConvError": sci(v["convolution_max_abs_mm"]),
        "HarmonicError": sci(v["harmonic_prediction_max_abs_mm"]),
        "DesignError": sci(v["designer_max_abs"]),
        "ResponseError": sci(v["response_max_abs"]),
        "PhaseError": sci(v["phase_amplitude_prediction_max_abs_mm"]),
        "RawAmplitude": f"{data['raw_amplitude_71_mm']:.5f}",
        "RawRMSE": f"{data['raw_lowband_rmse_mm']:.5f}",
        "FilteredMin": f"{phases['filtered_min_mm']:.8f}",
        "FilteredMax": f"{phases['filtered_max_mm']:.8f}",
        "FinePercent": f"{100 * data['filters'][2]['relative_amplitude_error']:.5f}",
        "FineAlias": sci(data["filters"][2]["alias_bound_mm"]),
        "FineBias": sci(data["filters"][2]["passband_bias_bound_mm"]),
    }
    rows = []
    for r in data["filters"]:
        rows.append(
            f"{r['taps']} & {r['delay_ms']:.3f} & {r['gain_71']:.7f} & "
            f"${sci(r['gain_185'])}$ & {r['amplitude_71_mm']:.7f} & "
            f"${sci(r['lowband_rmse_mm'])}$ " + r"\\"
        )
    commands["FilterRows"] = "\n".join(rows)
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
    inputs += [
        root / "CMakeLists.txt",
        root / "README.md",
        root / ".clang-format",
        root / ".github/workflows/validate.yml",
    ]
    inputs += sorted((root / "cpp").glob("*"))
    inputs += sorted((root / "docs").glob("*.md"))
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
