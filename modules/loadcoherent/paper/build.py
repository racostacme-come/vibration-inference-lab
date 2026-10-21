"""Bind manuscript claims to results, compile, and enforce page/layout checks."""

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

from pypdf import PdfReader


def sci(x):
    a, b = f"{x:.3e}".split("e")
    return rf"{a}\times10^{{{int(b)}}}"


def numbers(root):
    data = json.loads((root / "examples/results/metrics.json").read_text())
    fixed = data["fixed"]
    commands = {
        "OracleReduction": f"{100 * (1 - fixed['oracle']['risk_ratio']):.2f}",
        "WrongIncrease": f"{100 * (fixed['wrong_weights']['risk_ratio'] - 1):.2f}",
        "KKTError": sci(data["kkt_max_error"]),
        "Idempotence": sci(data["idempotence_max"]),
        "BeforeBalance": f"{data['path_base_max_imbalance_kw']:.4f}",
        "AfterBalance": sci(data["path_reconciled_max_imbalance_kw"]),
        "MaxMCZ": f"{max(abs(v['monte_carlo_z']) for v in fixed.values()):.3f}",
        "BiasPenalty": f"{data['oracle_bias_penalty']:.0f}",
        "BiasedRisk": f"{data['oracle_biased_total_mse']:.4f}",
    }
    labels = {
        "base": "Base",
        "bottom_up": "Bottom up",
        "ols": "OLS",
        "oracle": "Oracle",
        "wrong_weights": "Wrong weights",
    }
    commands["FixedRows"] = "\n".join(
        f"{labels[k]} & {v['total_mse']:.4f} & {v['risk_ratio']:.6f} & "
        f"{v['monte_carlo_mse']:.4f} & {v['monte_carlo_z']:.3f} " + r"\\"
        for k, v in fixed.items()
    )
    labels = {"sample": "Sample", "shrink_half": "Half shrinkage", "diagonal": "Diagonal"}
    commands["CalibrationRows"] = "\n".join(
        f"{r['samples']} & {labels[r['method']]} & {r['mean_risk_ratio']:.4f} & "
        f"{r['mean_standard_error']:.4f} & {r['q10']:.4f}--{r['q90']:.4f} & "
        f"{100 * r['fraction_worse_than_base']:.2f}\\% " + r"\\"
        for r in data["calibration"]
    )
    return "\n".join(rf"\newcommand{{\{k}}}{{{v}}}" for k, v in commands.items()) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", help="path to tectonic or pdflatex")
    args = parser.parse_args()
    folder = Path(__file__).resolve().parent
    root = folder.parent
    (folder / "numbers.tex").write_text(numbers(root), encoding="utf-8", newline="\n")
    engine = args.engine or shutil.which("tectonic") or shutil.which("pdflatex")
    if not engine:
        raise SystemExit("Install Tectonic or TeX Live and pass --engine if necessary")
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
    if not 1 <= len(pages) <= 5 or any(not (p.extract_text() or "").strip() for p in pages):
        raise SystemExit(f"Page count/empty-page check failed: {len(pages)}")
    inputs = [
        folder / "paper.tex",
        folder / "numbers.tex",
        folder / "build.py",
        root / "pyproject.toml",
        root / "README.md",
        root / "MANIFEST.in",
        root / "requirements-reproduced.txt",
        folder / "README.md",
        root / ".github/workflows/validate.yml",
    ]
    for directory, glob in [
        ("src", "*.py"),
        ("tests", "*.py"),
        ("scripts", "*.py"),
        ("docs", "*.md"),
        ("examples/results", "*"),
    ]:
        inputs += sorted((root / directory).rglob(glob))
    report = {
        "pages": len(pages),
        "page_limit": 5,
        "engine": name,
        "overfull_boxes": 0,
        "undefined_references": 0,
        "inputs_sha256": {
            p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in inputs
            if p.is_file()
        },
        "pdf_sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
    }
    shutil.copyfile(pdf, folder / "paper.pdf")
    (folder / "validation.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"Compiled {len(pages)} pages; no undefined references or overfull boxes.")


if __name__ == "__main__":
    main()
