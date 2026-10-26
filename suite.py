"""Common entry point for the integrated research suite (Python 3.11+)."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
MANIFEST = json.loads((ROOT / "suite.json").read_text(encoding="utf-8"))

def run(command, cwd=ROOT):
    subprocess.run(command, cwd=cwd, check=True)

def verify():
    count = 0
    for module in MANIFEST["modules"]:
        base = ROOT / "modules" / module["id"]
        for name, expected in module["sha256"].items():
            actual = hashlib.sha256((base / name).read_bytes()).hexdigest()
            if actual != expected:
                raise RuntimeError(f"Imported source changed: {module['id']}/{name}")
            count += 1
    print(f"Verified {count} imported files against the integration manifest.")

def install():
    run([sys.executable, "-m", "pip", "install", "pytest>=8,<10", "pytest-cov>=6,<8", "pypdf>=5,<7"])
    for module in MANIFEST["modules"]:
        run([sys.executable, "-m", "pip", "install", str(ROOT / "modules" / module["id"])])

def test():
    reports = []
    for module in MANIFEST["modules"]:
        started = time.monotonic()
        result = subprocess.run([sys.executable, "-m", "pytest", "tests", "-q"],
                                cwd=ROOT / "modules" / module["id"], text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        print(result.stdout)
        reports.append({"module": module["id"], "returncode": result.returncode,
                        "seconds": time.monotonic()-started, "output": result.stdout})
    result = subprocess.run([sys.executable, "-m", "pytest", "tests", "-q"],
                            cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(result.stdout)
    reports.append({"module": "integration", "returncode": result.returncode, "output": result.stdout})
    (ROOT / "validation").mkdir(exist_ok=True)
    (ROOT / "validation/tests.json").write_text(json.dumps(reports, indent=2)+"\n", encoding="utf-8", newline="\n")
    if any(r["returncode"] for r in reports):
        raise SystemExit("One or more test suites failed; inspect validation/tests.json")

def experiment():
    from integration import experiment
    data = experiment()
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results/integration.json").write_text(json.dumps(data, indent=2, allow_nan=False)+"\n", encoding="utf-8", newline="\n")
    print(json.dumps(data, indent=2))
    from paper_table import write_table
    write_table(data, ROOT / "results/table.tex")
    from verification import experiment as independent, write_table as independent_table
    reference_data = independent()
    (ROOT / "results/independent.json").write_text(
        json.dumps(reference_data, indent=2, allow_nan=False)+"\n", encoding="utf-8", newline="\n")
    independent_table(reference_data, ROOT / "results/independent-table.tex")
    print(json.dumps(reference_data, indent=2))

def native(build_root=None):
    """Build three preserved kernel tests and the original oscillator handoff."""
    build = Path(build_root).absolute() if build_root else ROOT / "build" / "native"
    commands = [
        ["cmake", "-S", str(ROOT), "-B", str(build), "-DBUILD_TESTING=ON", "-DCMAKE_BUILD_TYPE=Release"],
        ["cmake", "--build", str(build), "--config", "Release"],
        ["ctest", "--test-dir", str(build), "-C", "Release", "--verbose", "--output-on-failure", "--no-tests=error"],
    ]
    reports = []
    for command in commands:
        result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        print(result.stdout)
        reports.append({"command": command, "returncode": result.returncode, "output": result.stdout})
        if result.returncode: break
    (ROOT / "validation").mkdir(exist_ok=True)
    (ROOT / "validation/native.json").write_text(json.dumps(reports, indent=2)+"\n", encoding="utf-8", newline="\n")
    if any(r["returncode"] for r in reports):
        raise SystemExit("Native validation failed; inspect validation/native.json")


def paper(engine):
    from pypdf import PdfReader
    import shutil
    engine = engine or shutil.which("tectonic") or shutil.which("pdflatex")
    if not engine:
        raise SystemExit("Install Tectonic or pdfLaTeX, then pass --engine if needed.")
    command = ([engine, "--keep-logs", "paper.tex"] if "tectonic" in Path(engine).name.lower()
               else [engine, "-interaction=nonstopmode", "-halt-on-error", "paper.tex"])
    for _ in range(1 if "tectonic" in Path(engine).name.lower() else 2):
        run(command, ROOT / "paper")
    log = (ROOT / "paper/paper.log").read_text(encoding="utf-8", errors="replace")
    for marker in ("Overfull", "undefined references", "undefined citations", "LaTeX Error", "Rerun to get"):
        if marker in log:
            raise SystemExit("Paper diagnostic: "+marker)
    pages = len(PdfReader(ROOT / "paper/paper.pdf").pages)
    if not 1 <= pages <= 5:
        raise SystemExit(f"Expected 1-5 pages, got {pages}")
    inputs = ["paper/paper.tex", "results/integration.json", "results/table.tex", "integration.py", "suite.json", "verification.py", "results/independent.json", "results/independent-table.tex", "paper/paper.pdf"]
    report = {"pages": pages, "sha256": {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in inputs}}
    (ROOT/"paper/build-report.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8", newline="\n")
    print(f"Compiled and checked {pages} pages.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["verify", "install", "test", "native", "experiment", "paper"])
    parser.add_argument("--engine")
    parser.add_argument("--build-root", help="Short native build path on Windows")
    args = parser.parse_args()
    if args.command == "paper":
        paper(args.engine)
    elif args.command == "native":
        native(args.build_root)
    else:
        globals()[args.command]()
