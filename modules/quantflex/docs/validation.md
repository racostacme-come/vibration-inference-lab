# Executed validation

Local validation on 21 October 2026: Windows, Python 3.14.5, NumPy 2.5.4,
Matplotlib 3.11.2, pytest 9.1.1, Ruff 0.17.1 and Tectonic 0.17.0.

- 41 tests passed from source and the installed wheel.
- Ruff lint and format checks passed.
- All JSON values and all 174 CSV rows were recomputed and compared. The generated
  table also matches. Analytical mean, MSE and energy checks passed.
- Wheel and source archive built successfully. Installed-wheel import location,
  tests, complete reproduction and quickstart were checked.
- Four-page manuscript compiled without undefined references or overfull boxes.
  Every local page was rendered and inspected. Crowded logarithmic axis ticks were
  replaced with explicit sample-count ticks before the final review.
- Numerical code, results, manuscript inputs and PDF are hashed in
  `paper/build-report.json`; `--verify` checks for stale inputs.

Resolved during development: the first editable install preceded creation of the
source directory and failed; the later install succeeded. Initial LaTeX table
input caused a misplaced bottom-rule error; generating the complete tabular
block fixed it. Initial lint line lengths/import order were corrected. These
failures were not treated as passing checks.

The committed workflow runs two software platforms and an independent pdfLaTeX
build. Remote results are available in the repository's Actions history, tied to
actual commits. This record describes local checks; final remote run identifiers
and manuscript review are retained in the series index.

Limitations: one synthetic known-frequency linear oscillator. No experimental
physical validation, continuum convergence study, hardware implementation,
performance benchmark or claim of universal improvement from dither.
