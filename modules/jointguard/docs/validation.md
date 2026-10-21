# Validation record

Local checks on 20 October 2026 (Windows, Python 3.14.5, MSVC 19.44):

- 34 Python tests passed, including independent SciPy distance/density oracles,
  analytical Gaussian norm identities, mechanical equilibrium, invariances,
  invalid inputs and overflow rejection.
- Native CMake Release build and CTest passed with compiler warnings treated as
  errors for project sources. An initial local configure attempt passed quoted
  `pybind11 --cmakedir` output as a literal path; using `get_cmake_dir()` fixed
  it. Earlier checks attempted before dependency installation finished were
  rerun successfully. These were setup failures, not numerical failures.
- Ruff lint and format checks and clang-format checks passed.
- All JSON values and 164 CSV rows reproduced. Numerical results are in
  `results/metrics.json`; those files are generated directly by the study.
- Tectonic compiled a four-page manuscript with no unresolved references or
  overfull boxes. Every page was rendered with Poppler at 1500-pixel maximum
  dimension and visually inspected. Text, equations, table, figure and references
  are readable; no clipping or overlap was observed. Two initially overfull
  command paragraphs were corrected before this successful build.
- Source/artifact/PDF hashes pass `scripts/build_paper.py --verify`.

The GitHub Actions workflow independently builds and tests Linux and Windows
software, checks native address/undefined-behavior sanitizers on Linux, builds
sdist/wheel packages, tests the installed wheel, and compiles/renders the
manuscript using pdfLaTeX. Actual remote outcomes are available in the repository's
Actions history; workflow configuration alone is not evidence of a passing run.

All published metrics apply to the declared synthetic experiment. No measured
physical validation, continuum convergence, universal score ranking, or C++
performance speedup is claimed. Floating-point residuals can vary by platform.
