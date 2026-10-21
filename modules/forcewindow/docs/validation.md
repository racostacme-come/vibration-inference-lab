# Validation record

Recorded local run: 13 October 2026, Windows, CPython 3.14.5, MSVC 19.44.

- 39 Python tests passed: matrix/adjoint products, causal nullspace, underdamped,
  critically damped, overdamped and free-mass responses, augmented least-squares
  inversion, failure handling, invalid inputs, exact risk identity and smooth-input
  refinement.
- Fresh CMake `build/verified-native` build passed native `native_state` CTest.
- Ruff lint/format and clang-format checks passed after formatting.
- All JSON numerical metrics and 480 reconstruction / 164 regularization CSV rows
  reproduced. Comparison uses `rtol=2e-6`, `atol=1e-8` across platforms; LSQR iteration
  counts are not required to match, normal residual must be below `1e-12` and the
  independent SVD discrepancy below `1e-7 N`.
- Wheel and sdist builds succeeded. Compiler warnings about pip temporary build
  paths were nonfatal; a fresh direct native build passed separately.

One standalone configure attempt incorrectly retained quotation marks emitted by
`python -m pybind11 --cmakedir` for a path containing spaces. It failed to locate
pybind11. A stale CTest executable then ran in that old directory; that result is
not counted. The corrected command obtains the unquoted path with
`pybind11.get_cmake_dir()` and was checked in a new build directory. CI uses the
corrected command. The final C++ header consolidates the pre-existing unbuilt
scaffold core with the implemented bounded-window API, and the native tests use
that same header.

The manuscript build report is [paper/validation.json](../paper/validation.json).
Local and remote final verification details are recorded in the series index.
The project workflow provides independently observable branch/main CI results;
this document does not substitute for a successful workflow run.

The study establishes numerical behavior for synthetic data only. It supplies no
physical experimental validation, throughput benchmark, or universal ranking of
regularization choices.
