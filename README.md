# Vibration Inference Lab

Dynamics, missing-data inference, sampling, inverse forces, coherent forecasts, dependence risk and quantization.

This major project integrates 7 previously independent research implementations into one reproducible laboratory. Full Python and C++ source, tests, datasets, documentation, licenses and component papers are included under `modules/`; no submodule downloads are required.

## Start here

Python 3.11+ (CI uses 3.12), Git, internet for initial dependency installation, and a C++17 compiler are required. On Windows use Visual Studio 2022 with Desktop development with C++ and a Windows SDK. On Linux use GCC or Clang. Use a fresh virtual environment.

```sh
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python suite.py verify
python suite.py install
python suite.py test
python suite.py native
python check_results.py
python suite.py experiment
```

The runner stops on install failure and reports every module's test outcome. Tests execute in separate processes to avoid duplicate test-file/import names. `validation/tests.json` contains the actual outputs; `results/integration.json` records the connected experiment. Rerunning overwrites only these generated reports. For Windows builds, if compiler discovery fails, use a Visual Studio developer shell or set `CMAKE_GENERATOR=Visual Studio 17 2022`.

## Included components

| Component | Implementation and original paper |
| --- | --- |
| dampwatch | [Source and usage](modules/dampwatch/README.md), [paper](modules/dampwatch/paper/paper.pdf) |
| gapsmooth | [Source and usage](modules/gapsmooth/README.md), [paper](modules/gapsmooth/paper/paper.pdf) |
| aliasguard | [Source and usage](modules/aliasguard/README.md), [paper](modules/aliasguard/paper/paper.pdf) |
| forcewindow | [Source and usage](modules/forcewindow/README.md), [paper](modules/forcewindow/paper/paper.pdf) |
| loadcoherent | [Source and usage](modules/loadcoherent/README.md), [paper](modules/loadcoherent/paper/paper.pdf) |
| jointguard | [Source and usage](modules/jointguard/README.md), [paper](modules/jointguard/paper/paper.pdf) |
| quantflex | [Source and usage](modules/quantflex/README.md), [paper](modules/quantflex/paper/paper.pdf) |

## What is integrated

DampWatch generates an exact-in-time decaying oscillator response; QuantFlex applies seeded subtractive-dither quantization; GapSmooth estimates the state across a missing-data interval. An independent state-transition comparison checks the model handoff. Sampling, inverse loading, forecast coherence and joint-risk components provide complementary investigations through the same environment and validation interface.

This is a modular integration, not a universal multiphysics solver. Component APIs retain their explicit assumptions and units. Not every module is physically coupled to every other module.

## Paper and evidence

Read the new [synthesis paper](paper/paper.pdf) and [LaTeX source](paper/paper.tex).
The component papers are retained as detailed supporting studies. New numerical claims are based on the connected experiment, not on simply adding together earlier performance claims.

Compile with Tectonic or pdfLaTeX:
```sh
python suite.py paper --engine tectonic
```
The command rejects unresolved references, overfull boxes and more than five pages. CI independently builds the paper. [Architecture](docs/architecture.md), [source provenance](suite.json), and [development roadmap](ROADMAP.md) describe scope and future work.

## Provenance and development

Imported snapshots are pinned to original Git commits in `suite.json`, with SHA-256 for every imported file. `verify` detects unrecorded changes. Original repositories and histories remain available; this repository starts a new integration history. Intentional module changes require a documented patch record and manifest refresh. MIT notices are retained for every component; the integration layer is also MIT licensed.

CI runs all component Python tests and the integration test on Linux and Windows. C++ Python extensions are built during install. `native` builds three preserved C++17 kernel tests and a new continuous rectangular-load handoff check with warnings as errors. Sanitizer and full historical reproduction campaigns are not claimed. Recorded results are synthetic numerical verification, not experimental physical validation.

## Independent evidence and interpretation

[Recorded validation](docs/validation.md) distinguishes implementation agreement, discrete inference error, continuous-time error and physical validity. [Independent results](results/independent.json) include closed-form transitions in three damping regimes, covariance quadrature, eight dense-versus-recursive inference cases, and generalized-alpha refinement. [Connected results](results/integration.json) retain the fixed-state case and add 256 prior-matched trials.

The fixed-state smoothed outage RMSE is **slightly worse** (0.000561613 m versus 0.000556839 m). This is a useful counterexample to a universal per-trajectory improvement assertion. In the matched ensemble, smoothing lowers measured MSE by about 16.8%; Monte Carlo standard errors and predicted error covariance are reported. Uniform dither supports linear second-moment estimation, not exact Gaussian posterior probabilities. No physical sensors or structures were validated.

The continuous oscillator uses `x=[u,v]`, `M=m`, `C=2*m*zeta*omega`, `K=m*omega**2`; acceleration-noise intensity has units m²/s³. Dither is independent uniform noise on one quantization interval and is subtracted after rounding. Clipping invalidates the variance model `R=step**2/12`. Times are in seconds; a prior is defined before the observation at the first time; NaN marks a missing observation.

API entry points:

| API | Contract |
| --- | --- |
| `integration.experiment()` | Fixed-state experiment and actual filter/smoother ensemble; returns JSON-compatible metrics |
| `verification.transition(omega,zeta,time)` | Scalar closed-form 2x2 transition for bounded verification cases |
| `verification.process_covariance(...)` | Independent Gauss-Legendre process-covariance reference |
| `verification.batch_estimate(...)` | Dense small-grid second-moment oracle; filtered/smoothed means and covariances |
| `verification.experiment()` | Independent checks and continuous-time refinement evidence |

`python check_results.py` recomputes every recorded diagnostic without overwriting it (rtol 1e-6, atol 1e-11). The experiment command regenerates the JSON and both manuscript tables. The paper command requires these versioned inputs and Tectonic or pdfLaTeX. `requirements-reproduced.txt` records local dependency versions; install it before `suite.py install` to reproduce that environment. CMake 3.20+ is required for native tests. For long Windows workspace paths, use `python suite.py native --build-root ../../.native-vibration`.

[Original-blob audit](docs/provenance-audit.json) records 209 exact snapshot matches; no imported changes were needed. [Course/topic provenance](docs/course-provenance.md) identifies the read-only inspiration without distributing private material.
