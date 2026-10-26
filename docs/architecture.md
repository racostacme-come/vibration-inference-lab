# Architecture and compatibility

The composition boundary is a Python process with NumPy arrays and explicit model parameters. Native C++ extensions keep their distinct package names. Every module builds from its own pyproject.toml, preserving build configuration and package metadata.

The root suite.py coordinates installation, source-integrity verification, isolated test processes, the connected experiment, and manuscript compilation. It does not monkey-patch module internals. integration.py contains all cross-module mappings so changes in sign conventions, units or constraints are reviewable.

## Source evaluation

All included modules had clean Git worktrees when imported. Each already supplied numerical tests and an academic paper. The consolidation retains the complete tracked contents of each source commit. The original CI records are historical evidence, while validation/tests.json records the new integrated run.

## Compatibility boundaries

DampWatch uses dimensional M,C,K; GapSmooth uses mass-normalized omega,zeta,q. The bridge explicitly sets mass=1, stiffness=omega^2 and damping=2*zeta*omega. Quantization variance is step^2/12 only for independent unclipped subtractive dither. Its noise is uniform, so the Gaussian likelihood is not a calibrated posterior-density result. LoadCoherent and JointGuard are separate statistical models, not stages automatically applied to this oscillator.

## Extension policy

Design a research question first. Change module APIs only with regression tests and a migration note. Keep imported licenses, commit provenance and component papers. Do not interpret low residuals or passing tests as evidence of physical validity.

## Independent references and native handoff

`verification.py` uses scalar analytic transitions and numerical covariance integration, then dense covariance conditioning, without calling module transitions or recursive inference. `integration.py` includes the fixed-state realization and a separate prior-matched ensemble of actual component inference runs. Root CMake builds three preserved tests directly and one original rectangular-load convolution check; this leaves module manifests unchanged. `check_results.py` checks every recorded diagnostic; CI then regenerates reports for download. The manuscript compiles the committed, checked evidence and rejects layout/reference/page-count errors.
