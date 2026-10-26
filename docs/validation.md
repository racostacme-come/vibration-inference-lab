# Validation record: 25 October 2026

## Publication-gate extension, 26 October 2026

The day-6 contribution adds executable checks for all nine manuscript hashes,
the actual PDF page count, and both tables regenerated from versioned numerical
results. Eight acceptance/mutation tests reject altered code, results, PDF bytes,
tables, missing manifest entries and incorrect page counts. No imported module
or numerical experiment changed. CI checks the committed manuscript before
recompiling, then checks the independently built manuscript again.

Local execution passed **302 Python tests** (284 component and 18 root), all
four MSVC C++17 CTests, all 209 imported hashes and reproduction of every recorded
metric (rtol=1e-6, atol=1e-11). The local three-page PDF was rendered and all pages
visually reviewed again. The paper dated 25 October describes its original
294-test numerical baseline; the eight new publication checks supplement it.
All 19 original component repositories remain clean. Final hosted acceptance
for this contribution is recorded separately in `docs/release-evidence.json`.

## Original numerical baseline

Local Windows, Python 3.12.2, MSVC 19.44.35219.0. Dependency versions are recorded in `requirements-reproduced.txt`. This records execution, not a claim about future runs.

- All 209 manifest entries match imported bytes and original pinned Git blobs. No imported modifications.
- 284 component Python tests plus 10 root tests passed (294 total). Full output: `validation/tests.json`.
- Four C++17 CTests passed, with warnings as errors: streaming, state/adjoint, energy score and original oscillator handoff. Full output: `validation/native.json`.
- Closed-form transition maximum difference 2.665e-15; dimensional response 2.356e-15; process-covariance quadrature 2.165e-15.
- Eight independent dense-conditioning cases: mean difference <=2.387e-14; covariance difference <=4.303e-15. Nonuniform times, missing endpoints/interior/all/none, q=0 and q=0.6.
- Generalized-alpha finest observed orders 1.999257 (rho=0.5), 1.999805 (rho=1). Errors 2.885914e-5 and 1.924682e-5 m in the explicitly scaled state norm.
- Uniform dither sample variance/theoretical variance =1.001147 across 120000 draws. A clipped counterexample rejects the uniform-error assumption.
- Fixed-state outage RMSE: filtered 5.568395e-4 m, smoothed 5.616128e-4 m. The draft assertion of guaranteed realized improvement failed and was replaced by covariance/ensemble checks; its seed was retained.
- 256 actual prior-matched filter/smoother runs: MSE 1.031742e-6 and 8.589057e-7 m². Expected MSE 9.784194e-7 and 8.144805e-7; Monte Carlo SE 5.731130e-8 and 4.821708e-8. Paired improvement SE 3.350686e-8 m².
- Three-page Tectonic manuscript compiled; layout overflows were repaired and all pages visually inspected. See `paper/build-report.json` for input/PDF hashes.

Hosted [run 40239408015](https://github.com/racostacme-come/vibration-inference-lab/actions/runs/40239408015) passed Linux, Windows and manuscript jobs at code revision `d9178b5ab87e9aabe53b302722c6e82c20a38f9b`. Downloaded Linux and Windows reports each confirm 294 Python tests, four native tests and reproduced numerical metrics within rtol=1e-6, atol=1e-11. Both fresh local clones (`core.autocrlf=false` and `true`) pass all 209 hashes. All 19 original component repositories remain clean.

The independent CI pdfLaTeX output also has three pages. All nine input/output hashes were checked, and all three CI pages were visually inspected in addition to the three local pages. Two transient artifact-download transport failures were resolved by retrying through the GitHub connector/CLI; no validation failure was concealed. Release publication is recorded by Git tags and GitHub releases. No physical or external experimental validation is inferred from these software checks.

Implementation agreement (including native invariants) is separate from continuous-time refinement. Exact stochastic transition moments do not prove sample-path accuracy. The ensemble tests known-prior linear-estimation risk, not Gaussian posterior coverage. No physical validation, sanitizer run or complete historical reproduction campaign is claimed.
