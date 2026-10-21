# Validation design

The tests use an analytic two-estimator inverse-variance example and a separate
saddle-point KKT solve. They also check projection idempotence, preservation of
coherent inputs, weighted orthogonality, oracle covariance dominance, scale
invariance, batch shape, defensive copies, singular weights, invalid input,
coherent bias and a negative-output counterexample.

The full study is deterministic for the recorded NumPy generator and seed.
`scripts/check_samples.py` recomputes every numerical field in both CSV files and
the JSON record, including all 960 conditional calibration risks, using relative
tolerance 2e-8 and absolute tolerance 2e-11. Roundoff-scale diagnostics can vary
across BLAS implementations; this tolerance is not a physical error bound.
Plots are regenerated from those values; image bytes are not a numerical test.

The fixed-weight Gaussian experiment uses 60,000 independent error vectors.
Its comparison scale is the analytic Monte Carlo standard error
sqrt(2 trace((P V P^T)^2)/60000). A five-standard-error test detects substantial
regressions; it is not proof of a distributional model for real errors.
Calibration experiments use 80 independent training sets at each sample size,
with the same set shared across the three estimators for a paired comparison.
Each trained projection is evaluated with the *known population covariance*.
No test outcomes are used to select the shrinkage parameter (fixed at 0.5).
The reported 10th/90th percentiles describe replicate variation, not confidence
intervals. Mean standard errors across replicates are included separately.

The sum of seven node MSEs counts aggregated loads as separate targets. It is a
declared forecasting objective, not physical dissipated power or independent
building-level error. Node-level values remain in the JSON output.

No continuum PDE is discretized: quasi-steady truth is evaluated analytically,
so there is no spatial/time mesh convergence result. Floating-point projection
error, statistical estimation error and physical validity are distinct. No
experimental measurements establish physical validity.

CI validates Ubuntu/Python 3.11 and Windows/Python 3.14, source and installed
wheel tests, lint/format, complete sample reproduction, sdist/wheel builds,
manuscript input hashes and independent pdfLaTeX compilation with a five-page
limit and undefined-reference/overfull-box rejection. CI artifacts contain
the compiled paper and page renders for visual review.
