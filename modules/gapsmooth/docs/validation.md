# What the checks establish

1. **Discrete inference error:** recursions agree with a separately assembled
   dense joint Gaussian, including its observation marginal likelihood. The
   reference shares transitions, so this alone cannot verify discretization.
2. **Continuum moment error:** adaptive quadrature independently checks the
   continuous-time covariance integral. Matrix exponential, semigroup and analytical
   stationary covariance checks cover distinct identities. Euler refinement shows
   time-discretization error; exact transitions have only arithmetic error.
3. **Calibration:** 256 independent trajectories, seeds 10000-10255, are evaluated
   at t=4 s. State NEES uses displacement and velocity. Under the specified model
   the sum has chi-square distribution with 512 degrees of freedom. The interval
   is diagnostic, not proof; filter and smoother diagnostics are correlated.
   Coverage uses a pointwise 1.95996398454 standard-deviation displacement interval
   and has finite-sample error.
4. **Physical validity:** untested. Model consistency gives no evidence that a
   real structure has the assumed white forcing, parameters or sensor noise.

The displayed trajectory uses seed 2718; the irregular nine-time reference uses
seed 17. The ensemble uses the analytical stationary prior and the same model
as the estimator. Covariance reduction depends on schedule and parameters,
not the particular observed values.

Euler errors are Frobenius norms after coordinates are scaled by 1 m for
displacement, 1 m/s for velocity and 1 s for time. They depend on that scaling
and are not dimensional energy or displacement errors. Coarse Euler intervals
can be unstable; no unconditional stability is claimed. No continuum sample
path is reconstructed.

Reproduction uses rtol=1e-7, atol=2e-12; verification discrepancies must be below
1e-10. PDF bytes and font rendering can differ across engines. Hashes connect
each manuscript to exact source and numerical inputs.
