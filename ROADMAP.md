# Seven-day major-project cycle

This release consolidates existing work; its historical development is not portrayed as seven new days of work.

Future substantive extensions use a seven-day target, with a permitted 5–9-day window:
1. State research question, acceptance criteria and shared-model assumptions.
2. Implement the first numerical improvement and independent oracle.
3. Integrate the second component and test interface failure cases.
4. Complete the end-to-end experiment and parameter sensitivity.
5. Verify convergence, regression tests and cross-platform builds.
6. Draft and compile the <=5-page paper, reproduce figures and inspect pages.
7. Release after evidence review; document limitations and next questions.

Days 8–9 are reserved for justified fixes. A missed acceptance criterion blocks a completed-release claim. Each day must record a meaningful contribution and push the corresponding branch. The series portfolio tracks the active cycle.

## Consolidation recovery contribution, 25 October 2026

The active recovery cycle began 21 October, targets 27 October, and permits justified repairs through 29 October. This actual day-5 contribution completes the preserved integration draft on `feat/independent-vibration-validation`: 209 original-blob matches, independent scalar dynamics/covariance and dense inference references, time refinement, dither checks, a prior-matched ensemble, four native CTests and a three-page synthesis. The failed single-realization improvement assertion was corrected without changing its seed. All 294 Python tests and four native tests pass locally.

Next: publish this branch, verify Linux/Windows/manuscript CI and portable checkouts, inspect the independent PDF, then release only after all evidence passes. On any external failure, preserve this branch and repair the same suite. No new research begins before all three consolidation suites are accepted. The portfolio records the exact commits, hosted results and next cycle.
