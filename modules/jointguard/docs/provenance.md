# Course grounding and original work

On 20 October 2026 the local DTU ATSA, MLSP and MMME source folders were
inventoried read-only. The available ATSA lecture **Evaluation of Probabilistic
Forecasts** (October 2026), stored as
`Lectures/Lecture_6_multivariate_probabilistic_forecast_evaluation.pdf`,
motivated this project. Pages 31 and 34-35 discuss dependence evaluation,
variogram scoring and its blindness to a common mean shift; page 39 compares
diagnostic properties. These topic descriptions are provenance, not reproduced
course text. Public primary references are cited in the README and paper.

MLSP's chapter 5, *Sampling and Quantization*, and MMME's lecture 4 on explicit
thermal stability/interface matching were also inspected. The series rotation
returns to ATSA after ContactScale (MMME). No course acronym was used to infer
its content. No private notes, exercises, solutions, exams, datasets or copied
figures are distributed. DTU source folders were not modified.

The two-displacement spring system, numerical parameters, experiments, source
code, illustrations and manuscript are original to this repository. They are
an illustrative scientific-computing study, not a claim of a new scoring rule
or experimentally validated mechanics. The energy score, Gaussian scoring
identities and variogram score are established methods.

This project studies **forecast evaluation**, unlike earlier GapSmooth
(filtering/smoothing) and LoadCoherent (hierarchical reconciliation). It does not
fit forecasts or reuse their implementations. The series contains both Python
and C++; this project uses a custom C++17 pair-distance core with pybind11.

The manuscript build helper follows the series' existing verification convention.
