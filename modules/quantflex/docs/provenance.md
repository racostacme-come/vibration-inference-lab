# Concept provenance and originality

Read-only source inspection began on 20 October 2026; implementation continued
on 21 October. All three current DTU source folders were inventoried.

- **Primary: MLSP**, `Books/SPT-ML - ch05.pdf`, PDF pages 51-54,
  chapter "Sampling and Quantization" by Hakan Johansson. Sections on uniform
  rounding, ADC error, and the assumptions behind a white-error model motivate
  testing a deliberately small periodic displacement. This is a new topic
  relative to AliasGuard's anti-alias filtering and ForceWindow's inverse loading.
- **ATSA:** the opening of Lecture 6 on multivariate probabilistic forecast
  evaluation was inspected; the folder also contains a recently modified
  Assignment 1. No assignment, data or solution was used in this project.
- **MMME:** the opening of the Lecture 4 typeset notes identifies explicit heat
  diffusion stability and interface-cell matching; these topics are already
  represented in HeatStart and ContactScale.
- **Mechanics connection:** analytical forced response and cycle-average spring
  energy connect the signal-processing experiment to structural dynamics.

The code, synthetic signal, experiment design, plots and manuscript are original
to this repository. No private source pages, solutions, figures, course data or
course text are included. Course names and topic descriptions record provenance,
not endorsement. The manuscript cites publicly accessible research instead of
reproducing private teaching material. The dither theorem is established theory;
no novelty of that theorem or of least squares is claimed. No experimental
instrument or material is represented by the chosen numerical parameters.

## Public references inspected

1. S. P. Lipshitz, R. A. Wannamaker and J. Vanderkooy, "Quantization and
   Dither: A Theoretical Survey," JAES 40(5), 355-375 (1992).
   [Publisher record](https://aes.org/publications/elibrary/elibrary-page/?id=7047),
   [public paper](https://hajim.rochester.edu/ece/sites/zduan/teaching/ece472/reading/Lipshitz_1992.pdf).
   Section 4 establishes the subtractive uniform-dither conditions.
2. J. Rapp, R. M. A. Dawson and V. K. Goyal, "Estimation from Quantized Gaussian
   Measurements: When and How to Use Dither" (2019),
   [arXiv:1811.06856](https://arxiv.org/abs/1811.06856).
   Context for why adding dither need not minimize estimation error. Its Gaussian
   noise experiments and estimators are not reproduced here.
