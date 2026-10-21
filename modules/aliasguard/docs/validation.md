# Validation record

Reference environment: Windows, Python 3.14.5, MSVC 19.44.35219, CMake 4.4.3;
NumPy 2.5.3, SciPy 1.18.1, Matplotlib 3.11.2, pytest 9.1.1, Ruff 0.16.10,
clang-format 23.1.3 and pypdf 6.19.0. CI records separate Ubuntu/Python 3.11 and
Windows/Python 3.14 environments. Dependency ranges are in `pyproject.toml`.

The core branch passed 40 Python tests, native CTest and strict compiler warnings
before merging. The study adds full experiment and CLI tests. Tolerances account
for double precision summation and trigonometric argument roundoff; they are not
statistical error bars. Committed-data reproduction uses rtol=2e-7 and atol=5e-12.
The study has no random inputs. Random unit tests use fixed generator seed 551.

Metrics JSON records the paper's measured scalars. CSV files contain all four
filter lengths, 65 phase cases and 2048 time samples. The figure uses the same
study code. SciPy `lfilter`, `firwin` and `freqz` give independent implementation
comparisons; exact modal signals and phasors provide analytical benchmarks.

At 257 taps the residual-alias and passband-distortion bounds are 7.015e-6 mm and
4.975e-5 mm. High-mode suppression alone understates processing error. At 129 taps
the gain at 185 Hz is smaller than at 257 taps, although the latter better preserves
71 Hz. No measured structural experiment, speed benchmark or continuum convergence
claim is made. Local and CI results are recorded in the parent series index.
