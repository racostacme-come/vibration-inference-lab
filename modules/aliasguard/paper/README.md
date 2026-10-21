# Manuscript build

The original research-format manuscript is `paper.tex`; the reading copy is
`paper.pdf`. It claims neither peer review nor experimental physical validation.

From the repository root, with the development environment installed:

```sh
python scripts/check_samples.py
python paper/build.py --engine tectonic
# Or install TeX Live latex-extra and fonts-recommended:
python paper/build.py --engine pdflatex
python scripts/check_manifest.py
pdftoppm -r 110 -png paper/paper.pdf paper/build/page
```

Pass an absolute executable path to `--engine` if needed. Tectonic may download
its TeX bundle initially. pdfLaTeX runs twice with shell escape disabled. The build
generates `numbers.tex` from committed metrics, rejects undefined references and
overfull boxes, enforces 1--5 nonempty pages, and records input and PDF SHA-256
hashes in `validation.json`. It places the checked PDF at `paper/paper.pdf`.

Solver, tests, figure or result changes require a rebuild; the manifest checker
rejects stale tracked inputs. Numerical reproduction is tolerance-based across
supported libraries. PDF bytes can differ across engines. CI uploads its independently
compiled PDF, log and rendered pages. Visual inspection is a separate check;
the automated gate alone cannot establish readability.
