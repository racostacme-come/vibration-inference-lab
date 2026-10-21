# Manuscript build

The original paper is [paper.pdf](paper.pdf); its source is [paper.tex](paper.tex).
All figures come from the synthetic study. From the repository root, install the
project's development dependencies, then run:

```sh
forcewindow --output examples/results
python scripts/check_samples.py
python scripts/build_paper.py --engine pdflatex
pdftoppm -scale-to 1400 -png paper/paper.pdf paper/build/page
```

pdfLaTeX requires the standard recommended LaTeX/font packages (including Latin
Modern and microtype). Ubuntu CI installs `texlive-latex-base`,
`texlive-latex-recommended`, and `texlive-fonts-recommended`.
Alternatively supply a path to a Tectonic executable with `--engine`.
Tectonic may download its standard package bundle on first use.

The build runs sufficient LaTeX passes, rejects undefined references/citations and
overfull boxes, checks that the complete PDF has at most five pages, and writes
`validation.json` with input/output SHA-256 hashes. Temporary files and page renders
are kept under ignored `paper/build/`. All pages also require visual inspection;
passing the script is not a claim that visual review was automated.

CI builds independently with pdfLaTeX and uploads the PDF, logs, page PNGs, and
hash manifest. Publication uses the reviewed local PDF; CI PDF bytes may differ
because the TeX engine and embedded metadata differ.
