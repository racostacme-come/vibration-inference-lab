# Build and review the manuscript

From the repository root, install `.[dev]`, regenerate the study if changing
experiments, and run `python paper/build.py --engine tectonic` (or `pdflatex`).
An absolute executable path is supported. A multi-file paper with generated
figures is compiled using the local TeX toolchain, not a single-file editor.

The four-page target includes references and figures. The hard check allows
one to five nonempty pages and rejects undefined references and overfull boxes.
Numbers are loaded directly from `examples/results/metrics.json`. `validation.json`
binds the PDF to code, tests, results, documentation and CI through SHA-256 hashes.
`python scripts/check_manifest.py` checks the committed artifact before CI rebuilds it.

Render with `pdftoppm -r 110 -png paper/paper.pdf paper/build/page` and inspect
every page. Page count alone is not a visual quality check. Rebuild after any
input change; do not edit `numbers.tex` manually. The build directory is ignored.

The source has no invented author, affiliation or peer-review status. It is an
original reproducible computational study of an established algorithm.
