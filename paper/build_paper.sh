#!/usr/bin/env bash
# Rebuild the paper: figures and numbers from data/, then LaTeX (pdflatex + bibtex via latexmk).
# Usage: bash paper/build_paper.sh [path/to/paper_runs]   (optional: re-collect campaign summaries)
set -euo pipefail
cd "$(dirname "$0")/.."
if [ $# -ge 1 ]; then
  python code/paper/make_paper_figures.py --runs "$1"
else
  python code/paper/make_paper_figures.py
fi
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
latexmk -c main.tex
