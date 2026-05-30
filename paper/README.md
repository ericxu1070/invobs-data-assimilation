# Final report — Full 4D-Var with a Learned Inverse Observation Operator under Observation Noise

LaTeX source for the AM 170B capstone report by Eric Xu, Javier Sanchez, Jiayang Wu,
Wilson Xie, and Edmund Xu. The report extends Frerix et al. 2021
([arXiv:2102.11192](https://arxiv.org/abs/2102.11192)) by using the **full four-dimensional
variational (4D-Var) cost** `J = J_b + J_o` (with a background term and an explicit
observation-error covariance) instead of the paper's simplified observation-misfit objective,
by **adding Gaussian observation noise** to every experiment, and by adding an operational
**sliding-window (cycling) 4D-Var** scheme. Two systems: Lorenz-96 and 2D Kolmogorov flow.

## Build

From this `paper/` directory:

```
latexmk -pdf main.tex
```

(Uses `pdflatex` + `bibtex`; produces `main.pdf`, 29 pages.) The local build used TinyTeX.
The preamble drops the `caption`/`subcaption` packages so it builds on a minimal TeX Live
install; no other non-standard packages are required.

## Layout

- `main.tex` — top-level document (preamble, abstract, `\input`s the section files, bibliography).
- `_sections/*.tex` — one file per section: `intro`, `background`, `methods`, `results_l96`,
  `results_kflow`, `discussion`.
- `figures/` — the 18 figures used in the report (pulled from the project notebooks).
- `references.bib` — bibliography.
- `_build/` — reproducibility helpers, not needed to compile:
  - `_extract_notebooks.py` — regenerates figures + text digests from the `.ipynb` notebooks.
  - `_digests/` — plain-text digests of each source notebook (figures referenced inline).
  - `_manifest.json` — every figure extracted from the notebooks, with source cell + context.
  - `extracted_unused/` — the 50 extracted figures not used in the report.
  - `_author_guide.md`, `_preamble.tex` — the conventions/preamble used while authoring.

## Source notebooks for the figures

- `PyTorch_InvObs_DA_v2_PaperFaithful.ipynb` — inverter training, baseline init, tuned-σ_b 4D-Var.
- `PyTorch_InvObs_DA_v2_Integrator.ipynb` — RK4-vs-adaptive divergence, Lyapunov time.
- `PyTorch_InvObs_DA_v2_Kolmogorov.ipynb` — Kolmogorov training trajectory.
- `SlidingWindow_PyTorch.ipynb` — Lorenz-96 cycling experiments A/B/C/E.
- `SlidingWindow_Kolmogorov_PyTorch.ipynb` — Kolmogorov cycling experiments.
