# Author guide — invobs-DA final paper (shared by all workflow agents)

This file is the single source of truth for the LaTeX report. Every agent must follow it.

## 0. What this paper is

A final capstone report (group: Eric Xu, Javier Sanchez, Jiayang Wu, Wilson Xie, Edmund Xu;
course AM 170B). It extends Frerix et al. 2021, "Variational Data Assimilation with a Learned
Inverse Observation Operator" (arXiv:2102.11192), by (1) replacing the paper's simplified
observation-misfit objective with the **full four-dimensional variational (4D-Var) cost**
`J = J_b + J_o` that includes a background term and an explicit observation-error covariance,
and (2) **adding Gaussian observation noise** to every experiment. We also add an operational
**sliding-window (cycling) 4D-Var** extension. Two dynamical systems: the Lorenz-96 model (one
dimensional) and two-dimensional Kolmogorov flow (incompressible Navier–Stokes).

## 1. Document structure (final section order)

1. Abstract (written by the Assemble agent)
2. `\section{Introduction}`
3. `\section{Background: The Learned Inverse-Observation Approach and Our Extensions}`
   — summarize Frerix et al. 2021 and state precisely how we change it to test it.
4. `\section{Models and Methods}` — dynamical systems; the full 4D-Var cost; the learned
   inverse observation operator; observation noise; preconditioning; sliding-window cycling.
5. `\section{Results}` — `\subsection{Lorenz-96}` then `\subsection{Two-Dimensional Kolmogorov Flow}`.
6. `\section{Discussion and Conclusion}` — caveats/limitations of our experiments, future
   work, and a summary of findings.
7. Bibliography (Assemble agent adds `\bibliographystyle{unsrtnat}` + `\bibliography{references}`).

## 2. Hard rules

- **Acronyms:** spell out in full at first use, with the short form in parentheses, then use the
  short form thereafter. See the list in §4. Do this per the whole document (the first use is
  typically in the Introduction). Never use a short form before it is defined.
- **No fabricated numbers.** Use only numbers that appear in the digests or in the "results crib"
  (§6). If you want to state a number you cannot find, omit it.
- **Figures:** use ONLY the exact filenames in §5. Every figure needs a caption and must be
  referenced in the prose with `\ref`. Do not invent filenames. Files live in `paper/figures/`
  and the preamble sets `\graphicspath{{figures/}}`, so `\includegraphics{NAME.png}` (no path).
- **Citations:** use only the BibTeX keys in §3 via `\citep{...}` / `\citet{...}`.
- Use the notation macros in §4 for consistency.

## 3. Citation key menu (references.bib)

- `frerix2021invobs` — the paper we extend (Frerix et al. 2021).
- `lorenz1996predictability` — Lorenz-96 model origin.
- `kalnay2003atmospheric` — DA / NWP textbook (Kalnay 2003).
- `ledimet1986variational` — variational DA foundations (Le Dimet & Talagrand 1986).
- `courtier1994strategy` — incremental 4D-Var (Courtier et al. 1994).
- `liu1989limited` — L-BFGS optimizer (Liu & Nocedal 1989).
- `kochkov2021machine` — JAX-CFD / ML-accelerated CFD (Kochkov et al. 2021); source of the
  Kolmogorov-flow solver design the original paper used.
- `paszke2019pytorch` — PyTorch (our reimplementation framework).
- `elfwing2018sigmoid` — SiLU activation.
- `ioffe2015batch` — batch normalization.
- `dormand1980family` — Dormand–Prince (dopri5) adaptive integrator.
- `bocquet2019data` — DA + machine learning context.
- `geer2021learning` — ML vs DA perspective.

(If a key looks off, only these keys exist; cite by what is listed here.)

## 4. Notation and acronyms

Macros defined in the preamble (use them): `\xstate` (physical state x), `\yobs` (observation y),
`\Hop` (observation operator H), `\Hinv` (learned inverse observation operator), `\Mop` (forward
model / integrator), `\Jb`, `\Jo`, `\Bcov` (B), `\Rcov` (R), `\sigb`, `\sigobs`, `\rmse`.

Canonical math:
- Forward model / integrator: `\Mop^t(\xstate_0)` is the state at step t from initial condition x_0.
- Observation operator H subsamples the state (every 4th grid point for Lorenz-96; every 16th in
  each direction for Kolmogorov flow).
- Learned inverse observation operator `\Hinv`: a CNN mapping an observation sequence Y back to a
  physical-state sequence.
- Full 4D-Var cost (the central new equation):
  `J(\xstate_0) = \tfrac12 (\xstate_0-\xstate_b)^\top \Bcov^{-1} (\xstate_0-\xstate_b)
   + \tfrac12 \sum_{t} (\Hop \Mop^t(\xstate_0) - \yobs_t)^\top \Rcov^{-1} (\Hop \Mop^t(\xstate_0)-\yobs_t)`
  with `\Bcov = \sigb^2 \bm{C}` (C = spatial correlation) and `\Rcov = \sigobs^2 \bm{I}`.
- Physics-space (hybrid) companion cost replaces J_o with
  `\tfrac{1}{2\sigma_p^2}\sum_t \lVert \Mop^t(\xstate_0) - [\Hinv(Y)]_t \rVert^2`.
- Preconditioning: optimize in decorrelated coordinates `\bm{z} = \bm{C}^{-1/2}\xstate`, so the
  background term becomes `\lVert \bm{z}_0 - \bm{z}_b \rVert^2 / (2\sigb^2)`. (Lorenz-96 only;
  Kolmogorov flow optimizes the vorticity directly with B = sigma_b^2 I, no decorrelation,
  because a full 4096x4096 spatial covariance is too expensive.)

Acronyms to define on first use (full form — short form):
data assimilation (DA); numerical weather prediction (NWP); four-dimensional variational data
assimilation (4D-Var); ordinary differential equation (ODE); partial differential equation (PDE);
root-mean-square error (RMSE); mean squared error (MSE); convolutional neural network (CNN);
fourth-order Runge–Kutta (RK4); Dormand–Prince (dopri5); limited-memory
Broyden–Fletcher–Goldfarb–Shanno (L-BFGS); model time unit (MTU); sigmoid linear unit (SiLU);
Gaussian error linear unit (GELU); batch normalization (BN); one-dimensional (1D);
two-dimensional (2D); graphics processing unit (GPU).

## 5. Figure manifest (exact filenames; assigned section)

Place each figure in the assigned section ONLY. Suggested `\includegraphics[width=W\linewidth]`.
The Analyze agents write the final caption text for each; if a caption is missing, use the
description here as the caption.

### Methods section
- `noise_distributions.png` (W=0.95) — Gaussian observation-noise probability density functions
  for the three noise standard deviations sigma_obs in {0.1, 0.5, 1.0} used in the experiments.

### Results — Lorenz-96
- `L96_Integrator__cell08__out00.png` (W=0.7) — L2 divergence between fixed-step RK4 and adaptive
  Dormand–Prince trajectories vs model time, median and min/max band over 5 attractor states.
- `L96_Integrator__cell10__out00.png` (W=0.7) — Lyapunov-time fit: exponential separation of two
  trajectories started 1e-6 apart; fitted growth rate lambda and T_L = 1/lambda.
- `L96_PaperFaithful__cell10__out00.png` (W=0.7) — training-loss curves (log MSE vs epoch) for the
  original PyTorch port inverter vs the paper-faithful inverter at sigma_obs=0.
- `L96_PaperFaithful__cell13__out00.png` (W=0.75, optional) — x_0 reconstruction for 4 held-out
  trajectories: truth vs port vs paper inverter.
- `L96_PaperFaithful__cell19__out00.png` (W=0.92) — analysis RMSE bar chart comparing repeat- vs
  paper-baseline initialization, each with observation-space and hybrid optimization, across
  noise levels (untuned sigma_b = 1).
- `L96_PaperFaithful__cell20__out00.png` (W=0.95) — Hovmoller comparison at sigma_obs=0.5: truth
  vs repeat-baseline+obs vs paper-baseline+obs.
- `L96_PaperFaithful__cell28__out00.png` (W=0.95) — KEY: tuned-sigma_b full 4D-Var. Analysis RMSE,
  one panel per noise level, bars = (paper/invobs init) x (obs/hybrid opt), best sigma_b annotated.
- `L96_SlidingWindow__cell27__out00.png` (W=0.95) — Experiment A (single 8-step window): RMSE over
  window+forecast for 4 init x opt combos, plus analysis-RMSE bar, at sigma_obs in {0, 0.5}.
- `L96_SlidingWindow__cell30__out00.png` (W=0.92) — Experiment B1: sliding-window RMSE per stride
  in {2,4,8} for T_obs in {24,48} and sigma_obs in {0,0.5}, dashed best single-window reference.
- `L96_SlidingWindow__cell31__out00.png` (W=0.9, optional) — Experiment B2: per-cycle background vs
  analysis RMSE for the best-stride run.
- `L96_SlidingWindow__cell34__out00.png` (W=0.95) — KEY: Experiment C, sliding-window-best vs
  single-window invobs+hybrid vs single-window baseline+obs (RMSE over time, 2x2 grid).
- `L96_SlidingWindow__cell35__out00.png` (W=0.85, optional) — Experiment D: analysis RMSE at the
  last-window start (bar chart) for the three Experiment-C methods.
- `L96_SlidingWindow__cell36__out00.png` (W=0.95) — Experiment E: Hovmoller (truth / SW-best /
  invobs+hybrid + signed error) for each (T_obs, sigma).
- `L96_SlidingWindow__cell37__out01.png` (W=0.9, optional) — per-cycle L-BFGS loss curves for a
  sliding-window-best run (gradient-explosion-free convergence each cycle).

### Results — Two-Dimensional Kolmogorov Flow
- `KFlow_Full4DVar__cell06__out00.png` (W=0.95) — vorticity snapshots of one training trajectory
  (64x64 grid) at five times, illustrating the turbulent state being assimilated.
- `KFlow_SlidingWindow__cell19__out00.png` (W=0.95) — warmup diagnostic: enstrophy vs outer step
  and vorticity snapshots, confirming the flow reaches a statistically stationary regime.
- `KFlow_SlidingWindow__cell24__out00.png` (W=0.9) — Kolmogorov Experiment A (single 10-step
  window): spatial RMSE over window+forecast for 4 init x opt combos, plus analysis-RMSE bar.
- `KFlow_SlidingWindow__cell27__out00.png` (W=0.9) — Kolmogorov Experiment B1: RMSE per stride in
  {2,5,10} for T_obs in {20,30}.
- `KFlow_SlidingWindow__cell28__out00.png` (W=0.9, optional) — Kolmogorov Experiment B2: per-cycle
  background vs analysis RMSE for the best-stride run.
- `KFlow_SlidingWindow__cell31__out00.png` (W=0.9) — Kolmogorov Experiment C: sliding-window-best
  vs single-window methods (RMSE over time).
- `KFlow_SlidingWindow__cell32__out00.png` (W=0.85, optional) — Kolmogorov Experiment D: analysis
  RMSE at the last-window start (bar chart).
- `KFlow_SlidingWindow__cell33__out00.png` (W=0.85) — Kolmogorov vorticity-field snapshots: truth /
  SW-best / invobs+hybrid and signed error, at window start / end / forecast end (T_obs=20).

### Figure environment template (copy this pattern)
```
\begin{figure}[t]
  \centering
  \includegraphics[width=0.9\linewidth]{FILENAME.png}
  \caption{CAPTION TEXT.}
  \label{fig:SHORTKEY}
\end{figure}
```

## 6. Results crib — vetted numbers (ground truth; do not contradict)

**Lorenz-96 system:** grid size 40, forcing F=8, observe every 4th point (10 of 40 observed),
outer step dt=0.1 (10 inner RK4 substeps of 0.01).

**Integrator diagnostic (Phase 3):**
- RK4-vs-dopri5 divergence reaches order 1 at t* = 6.30 MTU for one trajectory; across 5
  attractor states median t* = 7.70 MTU, range [6.50, 8.30] MTU. Divergence at t=19.9 MTU ~ 28.5.
- Lyapunov fit: lambda = 2.263 / MTU, T_L = 1/lambda = 0.442 MTU (paper expects ~0.6; ours is a
  bit short, a known consequence of the fixed-step integrator — report honestly).

**Inverter training (Phase 1, sigma_obs=0):** port final MSE 0.3714; paper-faithful final MSE
0.4924. Held-out (100 trajectories, seed 100): port test L1 = 0.3838, x0 RMSE 0.9973 +/- 0.3261;
paper test L1 = 0.4470, x0 RMSE 1.1635 +/- 0.4036; paper/port L1 ratio 1.165 (within 30%).
Per-noise paper inverter final MSE: sigma=0.1 -> 0.6321; 0.5 -> 1.3338; 1.0 -> 2.2201.

**Baseline initialization (Phase 2, sigma_obs=0, T=10, N=100):** repeat-interleave baseline x0
RMSE 4.7828 +/- 0.5006; paper (climatological-mean) baseline 3.0892 +/- 0.2539; ratio 0.646
(paper baseline ~35% lower). Spatial-correlation matrix C condition number 8.72; climatological
mean ~2.34.

**Tuned-sigma_b full 4D-Var (Phase 4, N=16, T=10, sigma_b grid {0.1,0.3,1,3,10}):** best analysis
RMSE per cell (best sigma_b in parentheses):
- sigma_obs=0.1: paper+obs 2.466 (0.1); paper+hybrid 0.621 (3.0); invobs+obs 0.883 (0.3);
  invobs+hybrid 0.620 (10.0).
- sigma_obs=0.5: paper+obs 2.393 (0.3); paper+hybrid 1.223 (1.0); invobs+obs 1.550 (0.3);
  invobs+hybrid 1.168 (1.0).
- sigma_obs=1.0: paper+obs 2.708 (0.3); paper+hybrid 1.846 (1.0); invobs+obs 1.867 (1.0);
  invobs+hybrid 1.676 (1.0).
Takeaways: (a) invobs init + hybrid optimization gives the lowest analysis RMSE at every noise
level; (b) hybrid optimization dramatically beats observation-only optimization for the
climatological-mean baseline (e.g. 2.39 -> 1.22 at sigma=0.5); (c) the expected ordering
"invobs+hybrid prefers a SMALLER optimal sigma_b than paper+obs" did NOT hold (invobs+hybrid often
preferred a larger sigma_b) — report this as a negative/honest result; (d) tuning sigma_b helped
only marginally over sigma_b=1 in most cells.

**Sliding-window Lorenz-96:** per-cycle window WINDOW_T=8, forecast horizon 50, ensemble N=50,
noise {0.0, 0.5}; single-window background sigma_b=1.0, cycling sigma_b=0.3; T_obs in {24,48},
strides {2,4,8}. Best stride: (T=24,sigma=0)->8, (T=24,sigma=0.5)->2, (T=48,sigma=0)->8,
(T=48,sigma=0.5)->8. Best single-window method: sigma=0 -> baseline+hybrid; sigma=0.5 ->
invobs+hybrid. Cycle counts: T=24 stride{2,4,8}->{9,5,3}; T=48 ->{21,11,6}.

**Two-dimensional Kolmogorov flow:** vorticity formulation, 64x64 grid, viscosity nu=1e-2, drag
alpha=0.1, forcing wavenumber k=4, outer step 0.18 (25 inner RK4 substeps), observe every 16th
point (4x4 = 16 of 4096 observed). Sliding-window run: WINDOW_T=10, forecast 15, ensemble N=8,
noise {0.0}; T_obs in {20,30}, strides {2,5,10}; best stride 10 for both T_obs; best single-window
method invobs+hybrid. Experiment-A final L-BFGS losses: invobs+hybrid 56,385; invobs+obs 45,760;
baseline+hybrid 168,341; baseline+obs 138,254 (invobs initialization starts the optimizer in a
far lower-loss basin). Warmup enstrophy rises from 0.50 to ~9.8 and is quasi-stationary
(last-20-step drift ~10.8%); training-trajectory vorticity range about [-12.3, 14.1].
The separate full-4D-Var Kolmogorov notebook uses T_assim=10, a 4x4 observation grid, 500 training
trajectories, and sweeps sigma_obs in {0, 0.1, 0.5}; its main usable figure here is the
training-trajectory vorticity panel.

## 7. Tone

Concise, technical, honest. This is a course capstone, not a conference submission: it is fine
(and expected) to report negative results, smoke-scale caveats (small ensembles, CPU-limited
Kolmogorov runs, inverter trained at one noise level then evaluated at another, short Lyapunov
time from the fixed integrator), and to separate "reproduced the paper" from "our extensions."
