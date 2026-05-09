# Prompt for the next Claude Code session

Paste everything below the line into a fresh Claude Code session run from this directory
(`C:\Users\ericx\Downloads\Spring 2026\AM 170B\invobs-data-assimilation`). The work is
phased — you can stop at the end of any phase, commit, and continue later. Phases 1-4 are
"reproduce the paper"; phases 5-8 are "extend beyond the paper."

---

You are continuing my AM 170B capstone — a PyTorch port and extension of Frerix et al.
2021 ([arXiv:2102.11192](https://arxiv.org/abs/2102.11192)), "Variational Data Assimilation
with a Learned Inverse Observation Operator." The paper's original JAX/Flax implementation
lives in `paper_scripts/`. My PyTorch port is in three notebooks at the repo root:

- `PyTorch_InvObs_DA.ipynb` — full port plus a $J_b + J_o$ 4D-Var noise sweep (paper does
  not run this).
- `PyTorch_InvObs_DA_WindowSweep.ipynb` — first window sweep $T \in \{1..100\}$ with
  gradient diagnostics. Has methodology issues.
- `PyTorch_InvObs_DA_WindowSweep_Corrected.ipynb` — cleaned-up version (shared trajectories,
  per-sample diagnostics, cross-window CNN generalization test).

`CLAUDE.md` and `AGENTS.md` describe the legacy JAX stack (`jax==0.2.6`, `flax==0.3.0`,
`jax_cfd==0.1.0`). **Do not try to run `paper_scripts/`** — read it for ground truth and
reimplement equivalent behavior in PyTorch.

A code review identified concrete reproducibility gaps. Your job is to fix them in the
order below. Confirm with me before moving from "reproduce" (phases 1-4) to "extend"
(phases 5-8).

## Two non-negotiable rules for every phase

### Rule 1 — Never modify the existing three notebooks.

`PyTorch_InvObs_DA.ipynb`, `PyTorch_InvObs_DA_WindowSweep.ipynb`, and
`PyTorch_InvObs_DA_WindowSweep_Corrected.ipynb` are frozen artifacts. They currently embed
hours of cached training runs and saved figures, and the review note in the vault
references their figures by cell index. Each phase below produces a **new notebook** at the
repo root with a `_v2_` prefix. New shared helper code goes in **new `.py` files** at the
repo root (e.g. `pytorch_paper_inverter.py`); do not edit `paper_scripts/`. Existing helper
code that the old notebooks define inline (e.g. `Lorenz96`, `decorrelate`, `lbfgs_minimize`)
should be **lifted into a new shared module** `pytorch_invobs_lib.py` so the new notebooks
import it instead of redefining it. Copy verbatim — do not refactor while moving.

### Rule 2 — Every new notebook must run unchanged in three environments.

The notebooks must work in all of:

1. **Local Jupyter / VS Code Jupyter kernel on this Windows machine** (no GPU, no Drive).
2. **Google Colab in the browser** (GPU optional, Drive available).
3. **VS Code Colab extension** ("Colab" by Google, publisher `googlecolab`) — connects
   VS Code to a Colab runtime, so `import google.colab` works the same as in the browser.

The existing three notebooks already do most of this with an `IN_COLAB` block and an
`INVOBS_CACHE_DIR` env var. Lift that pattern into a single `notebook_setup.py` helper and
make every new notebook use it (see "Setup boilerplate" below). Test on at least the local
VS Code Jupyter kernel before declaring a phase done; flag explicitly which environments
you actually verified.

## Setup boilerplate (every new notebook starts with these cells)

Create a new file `notebook_setup.py` at the repo root:

```python
# notebook_setup.py
"""Environment detection + cache + dependency install helpers shared by all v2 notebooks.

Works in local Jupyter, Google Colab (browser), and VS Code Colab extension.
"""
import importlib
import os
import subprocess
import sys


def detect_colab() -> bool:
    """True for Google Colab runtimes, including VS Code Colab extension sessions."""
    try:
        importlib.import_module('google.colab')
        return True
    except ImportError:
        return False


def pip_install(*pkgs, quiet: bool = True) -> None:
    args = [sys.executable, '-m', 'pip', 'install']
    if quiet:
        args.append('-q')
    args.extend(pkgs)
    subprocess.check_call(args)


def ensure_packages(packages: dict) -> None:
    """packages: {import_name: pip_spec}. Install only missing ones."""
    missing = []
    for import_name, pip_spec in packages.items():
        try:
            importlib.import_module(import_name)
        except ImportError:
            missing.append(pip_spec)
    if missing:
        print(f'Installing: {missing}')
        pip_install(*missing)


def setup_cache(default_local: str = '~/invobs_cache') -> str:
    """Return cache directory path. Mount Drive on Colab if needed.

    Resolution order:
      1. INVOBS_CACHE_DIR env var (highest priority, used by all environments).
      2. /content/drive/MyDrive/invobs_cache on Colab.
      3. ~/invobs_cache locally.
    """
    in_colab = detect_colab()
    env_override = os.environ.get('INVOBS_CACHE_DIR')

    if env_override:
        cache_dir = env_override
    elif in_colab:
        cache_dir = '/content/drive/MyDrive/invobs_cache'
    else:
        cache_dir = os.path.expanduser(default_local)

    if in_colab and cache_dir.startswith('/content/drive') and not os.path.ismount('/content/drive'):
        from google.colab import drive  # type: ignore
        drive.mount('/content/drive')

    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def setup_device():
    """Return a torch.device, preferring CUDA when available."""
    import torch
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def banner(cache_dir: str, device, in_colab: bool) -> None:
    print(f'environment: {"colab" if in_colab else "local"}')
    print(f'device:      {device}')
    print(f'cache dir:   {cache_dir}')
```

Then every new notebook starts with these three cells (paste verbatim, do not duplicate or
inline this logic — import it):

```python
# Cell 1 — package install (idempotent; runs each session because Colab loses state)
import importlib, os, subprocess, sys

# Bootstrap: clone setup helper into Colab if needed.
if 'google.colab' in sys.modules and not os.path.exists('notebook_setup.py'):
    raise RuntimeError(
        'notebook_setup.py is missing. Either upload it to /content or clone the repo into '
        '/content with `!git clone <repo> /content/invobs-data-assimilation && %cd /content/...`'
    )

import notebook_setup as ns

ns.ensure_packages({
    'torch': 'torch',
    'numpy': 'numpy',
    'matplotlib': 'matplotlib',
    'pandas': 'pandas',
    # Phase-specific extras get added here per notebook (e.g. 'torchdiffeq': 'torchdiffeq').
})
```

```python
# Cell 2 — environment + cache + device
import torch
import numpy as np
import matplotlib.pyplot as plt

IN_COLAB  = ns.detect_colab()
CACHE_DIR = ns.setup_cache()
device    = ns.setup_device()
ns.banner(CACHE_DIR, device, IN_COLAB)

torch.set_default_dtype(torch.float32)
torch.manual_seed(0)
np.random.seed(0)
```

```python
# Cell 3 — shared library import
from pytorch_invobs_lib import (
    Lorenz96, generate_data, estimate_correlation,
    decorrelate, correlate, lbfgs_minimize,
    InverseObsLorenz96,         # original residual-stack port
    PeriodicSpaceConv2d,
    save_cache, load_cache,
)
# Phase-specific imports go below this line per notebook.
```

`pytorch_invobs_lib.py` should expose:

- `Lorenz96` (with `integrate`, `warmup`, `step`, `observe`, and a new `integrate_adaptive`
  added in phase 3).
- `generate_data`, `estimate_correlation`.
- `decorrelate`, `correlate`.
- `lbfgs_minimize` and `make_da_loss` (lift from the corrected notebook).
- `InverseObsLorenz96` and `PeriodicSpaceConv2d` (the original port — keep available).
- `baseline_init_l96` (the existing `repeat_interleave` version) and
  `baseline_init_l96_paper` (added in phase 2).
- `save_cache(obj, name, cache_dir)` and `load_cache(name, cache_dir, force=False)`.
  Pass `cache_dir` explicitly so this module has no hidden globals.

### VS Code Colab extension specifics

The "Colab" extension by Google (publisher `googlecolab`) connects a VS Code instance to a
real Colab runtime. From your point of view as the agent, the runtime *is* a Colab runtime:
`import google.colab` works, `/content` is the cwd, Drive mount works the same way. The
boilerplate above already handles this — no extra code needed.

A note for the user (don't include in the notebook): to launch with this extension,
install "Colab" from the VS Code marketplace, open the notebook in VS Code, click the
kernel picker in the top-right, choose "Connect to Colab," then run cells normally. The
`drive.mount` call will pop a browser auth window the same as in the Colab UI.

---

## Phase 1 — Port the paper's exact inverse-observation CNN

**Goal:** add a faithful PyTorch translation of the paper's `ObservationInverterLorenz96`
so future result differences can be attributed to method choices, not architecture choices.

**Read first:**
- `paper_scripts/lorenz96_ml.py`
- `paper_scripts/lorenz96_methods.py::interpolate_periodic_lorenz96`
- The existing `InverseObsLorenz96` class.

**Files to create:**
- `pytorch_paper_inverter.py` — at repo root. Contains:
  - `class CubicPeriodicUpsample1d(nn.Module)` — cubic periodic upsampling along one axis,
    mirroring the JAX version (pad with `mode='circular'` before resize, crop after). Add a
    docstring noting `F.interpolate` is bicubic 2D so you must transpose-or-reshape to feed
    it.
  - `class ObservationInverterLorenz96Paper(nn.Module)` — the faithful port:
    - Activation **SiLU**.
    - Channel widths `[128, 64, 32, 16]`, four blocks.
    - **`BatchNorm` after every conv.**
    - Upsampling at factors `(1, 2, 2, 1)` between blocks.
    - No skip connections.
- `tests/test_paper_inverter.py` — pytest suite confirming:
  - Constant input → constant output (within float tolerance).
  - Low-frequency sinusoid `sin(2π k x / N)` with `k <= N/4` is preserved by the upsampler.
  - Periodicity is exact: `up(x)[..., 0]` matches the wrap of `up(x)[..., -1]`.
- `PyTorch_InvObs_DA_v2_PaperFaithful.ipynb` — new notebook at repo root. Setup
  boilerplate cells 1-3, then:
  - Train both inverters (`InverseObsLorenz96` and `ObservationInverterLorenz96Paper`) at
    `obs_noise_std=0.0`, `n_train=32000`, `T_train=20`, `n_epochs=500`, `batch_size=256`.
  - Cache keys `l96_inverter_PORT_sigma0.0_n32000_ep500.pt` (rename existing) and
    `l96_inverter_PAPER_sigma0.0_n32000_ep500.pt` (new).
  - Plot both training curves on the same axis.
  - Eval on a held-out 100-trajectory batch (seed 100): test L1 reconstruction + `x0` RMSE.

**Acceptance criteria:**
- `pytest tests/test_paper_inverter.py` passes.
- Both inverters train without NaN. Paper inverter's test L1 should be within ~30% of the
  port's; if it's catastrophically worse, the cubic upsample or BatchNorm placement is
  wrong — debug before moving on.
- Notebook runs end-to-end on local VS Code Jupyter kernel. Verified on Colab is a bonus,
  not a blocker (if the user has Colab access, run there too).

---

## Phase 2 — Add the paper's baseline initialization

**Goal:** match `average_da_init_lorenz96` so "baseline init" is a fair prior.

**Read first:** `paper_scripts/lorenz96_methods.py::average_da_init_lorenz96`. The recipe:
fill all grid points with the dataset mean of `X0`, then overwrite observed positions with
`Y[:, 0]`.

**Files to modify / create:**
- `pytorch_invobs_lib.py` — add `baseline_init_l96_paper(dyn_sys, Y, X0_mean)` next to the
  existing function. Do **not** delete the old one — `_v1_` notebooks still depend on it.
- Add `estimate_climatological_mean(dyn_sys, n_samples, n_warmup, seed)` returning the
  per-grid-point mean over a long warmed-up ensemble. Cache it (`l96_x0_mean.pt`).
- `PyTorch_InvObs_DA_v2_PaperFaithful.ipynb` (extend the same phase 1 notebook):
  - Add a `BASELINE_INIT` config flag with values `'paper'` and `'repeat'`.
  - Run the noise-sweep bar chart **for both** baselines so the figure shows the size of
    the prior effect.
  - Save the figure to `figures/v2_baseline_comparison.png`.

**Acceptance criteria:**
- For `sigma_obs = 0`, `T = 10`: paper-baseline `x0` RMSE on 100 trajectories is noticeably
  lower than repeat-baseline RMSE.
- The Hovmöller comparison from the original notebook 1 (cell `fe17b0e0`) reimplemented in
  the new notebook now shows "paper-baseline + obs" visibly closer to truth than
  "repeat-baseline + obs."

---

## Phase 3 — Diagnose / replace the integrator

**Goal:** quantify fixed-RK4 vs adaptive `odeint` divergence and offer the adaptive option
for fair paper comparison.

**Read first:**
- `paper_scripts/dynamical_system.py::Lorenz96.integrate` — `jax.experimental.ode.odeint`,
  `dt=0.1` is **output-spacing** (variable internal step).
- The existing `Lorenz96` class — fixed RK4, `dt=0.01`, 10 inner steps per outer.

**Files to create:**
- `requirements.txt` — append `torchdiffeq` (no version pin yet; pin once you know what's
  on the local box).
- `pytorch_invobs_lib.py::Lorenz96` — add `integrate_adaptive(self, x0, n_steps, rtol=1e-7,
  atol=1e-9)` using `torchdiffeq.odeint(method='dopri5')`. Keep the existing `integrate`
  method untouched.
- `PyTorch_InvObs_DA_v2_Integrator.ipynb` — new diagnostic notebook. Setup boilerplate
  (add `'torchdiffeq': 'torchdiffeq'` to the package dict in cell 1), then:
  - Pick one warmed-up `x0`, integrate 200 outer steps with both methods.
  - Plot `||x_RK4(t) - x_adaptive(t)||` vs `t` on a log axis.
  - Report `t*` = first time the divergence reaches 1.0.
  - Repeat from 5 different `x0` to get a confidence band.
  - Save figure `figures/v2_integrator_divergence.png`.
  - Also add a Lyapunov-time measurement cell (see cross-cutting tasks).

**Acceptance criteria:**
- Two integrators agree to machine precision for `t` ≪ Lyapunov time.
- They diverge at chaotic rate (slope $\approx \lambda$ on log axis) afterwards.
- Report `t* / outer_dt` — this sets the ceiling on how long an assimilation window can be
  before the two simulators are doing different physics.
- Notebook runs on local VS Code Jupyter kernel (the one Eric will use most).

---

## Phase 4 — Sweep $\sigma_b$ in the full 4D-Var bar chart

**Goal:** stop comparing tuned and untuned methods.

**Files to modify / create:**
- `PyTorch_InvObs_DA_v2_PaperFaithful.ipynb` — extend with a new section:
  - `SIGMA_B_GRID = [0.1, 0.3, 1.0, 3.0, 10.0]`.
  - For each (init × mode × noise) cell, sweep $\sigma_b$ and record the lowest-RMSE
    choice.
  - Replace the old single-`sigma_b=1.0` bar chart with a new panel-of-bars, one panel per
    noise level, each cell showing the *tuned* result with chosen $\sigma_b$ annotated.
  - Cache key includes `sigmab` so existing fixed-`sigma_b=1` results are not clobbered.
  - Save figure `figures/v2_4dvar_tuned_sigmab.png`.

**Acceptance criteria:**
- "invobs + hybrid" cell's optimal $\sigma_b$ is smaller than "baseline + obs" cell's.
- Tuned bars show smaller absolute RMSE than the previous untuned bars in every cell.
- After phases 1+2+4 complete: the bar chart at `sigma_obs = 0.5` should be within ~20% of
  the paper's Figure 4 Lorenz96 panel. If not, the gap is either a remaining hyperparameter
  choice or a real bug — check in with the user before diving into phase 5.

---

## Phase 5 — Implement cycling (sliding-window) 4D-Var

**Goal:** the operational fix for gradient explosion. Demonstrate that many short cycles
beat a single long window at the same total time.

**Background:** vault note `Gradient Explosion in 4D-Var.md`; paper Section 3.

**Files to create:**
- `PyTorch_InvObs_DA_v2_Cycling.ipynb` — new notebook, setup boilerplate, then:
  ```python
  T_cycle  = 10
  N_cycles = 5
  T_total  = T_cycle * N_cycles  # 50

  x_b = climatological_mean   # or invobs of the first window
  for k in range(N_cycles):
      Y_k = Y[:, k*T_cycle : (k+1)*T_cycle]
      x_0_k = run_4dvar(x_b, Y_k, T_cycle, sigma_b=...)
      x_b   = M(x_0_k, T_cycle)   # forecast becomes next background
  forecast = M(x_0_k, T_FORECAST)
  ```
- Compare three setups on the same shared trajectories:
  1. Single window of length `T_total = 50`, baseline init.
  2. Single window of length `T_total = 50`, invobs init + hybrid opt.
  3. Cycling with `T_cycle = 10`, `N_cycles = 5`, both init choices.
- Forecast L1 vs lead-time plot; save `figures/v2_cycling_forecast.png`.

**Acceptance criteria:**
- Cycling produces a finite stable result where single-window-50 hits the gradient
  explosion threshold.
- Forecast skill at lead 10 is better for cycling than either single-window run.

---

## Phase 6 — Kolmogorov flow port (deferred)

**Goal:** port the 2D half. Same machinery, ~10× compute.

**Read first:** `paper_scripts/{kolmogorov_methods.py, kolmogorov_ml.py, dynamical_system.py::KolmogorovFlow}`.

**Major decision:** depend on `jax_cfd==0.1.0` (legacy stack) vs reimplement the
Navier-Stokes step in PyTorch (~200 LOC). Recommended: reimplement; lift operators from
phiflow's PyTorch backend if you want a head start.

**Files to create:**
- `pytorch_kolmogorov.py` — `KolmogorovFlow` class with `integrate`, `observe`, `warmup`.
- `PyTorch_InvObs_DA_v2_Kolmogorov.ipynb` — new notebook with phases 1+2+4 ported to 2D
  (smaller defaults: 16 trajectories, $T = 5$).

**Acceptance criteria:**
- 50-step trajectory from a filtered random init is energy-conserving to within expected
  viscous decay.
- Enstrophy spectrum at $t=5$ matches paper Figure 5 visually.
- Inverter trains and recovers $x_0$ on a 16-trajectory test set.

---

## Phase 7 — Correlated $R$

**Goal:** AR(1) correlation along the spatial axis.

**Files to create:**
- `PyTorch_InvObs_DA_v2_CorrelatedR.ipynb` — extends phase 4's tuned 4D-Var.
- AR(1) covariance with $\rho \in \{0.1, 0.5, 0.9\}$. Use closed-form tridiagonal inverse
  for `R_inv`.
- New bar chart with one panel per $\rho$.

**Acceptance criterion:** "invobs + hybrid" should widen its lead at high $\rho$ — diagonal
$R$ in the paper made obs-only over-trust correlated observations.

---

## Phase 8 — Ensemble background covariance (EnVar)

**Goal:** replace climatological $C$ with an ensemble-sampled $B_k$ at each cycle.

**Files to create:**
- `PyTorch_InvObs_DA_v2_EnVar.ipynb` — extends phase 5's cycling notebook.
- At the start of each window, integrate $N_{ens}=32$ perturbations of the current
  background and use their covariance as $B_k$.

**Acceptance criterion:** forecast skill improves in windows immediately following a
model-error event (where climatological $C$ is misspecified).

---

## Cross-cutting tasks (alongside the phases)

- **Lyapunov-time measurement** (in `PyTorch_InvObs_DA_v2_Integrator.ipynb`): integrate two
  trajectories from `x_0` and `x_0 + 1e-6 * unit_vector` for 200 outer steps, plot
  $\log\|x^{(1)}_t - x^{(2)}_t\|$ vs `t`, read off the slope $\lambda$, report
  $T_L = 1/\lambda$. Expect ~0.6 model time units for $F=8$. Discrepancy means the
  integrator (phase 3) is suspect.
- **Re-run the corrected window sweep with `RUN_PROFILE='final'`** on a smaller subset
  (`ASSIM_WINDOWS=[5, 10, 20]`) after phase 1 lands to confirm hybrid does converge cleanly
  when the optimizer isn't truncated. Save `figures/v2_window_sweep_final.png`. Do this in
  a new notebook `PyTorch_InvObs_DA_v2_WindowSweepFinal.ipynb`, **do not** edit the
  existing corrected notebook.
- **`tests/test_data_assimilation.py` is broken** (imports `from data_assimilation` but the
  module is `run_data_assimilation`). Fix the import or skip the file with an explanatory
  comment. Do not silently rename the module.
- **Update `CLAUDE.md` and `AGENTS.md`** at the very end with a new "PyTorch port v2"
  section pointing at the new notebooks and helper modules. Don't touch other sections.
- **Write a top-level `README_v2.md`** at the end summarizing what's new, where each phase
  lives, what cache keys it produced, and how to run each notebook in each of the three
  environments. Reference this prompt file (`INSTRUCTIONS_FROM_REVIEW.md`) so the
  provenance is clear.

## Definition of done

- Phases 1-4 complete: `PyTorch_InvObs_DA_v2_PaperFaithful.ipynb` + `_Integrator.ipynb`
  rebuild end-to-end on a fresh cache and the bar chart at `sigma_obs = 0.5` is within
  ~20% of paper Figure 4 (Lorenz96 panel). If not, the remaining gap is either a
  hyperparameter or a bug — write up what's left and stop.
- All v2 notebooks run unchanged on local VS Code Jupyter and (verified) on at least one
  Colab environment (browser or VS Code Colab extension).
- Phases 5-8 are research extensions; do them only after the user confirms phases 1-4 are
  on the rails.

Report progress at the end of each phase: what changed, which file/notebook to look at,
which environments you verified, and what cache keys it produced. Do not conflate phases —
do not start phase 2 before phase 1's acceptance criteria are met, and do not start the
"extend" half (5-8) without checking in.
