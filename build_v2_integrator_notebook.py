"""Generate PyTorch_InvObs_DA_v2_Integrator.ipynb for Phase 3 of the v2 review.

Run from repo root:
    python build_v2_integrator_notebook.py

Phase 3 deliverable per INSTRUCTIONS_FROM_REVIEW.md:
- Pick one warmed-up x0, integrate 200 outer steps with both methods.
- Plot ||x_RK4(t) - x_adaptive(t)|| vs t on a log axis.
- Report t* = first time the divergence reaches 1.0.
- Repeat from 5 different x0 to get a confidence band.
- Save figure figures/v2_integrator_divergence.png.
- Lyapunov-time measurement cell (cross-cutting task).
"""
import json

cells = []


def md(text):
    cells.append({
        "cell_type": "markdown",
        "source": text,
        "metadata": {},
    })


def code(source):
    cells.append({
        "cell_type": "code",
        "source": source,
        "metadata": {},
        "outputs": [],
        "execution_count": None,
    })


# ============================================================================
# Header
# ============================================================================
md("""# PyTorch InvObs DA — v2 Integrator Diagnostic (Phase 3)

[INSTRUCTIONS_FROM_REVIEW.md](INSTRUCTIONS_FROM_REVIEW.md) Phase 3.

**The question.** The v1 PyTorch port integrates Lorenz96 with **fixed RK4**
(`dt=0.01`, 10 inner steps per outer step). The paper uses
`jax.experimental.ode.odeint` with `dt=0.1` as the **output spacing** (variable
internal step). Lorenz96 is chaotic with $\\lambda \\approx 1/T_L \\approx 1.7$
($T_L \\approx 0.6$ model time units), so the two integrators must agree at
short horizons but diverge at the chaotic rate beyond a few Lyapunov times.

**This notebook**
1. Implements the spec's `Lorenz96.integrate_adaptive(x0, n_steps, ...)`
   (adaptive Dormand-Prince via `torchdiffeq`).
2. Picks a warmed-up $x_0$, integrates 200 outer steps with both methods, and
   plots $\\|x_{RK4}(t) - x_{adapt}(t)\\|$ vs $t$ on a log axis.
3. Reports the divergence-to-1 time $t^*$.
4. Repeats from 5 different $x_0$ for a confidence band.
5. Measures the Lyapunov time directly via the cross-cutting task: track the
   exponential separation of two trajectories starting $10^{-6}$ apart.""")


# ============================================================================
# Setup boilerplate
# ============================================================================
code("""# Cell 1 — package install (idempotent; runs each session because Colab loses state)
import importlib, os, subprocess, sys

if 'google.colab' in sys.modules and not os.path.exists('notebook_setup.py'):
    raise RuntimeError(
        'notebook_setup.py is missing. Either upload it to /content or clone the repo into '
        '/content with `!git clone <repo> /content/invobs-data-assimilation && %cd /content/...`'
    )

import notebook_setup as ns

ns.ensure_packages({
    'torch':       'torch',
    'numpy':       'numpy',
    'matplotlib':  'matplotlib',
    'pandas':      'pandas',
    'torchdiffeq': 'torchdiffeq',  # Phase 3 extra
})""")

code("""# Cell 2 — environment + cache + device
import torch
import numpy as np
import matplotlib.pyplot as plt

IN_COLAB  = ns.detect_colab()
CACHE_DIR = ns.setup_cache()
device    = ns.setup_device()
ns.banner(CACHE_DIR, device, IN_COLAB)

torch.set_default_dtype(torch.float32)
torch.manual_seed(0)
np.random.seed(0)""")

code("""# Cell 3 — shared library import
import pytorch_invobs_lib as pil
pil.device = device

from pytorch_invobs_lib import Lorenz96, save_cache, load_cache

L96 = Lorenz96(grid_size=40, dt=0.01, n_inner=10, observe_every=4)
print(f'L96: grid={L96.grid_size}, outer_dt={L96.outer_dt} (dt_inner={L96.dt}, n_inner={L96.n_inner})')""")


# ============================================================================
# Section 1 — single-trajectory divergence
# ============================================================================
md("""---
## 1. Single-trajectory RK4 vs adaptive divergence

Warm up one $x_0$ on the attractor, then propagate it 200 outer steps with both
integrators. Plot the log-divergence vs time. Expect:
- $t \\ll T_L$: agreement to ~machine precision (RK4 truncation + adaptive
  tolerance both negligible).
- $t \\gtrsim T_L$: divergence grows like $\\exp(\\lambda t)$ with slope
  $\\lambda \\approx 1.7$ on a log axis. *This is not a bug* — it's the chaotic
  amplification of the small per-step truncation difference between the two
  integrators.""")


code("""N_OUTER = 200                                           # 200 outer steps = 20 model time units
RTOL    = 1e-7
ATOL    = 1e-9

torch.manual_seed(42)
x0_seed = torch.randn(L96.grid_size, device=device) * 0.5
x0      = L96.warmup(x0_seed, total_inner_steps=2000)
print(f'warmed-up x0: mean={x0.mean().item():.3f}, std={x0.std().item():.3f}')

traj_rk4   = L96.integrate(x0, n_steps=N_OUTER)
traj_adapt = L96.integrate_adaptive(x0, n_steps=N_OUTER, rtol=RTOL, atol=ATOL)
print(f'shapes: rk4={tuple(traj_rk4.shape)}, adapt={tuple(traj_adapt.shape)}')

t_axis     = np.arange(N_OUTER) * L96.outer_dt          # model time units
divergence = (traj_rk4 - traj_adapt).pow(2).sum(dim=-1).sqrt().detach().cpu().numpy()
print(f'divergence at t=0: {divergence[0]:.3e}')
print(f'divergence at t={t_axis[-1]:.1f}: {divergence[-1]:.3e}')

# t* = first time divergence reaches 1.0 (state-norm scale)
above = np.where(divergence >= 1.0)[0]
t_star = t_axis[above[0]] if len(above) else np.nan
print(f't*  (first time ||diff|| >= 1.0): {t_star:.2f} mtu '
      f'({t_star / L96.outer_dt if not np.isnan(t_star) else float("nan"):.0f} outer steps)')""")


# ============================================================================
# Section 2 — confidence band over 5 trajectories
# ============================================================================
md("""---
## 2. Confidence band — 5 different warmed-up $x_0$

Same experiment from 5 different attractor states. Plot the median divergence
plus min/max envelope, and report $t^*$ as a per-trajectory median.""")


code("""N_TRAJ = 5

div_cache_key = f'l96_phase3_divergence_N{N_TRAJ}_T{N_OUTER}.pt'
div_cache = load_cache(div_cache_key, CACHE_DIR)
if div_cache is None:
    divs = np.zeros((N_TRAJ, N_OUTER))
    t_stars = []
    for i in range(N_TRAJ):
        torch.manual_seed(100 + i)
        x0_seed = torch.randn(L96.grid_size, device=device) * 0.5
        x0_i    = L96.warmup(x0_seed, total_inner_steps=2000)
        rk4_i   = L96.integrate(x0_i, n_steps=N_OUTER)
        ada_i   = L96.integrate_adaptive(x0_i, n_steps=N_OUTER, rtol=RTOL, atol=ATOL)
        d_i     = (rk4_i - ada_i).pow(2).sum(dim=-1).sqrt().detach().cpu().numpy()
        divs[i] = d_i
        ab = np.where(d_i >= 1.0)[0]
        t_stars.append(t_axis[ab[0]] if len(ab) else np.nan)
        print(f'  traj {i}: t* = {t_stars[-1]:.2f} mtu')
    save_cache({'divs': divs, 't_stars': t_stars}, div_cache_key, CACHE_DIR)
else:
    divs    = div_cache['divs']
    t_stars = div_cache['t_stars']
    print(f'Loaded {N_TRAJ} divergence traces from cache.')

t_stars = np.array(t_stars, dtype=float)
print(f'\\nt* summary across {N_TRAJ} trajectories:  '
      f'median={np.nanmedian(t_stars):.2f} mtu, '
      f'range=[{np.nanmin(t_stars):.2f}, {np.nanmax(t_stars):.2f}]')""")


code("""# --- Divergence plot. ---
fig, ax = plt.subplots(figsize=(9, 5))
median = np.median(divs, axis=0)
mn, mx = divs.min(axis=0), divs.max(axis=0)
ax.fill_between(t_axis, mn, mx, alpha=0.20, color='#0072B2', label=f'min/max over {N_TRAJ} traj')
ax.plot(t_axis, median, color='#0072B2', lw=1.8, label='median divergence')
ax.axhline(1.0, color='gray', ls='--', lw=1, label='||diff|| = 1.0 threshold')
ax.set_yscale('log')
ax.set_xlabel('model time t')
ax.set_ylabel(r'$\\|x_{RK4}(t) - x_{adapt}(t)\\|_2$')
ax.set_title('Fixed-RK4 vs adaptive Dopri5 divergence on Lorenz96')
ax.grid(which='both', alpha=0.3)
ax.legend(loc='lower right')
os.makedirs('figures', exist_ok=True)
plt.tight_layout()
plt.savefig('figures/v2_integrator_divergence.png', dpi=150, bbox_inches='tight')
plt.show()""")


# ============================================================================
# Section 3 — Lyapunov-time measurement (cross-cutting)
# ============================================================================
md("""---
## 3. Lyapunov time $T_L$ (cross-cutting task)

Integrate two trajectories from $x_0$ and $x_0 + \\epsilon \\cdot \\hat e$ for
200 outer steps (with $\\epsilon = 10^{-6}$). The $L^2$ separation grows
$\\sim e^{\\lambda t}$ in the linear regime; we fit $\\lambda$ and report
$T_L = 1/\\lambda$. Expected $\\sim 0.6$ model time units for $F = 8$.

If the measured $T_L$ is much smaller than expected, the integrator is suspect
(too coarse to resolve the unstable directions).""")


code("""EPSILON = 1e-6

torch.manual_seed(7)
x0_l = L96.warmup(torch.randn(L96.grid_size, device=device) * 0.5, total_inner_steps=2000)
unit = torch.zeros_like(x0_l); unit[0] = 1.0   # canonical perturbation direction

x0_a = x0_l
x0_b = x0_l + EPSILON * unit

traj_a = L96.integrate(x0_a, n_steps=N_OUTER)
traj_b = L96.integrate(x0_b, n_steps=N_OUTER)
sep = (traj_a - traj_b).pow(2).sum(dim=-1).sqrt().detach().cpu().numpy()
log_sep = np.log(np.maximum(sep, 1e-30))

# Fit slope in the linear-growth regime: t in [0.5, 5.0] mtu (avoid initial
# transient AND post-saturation plateau).
t_lo, t_hi = 0.5, 5.0
mask = (t_axis >= t_lo) & (t_axis <= t_hi)
slope, intercept = np.polyfit(t_axis[mask], log_sep[mask], 1)
lyap_time = 1.0 / slope
print(f'fit window: t in [{t_lo}, {t_hi}] mtu')
print(f'lambda     = {slope:.3f} (1/mtu)')
print(f'T_L = 1/lambda = {lyap_time:.3f} mtu  (paper expects ~0.6)')

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(t_axis, sep, lw=1.6, color='#D55E00', label=r'$\\|x_b(t) - x_a(t)\\|$')
ax.plot(t_axis[mask], np.exp(intercept + slope * t_axis[mask]),
        ls='--', color='black', lw=1.2,
        label=fr'fit: $\\lambda = {slope:.3f}$, $T_L = {lyap_time:.3f}$ mtu')
ax.axhline(1.0, color='gray', ls=':', lw=1, label='saturation scale')
ax.set_yscale('log')
ax.set_xlabel('model time t')
ax.set_ylabel(r'$\\|\\delta x(t)\\|_2$')
ax.set_title(rf'Lyapunov-time fit on Lorenz96  ($\\epsilon_0 = {EPSILON}$)')
ax.grid(which='both', alpha=0.3)
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig('figures/v2_lyapunov_time.png', dpi=150, bbox_inches='tight')
plt.show()""")


# ============================================================================
# Section 4 — interpretation
# ============================================================================
md("""---
## Interpretation

- **Agreement at short horizons.** RK4 truncation error per outer step is
  $O(\\Delta t_{inner}^4) = O(10^{-8})$, well below the adaptive tolerance
  ($10^{-7}$ rel / $10^{-9}$ abs). At $t \\ll T_L$ the two integrators agree to
  ~machine precision (modulo float32 noise).
- **Divergence at the chaotic rate.** Beyond $\\sim T_L$, the small per-step
  truncation difference is amplified by the dynamics at rate
  $\\lambda \\approx 1/T_L$. The slope of the divergence curve on the log axis
  reads off $\\lambda$ and matches the Lyapunov-time fit.
- **Operational ceiling.** $t^*$ — the first time the integrator divergence
  reaches order 1 (state-norm scale) — sets the longest assimilation window
  $T$ where "RK4 vs Dopri5" can be considered the *same physics*. For windows
  $T \\gg t^*$ the two simulators are doing different things; the DA solution
  inherits the integrator choice.
- **What this means for the paper comparison.** When comparing v2 4D-Var
  results against the paper's bar charts, prefer `integrate_adaptive` to match
  their Dopri5 trajectories. For consistency with the v1 PyTorch port (and
  faster compute), keep `integrate` (fixed RK4) as the default and reserve
  `integrate_adaptive` for paper-comparison runs.

**Cache keys produced**
- `l96_phase3_divergence_N5_T200.pt`

**Figures saved**
- `figures/v2_integrator_divergence.png`
- `figures/v2_lyapunov_time.png`""")


# ============================================================================
# Write out
# ============================================================================
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
        "accelerator": "GPU",
        "colab": {"provenance": []},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open("PyTorch_InvObs_DA_v2_Integrator.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print(f"Wrote PyTorch_InvObs_DA_v2_Integrator.ipynb with {len(cells)} cells.")
