"""Generate PyTorch_InvObs_DA_v2_PaperFaithful.ipynb for Phase 1 of the v2 review.

Run from repo root:
    python build_v2_paperfaithful_notebook.py

Phase 1 deliverable per INSTRUCTIONS_FROM_REVIEW.md:
- Train both inverters (InverseObsLorenz96 port and ObservationInverterLorenz96Paper)
  at obs_noise_std=0.0, n_train=32000, T_train=20, n_epochs=500, batch_size=256.
- Plot both training curves on the same axis.
- Eval on held-out 100-trajectory batch (seed 100): test L1 reconstruction + x0 RMSE.

Future phases (2 / 4 / cross-cutting) extend the same notebook.
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
md("""# PyTorch InvObs DA — v2 Paper-Faithful

Phase 1 of the post-review rebuild ([INSTRUCTIONS_FROM_REVIEW.md](INSTRUCTIONS_FROM_REVIEW.md)).

**What this notebook does**
1. Loads shared helpers from `pytorch_invobs_lib.py` (lifted verbatim from the
   v1 notebooks) and the paper-faithful inverter from `pytorch_paper_inverter.py`.
2. Trains both inverters end-to-end at $\\sigma_{obs} = 0$ on the same training
   data, with identical optimizer settings, and plots their training curves on
   one axis.
3. Evaluates both on a held-out 100-trajectory batch (seed 100) — reports test
   L1 reconstruction loss and $x_0$ RMSE.

**Architecture differences (port vs paper)**

| | Port (`InverseObsLorenz96`) | Paper (`ObservationInverterLorenz96Paper`) |
|---|---|---|
| Activation | GELU | SiLU |
| Channel widths | flat 32 (×6 blocks) | 128 → 64 → 32 → 16 (4 blocks) |
| Normalization | none | BatchNorm2d after every conv |
| Skip connections | residual | none |
| Spatial upsample | one-shot bilinear at start | cubic periodic at factors (1, 2, 2, 1) |

The paper inverter follows `paper_scripts/lorenz96_ml.py`. Future phases extend
this notebook with paper baseline init (Phase 2) and a $\\sigma_b$ sweep (Phase 4).""")


# ============================================================================
# Cell 1 — package install (matches the spec's setup boilerplate)
# ============================================================================
code("""# Cell 1 — package install (idempotent; runs each session because Colab loses state)
import importlib, os, subprocess, sys

# Bootstrap: clone setup helper into Colab if needed.
if 'google.colab' in sys.modules and not os.path.exists('notebook_setup.py'):
    raise RuntimeError(
        'notebook_setup.py is missing. Either upload it to /content or clone the repo:\\n'
        '  !git clone https://github.com/ericxu1070/invobs-data-assimilation.git /content/invobs-data-assimilation\\n'
        '  %cd /content/invobs-data-assimilation'
    )

import notebook_setup as ns

ns.ensure_packages({
    'torch': 'torch',
    'numpy': 'numpy',
    'matplotlib': 'matplotlib',
    'pandas': 'pandas',
})""")


# ============================================================================
# Cell 2 — environment + cache + device
# ============================================================================
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


# ============================================================================
# Cell 3 — shared library import
# ============================================================================
code("""# Cell 3 — shared library import
import pytorch_invobs_lib as pil
pil.device = device  # one-time override; lib functions read this on construction

from pytorch_invobs_lib import (
    Lorenz96, generate_data, estimate_correlation,
    decorrelate, correlate, lbfgs_minimize, make_da_loss,
    InverseObsLorenz96,         # original residual-stack port
    PeriodicSpaceConv2d,
    baseline_init_l96, invobs_init_l96,
    estimate_climatological_mean, baseline_init_l96_paper,
    var4d_cost_obs, var4d_cost_phys, run_4dvar_l96,
    save_cache, load_cache,
)

# Phase-1 specific imports
from pytorch_paper_inverter import (
    CubicPeriodicUpsample1d,
    ObservationInverterLorenz96Paper,
)

L96 = Lorenz96(grid_size=40, dt=0.01, n_inner=10, observe_every=4)  # outer_dt = 0.1
print(f'L96: grid={L96.grid_size}, observe_every={L96.observe_every}, outer_dt={L96.outer_dt}')""")


# ============================================================================
# Section: shared training infrastructure
# ============================================================================
md("""---
## Shared training data + harness

Both inverters train on the same `(X, Y)` dataset and the same optimizer settings.
The dataset is cached separately from the model checkpoints, so adding a new
inverter variant later only retrains the network — not the data.""")


code("""# Training-set parameters from INSTRUCTIONS_FROM_REVIEW.md (Phase 1).
N_TRAIN     = 32000
T_TRAIN     = 20
N_WARMUP    = 1000
N_EPOCHS    = 500
BATCH_SIZE  = 256
LR          = 1e-3
SIGMA_TRAIN = 0.0   # obs_noise_std for Phase 1: clean observations.

DATA_KEY = f'l96_train_data_n{N_TRAIN}_T{T_TRAIN}_warmup{N_WARMUP}_sigma{SIGMA_TRAIN}.pt'

data = load_cache(DATA_KEY, CACHE_DIR)
if data is None:
    print(f'Generating {N_TRAIN} training trajectories (sigma={SIGMA_TRAIN})...')
    _, X_train, Y_train, _ = generate_data(L96, N_TRAIN, T_TRAIN, N_WARMUP,
                                           obs_noise_std=SIGMA_TRAIN, seed=42)
    X_train = X_train.detach()
    Y_train = Y_train.detach()
    save_cache({'X': X_train, 'Y': Y_train}, DATA_KEY, CACHE_DIR)
else:
    X_train, Y_train = data['X'], data['Y']
    print(f'Loaded training set: X={tuple(X_train.shape)}, Y={tuple(Y_train.shape)}')""")


code("""def train_inverter_v2(model, X, Y, n_epochs=N_EPOCHS, batch_size=BATCH_SIZE, lr=LR,
                       label='inverter', log_every=25):
    \"\"\"Standard supervised training loop on (Y -> X). Returns per-epoch MSE history.\"\"\"
    import torch.nn.functional as F
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = X.shape[0]
    history = []
    for epoch in range(n_epochs):
        perm = torch.randperm(n, device=X.device)
        ep_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            y_b, x_b = Y[idx], X[idx]
            pred = model(y_b)
            loss = F.mse_loss(pred, x_b)
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item() * idx.numel()
        ep_loss /= n
        history.append(ep_loss)
        if epoch % log_every == 0 or epoch == n_epochs - 1:
            print(f'  [{label}] epoch {epoch:3d}  loss={ep_loss:.4f}')
    return history""")


# ============================================================================
# Section: train both inverters
# ============================================================================
md("""---
## Train both inverters

We train the original PyTorch port and the paper-faithful port to convergence at
$\\sigma_{obs} = 0$ with identical optimizer settings.

Cache keys (per the spec):
- `l96_inverter_PORT_sigma0.0_n32000_ep500.pt` — original residual-stack port.
- `l96_inverter_PAPER_sigma0.0_n32000_ep500.pt` — paper-faithful port.

If a cache entry exists the cell loads it and skips training. To retrain, delete
the file under `CACHE_DIR` (or pass `force=True` when calling `load_cache`).""")


code("""# --- Train the original PyTorch port (residual stack with GELU + bilinear upsample). ---
PORT_KEY = f'l96_inverter_PORT_sigma{SIGMA_TRAIN}_n{N_TRAIN}_ep{N_EPOCHS}.pt'

inverter_port = InverseObsLorenz96(obs_grid=10, full_grid=40, hidden=32, n_layers=6).to(device)
ckpt = load_cache(PORT_KEY, CACHE_DIR)
if ckpt is None:
    print(f'Training PORT inverter ({sum(p.numel() for p in inverter_port.parameters())} params)...')
    hist_port = train_inverter_v2(inverter_port, X_train, Y_train, label='PORT')
    save_cache({'state_dict': inverter_port.state_dict(), 'hist': hist_port},
               PORT_KEY, CACHE_DIR)
else:
    inverter_port.load_state_dict(ckpt['state_dict'])
    hist_port = ckpt['hist']
    print(f'Loaded PORT from cache: {len(hist_port)} epochs, final MSE = {hist_port[-1]:.4f}')""")


code("""# --- Train the paper-faithful port (SiLU + cubic upsample + BN, no skips). ---
PAPER_KEY = f'l96_inverter_PAPER_sigma{SIGMA_TRAIN}_n{N_TRAIN}_ep{N_EPOCHS}.pt'

inverter_paper = ObservationInverterLorenz96Paper(obs_grid=10, full_grid=40).to(device)
ckpt = load_cache(PAPER_KEY, CACHE_DIR)
if ckpt is None:
    print(f'Training PAPER inverter ({sum(p.numel() for p in inverter_paper.parameters())} params)...')
    hist_paper = train_inverter_v2(inverter_paper, X_train, Y_train, label='PAPER')
    save_cache({'state_dict': inverter_paper.state_dict(), 'hist': hist_paper},
               PAPER_KEY, CACHE_DIR)
else:
    inverter_paper.load_state_dict(ckpt['state_dict'])
    hist_paper = ckpt['hist']
    print(f'Loaded PAPER from cache: {len(hist_paper)} epochs, final MSE = {hist_paper[-1]:.4f}')""")


code("""# --- Side-by-side training curves on the same axis. ---
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(hist_port,  label=f'PORT   (final MSE = {hist_port[-1]:.4f})',  color='#0072B2', lw=1.6)
ax.plot(hist_paper, label=f'PAPER  (final MSE = {hist_paper[-1]:.4f})', color='#D55E00', lw=1.6)
ax.set_yscale('log')
ax.set_xlabel('epoch')
ax.set_ylabel('training MSE')
ax.set_title(f'Inverse-obs training curves @ sigma_obs={SIGMA_TRAIN}')
ax.grid(alpha=0.3)
ax.legend()
os.makedirs('figures', exist_ok=True)
plt.tight_layout()
plt.savefig('figures/v2_inverter_training_curves.png', dpi=150, bbox_inches='tight')
plt.show()""")


# ============================================================================
# Section: held-out evaluation
# ============================================================================
md("""---
## Held-out evaluation

100-trajectory batch with seed 100. Report:
- **Test L1 reconstruction**: mean $|H^{-1}_\\theta(Y) - X|$ across the trajectory.
- **$x_0$ RMSE**: per-sample RMSE of the recovered first frame against truth.

Acceptance criterion (from the review): the paper inverter's test L1 should be
within ~30% of the port's. If it's catastrophically worse, the cubic upsample or
BatchNorm placement is wrong — debug before moving on.""")


code("""# Held-out evaluation set.
N_EVAL_PHASE1 = 100
T_EVAL_PHASE1 = T_TRAIN  # match training horizon

X0_eval, X_eval, Y_eval, _ = generate_data(L96, n_samples=N_EVAL_PHASE1,
                                            n_time_steps=T_EVAL_PHASE1,
                                            n_warmup=N_WARMUP,
                                            obs_noise_std=SIGMA_TRAIN, seed=100)
print(f'Eval shapes: X={tuple(X_eval.shape)}, Y={tuple(Y_eval.shape)}')


def eval_inverter(model, label):
    model.eval()
    with torch.no_grad():
        X_pred = model(Y_eval)                                  # (N, T, grid)
    test_l1 = (X_pred - X_eval).abs().mean().item()
    x0_rmse_per_sample = ((X_pred[:, 0] - X0_eval) ** 2).mean(dim=-1).sqrt().cpu().numpy()
    print(f'  [{label:5s}] test L1 = {test_l1:.4f}   '
          f'x0 RMSE = {x0_rmse_per_sample.mean():.4f} +/- {x0_rmse_per_sample.std():.4f}')
    return dict(test_l1=test_l1,
                x0_rmse_mean=float(x0_rmse_per_sample.mean()),
                x0_rmse_std=float(x0_rmse_per_sample.std()),
                X_pred=X_pred.detach().cpu())

eval_port  = eval_inverter(inverter_port,  'PORT')
eval_paper = eval_inverter(inverter_paper, 'PAPER')

ratio = eval_paper['test_l1'] / eval_port['test_l1']
print(f'\\nPAPER/PORT test L1 ratio = {ratio:.3f}  '
      f'({"WITHIN 30%" if 0.7 <= ratio <= 1.3 else "OUTSIDE 30%"} of port)')""")


code("""# Visual sanity: per-sample x0 reconstruction for the first 4 trajectories.
fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
for i, ax in enumerate(axes):
    truth = X0_eval[i].cpu().numpy()
    rec_port  = eval_port['X_pred'][i, 0].numpy()
    rec_paper = eval_paper['X_pred'][i, 0].numpy()
    ax.plot(truth,     'k-', lw=2.0, label='truth' if i == 0 else None)
    ax.plot(rec_port,  '--', color='#0072B2', lw=1.2, label='PORT'  if i == 0 else None)
    ax.plot(rec_paper, ':',  color='#D55E00', lw=1.2, label='PAPER' if i == 0 else None)
    ax.set_ylabel(f'sample {i}')
axes[0].legend(fontsize=8, ncol=3, loc='upper right')
axes[-1].set_xlabel('grid index k')
plt.suptitle('x0 reconstruction on held-out evaluation set (seed 100)')
plt.tight_layout()
plt.savefig('figures/v2_x0_reconstruction.png', dpi=150, bbox_inches='tight')
plt.show()""")


# ============================================================================
# Section: Phase 1 summary
# ============================================================================
md("""---
## Phase 1 summary

| metric | PORT | PAPER |
|---|---|---|
| param count | (printed in train cell) | (printed in train cell) |
| final training MSE | (see plot legend) | (see plot legend) |
| test L1 (held-out 100 traj) | `eval_port['test_l1']` | `eval_paper['test_l1']` |
| $x_0$ RMSE | `eval_port['x0_rmse_mean']` | `eval_paper['x0_rmse_mean']` |

**Cache keys produced by this notebook**
- `l96_train_data_n32000_T20_warmup1000_sigma0.0.pt`
- `l96_inverter_PORT_sigma0.0_n32000_ep500.pt`
- `l96_inverter_PAPER_sigma0.0_n32000_ep500.pt`

**Figures saved**
- `figures/v2_inverter_training_curves.png`
- `figures/v2_x0_reconstruction.png`""")


# ============================================================================
# PHASE 2 — Paper baseline initialization
# ============================================================================
md("""---
# Phase 2 — Paper baseline initialization

The original (`repeat-interleave`) baseline copies each $t=0$ observation
across its block of $4$ unobserved grid points. The paper's baseline
([`average_da_init_lorenz96`](paper_scripts/lorenz96_methods.py)) instead
fills unobserved positions with the **climatological mean** $\\bar X$ of the
attractor and overwrites observed positions with $y_0$. This is a fairer
prior because it does not introduce step discontinuities at observation
boundaries.

This phase:
1. Estimates $\\bar X$ from a long warmed-up ensemble and caches it.
2. Verifies the acceptance criterion: at $\\sigma_{obs} = 0$, $T = 10$, the
   paper-baseline $x_0$ RMSE on 100 trajectories is noticeably lower than
   the repeat-baseline RMSE.
3. Compares both baselines through 4D-Var across noise levels (bar chart).
4. Plots a Hovmöller comparison: truth vs. each baseline + obs-mode 4D-Var.""")


code("""# --- Climatological mean (cached). ---
X0_MEAN_KEY = 'l96_x0_mean.pt'
ckpt = load_cache(X0_MEAN_KEY, CACHE_DIR)
if ckpt is None:
    print('Estimating climatological mean over 2000 warmed-up trajectories...')
    X0_mean = estimate_climatological_mean(L96, n_samples=2000, n_warmup=1000, seed=2)
    save_cache({'X0_mean': X0_mean}, X0_MEAN_KEY, CACHE_DIR)
else:
    X0_mean = ckpt['X0_mean']
print(f'X0_mean: shape={tuple(X0_mean.shape)}, mean={X0_mean.mean().item():.4f}, '
      f'std={X0_mean.std().item():.4f}')""")


code("""# --- Acceptance check: direct init RMSE comparison at sigma_obs=0, T=10. ---
T_PHASE2     = 10
N_EVAL_BASE  = 100

X0_truth, _, Y_clean, _ = generate_data(L96, n_samples=N_EVAL_BASE,
                                        n_time_steps=T_PHASE2, n_warmup=N_WARMUP,
                                        obs_noise_std=0.0, seed=200)

init_repeat = baseline_init_l96(L96, Y_clean)
init_paper  = baseline_init_l96_paper(L96, Y_clean, X0_mean)

rmse_repeat = ((init_repeat - X0_truth) ** 2).mean(dim=-1).sqrt()
rmse_paper  = ((init_paper  - X0_truth) ** 2).mean(dim=-1).sqrt()
print(f'sigma_obs=0, T=10, N=100 trajectories — direct init x0 RMSE:')
print(f'  repeat-baseline: {rmse_repeat.mean().item():.4f} +/- {rmse_repeat.std().item():.4f}')
print(f'  paper-baseline:  {rmse_paper.mean().item():.4f} +/- {rmse_paper.std().item():.4f}')
ratio = rmse_paper.mean().item() / rmse_repeat.mean().item()
print(f'  paper/repeat ratio = {ratio:.3f}  '
      f'({"PASS: paper noticeably lower" if ratio < 0.95 else "FAIL: not noticeably lower"})')""")


code("""# --- Noise-sweep bar chart of 4D-Var endpoints (sigma_b=1.0 fixed). ---
# Phase 4 will sweep sigma_b; this is the untuned baseline for context.
NOISE_LEVELS_PHASE2 = [0.1, 0.5, 1.0]
SIGMA_B_PHASE2      = 1.0
SIGMA_P_PHASE2      = 0.5

# We use the Phase-1 PAPER inverter for the hybrid mode. It is trained at
# sigma=0 only, so hybrid at noise > 0 is mismatched (sub-optimal). This is
# documented honestly; Phase 4 / future phases can train per-noise inverters.

phase2_combos = {
    'repeat + obs':    dict(init='repeat', mode='obs'),
    'paper  + obs':    dict(init='paper',  mode='obs'),
    'repeat + hybrid': dict(init='repeat', mode='hybrid'),
    'paper  + hybrid': dict(init='paper',  mode='hybrid'),
}

# Spatial correlation C (cached) — used for 4D-Var preconditioning in z-coords.
CORR_KEY = 'l96_corr.pt'
ckpt_corr = load_cache(CORR_KEY, CACHE_DIR)
if ckpt_corr is None:
    corr = estimate_correlation(L96, n_samples=1000, n_warmup=500)
    save_cache(corr, CORR_KEY, CACHE_DIR)
else:
    corr = ckpt_corr
print(f'C: shape={tuple(corr["C"].shape)}, '
      f'cond={torch.linalg.cond(corr["C"]).item():.3g}')

p2_cache_key = f'l96_phase2_baseline_sweep_N{N_EVAL_BASE}_T{T_PHASE2}.pt'
p2_cache = load_cache(p2_cache_key, CACHE_DIR)
if p2_cache is None:
    p2_results = {sigma: {} for sigma in NOISE_LEVELS_PHASE2}
    for sigma in NOISE_LEVELS_PHASE2:
        print(f'\\n=== sigma_obs = {sigma} ===')
        X0_gt, _, Y, _ = generate_data(L96, n_samples=N_EVAL_BASE,
                                       n_time_steps=T_PHASE2, n_warmup=N_WARMUP,
                                       obs_noise_std=sigma, seed=300)
        inits = {
            'repeat': baseline_init_l96(L96, Y),
            'paper':  baseline_init_l96_paper(L96, Y, X0_mean),
        }
        for name, cfg in phase2_combos.items():
            X0_init = inits[cfg['init']]
            X0_opt, _ = run_4dvar_l96(L96, inverter_paper, corr, X0_init, Y, T=T_PHASE2,
                                      sigma_b=SIGMA_B_PHASE2, sigma_obs=sigma, sigma_p=SIGMA_P_PHASE2,
                                      mode=cfg['mode'], physics_steps=200, obs_steps=300)
            rmse = ((X0_opt - X0_gt) ** 2).mean(dim=-1).sqrt().detach().cpu().numpy()
            p2_results[sigma][name] = rmse
            print(f'  {name:18s}  RMSE = {rmse.mean():.4f} +/- {rmse.std():.4f}')
    save_cache({'p2_results': p2_results}, p2_cache_key, CACHE_DIR)
else:
    p2_results = p2_cache['p2_results']
    print('Loaded Phase 2 results from cache.')""")


code("""# --- Bar chart: panel per noise level, baseline x mode bars. ---
phase2_names  = list(phase2_combos.keys())
phase2_colors = ['#999999', '#0072B2', '#D55E00', '#009E73']
phase2_hatches = ['', '//', '\\\\\\\\', 'xx']

fig, ax = plt.subplots(figsize=(10, 4.5))
xpos  = np.arange(len(NOISE_LEVELS_PHASE2))
width = 0.2

for i, name in enumerate(phase2_names):
    means = np.array([p2_results[s][name].mean() for s in NOISE_LEVELS_PHASE2])
    stds  = np.array([p2_results[s][name].std()  for s in NOISE_LEVELS_PHASE2])
    ax.bar(xpos + (i - 1.5) * width, means, width, yerr=stds, capsize=3,
           label=name, color=phase2_colors[i], hatch=phase2_hatches[i],
           edgecolor='black', linewidth=0.6)

ax.set_xticks(xpos)
ax.set_xticklabels([f'$\\\\sigma_{{obs}}={s}$' for s in NOISE_LEVELS_PHASE2])
ax.set_ylabel('state RMSE on $x_0$')
ax.set_title(f'Phase 2 baseline comparison (sigma_b = {SIGMA_B_PHASE2}, untuned)')
ax.legend(fontsize=9, loc='upper left')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
os.makedirs('figures', exist_ok=True)
plt.savefig('figures/v2_baseline_comparison.png', dpi=150, bbox_inches='tight')
plt.show()""")


code("""# --- Hovmöller comparison at sigma=0.5: truth vs each baseline + obs. ---
sigma_show = 0.5
T_HOV = 30
sample_idx = 0

X0_gt, _, Y_show, _ = generate_data(L96, n_samples=N_EVAL_BASE,
                                    n_time_steps=T_PHASE2, n_warmup=N_WARMUP,
                                    obs_noise_std=sigma_show, seed=300)

inits_show = {
    'repeat-baseline + obs': baseline_init_l96(L96, Y_show),
    'paper-baseline  + obs': baseline_init_l96_paper(L96, Y_show, X0_mean),
}
xrec_show = {}
for name, init in inits_show.items():
    X0_opt, _ = run_4dvar_l96(L96, inverter_paper, corr, init, Y_show, T=T_PHASE2,
                              sigma_b=SIGMA_B_PHASE2, sigma_obs=sigma_show, sigma_p=SIGMA_P_PHASE2,
                              mode='obs', physics_steps=0, obs_steps=300)
    xrec_show[name] = X0_opt.detach()

truth_traj = L96.integrate(X0_gt[sample_idx], T_HOV).detach().cpu().numpy()
panels = [('Truth', truth_traj.T)]
for name, X0_opt in xrec_show.items():
    pred = L96.integrate(X0_opt[sample_idx], T_HOV).detach().cpu().numpy()
    panels.append((name, pred.T))

vlim = max(np.abs(p).max() for _, p in panels)
fig, axes = plt.subplots(1, 3, figsize=(13, 3.2), sharex=True, sharey=True)
for ax, (name, data) in zip(axes, panels):
    im = ax.imshow(data, aspect='auto', cmap='RdBu_r', vmin=-vlim, vmax=vlim)
    ax.axvline(T_PHASE2 - 0.5, color='k', lw=0.8, ls='--')
    ax.set_title(name, fontsize=10)
    ax.set_xlabel('time step')
axes[0].set_ylabel('grid index k')
fig.colorbar(im, ax=axes, fraction=0.025, pad=0.01, label='$x_k$')
fig.suptitle(f'Hovmöller comparison at sigma_obs={sigma_show} '
             '(dashed = end of assimilation)', y=1.02)
plt.savefig('figures/v2_baseline_hovmoller.png', dpi=150, bbox_inches='tight')
plt.show()""")


# ============================================================================
# Phase 2 summary
# ============================================================================
md("""---
## Phase 2 summary

**What changed**
- New helpers in `pytorch_invobs_lib.py`: `estimate_climatological_mean`,
  `baseline_init_l96_paper` (existing `baseline_init_l96` is unchanged).
- 4D-Var driver and cost functions (`run_4dvar_l96`, `var4d_cost_obs`,
  `var4d_cost_phys`) are now also in the lib so Phase 4 can sweep $\\sigma_b$.
- This notebook gained an acceptance check, a baseline-comparison bar chart,
  and a Hovmöller comparison.

**Cache keys produced**
- `l96_x0_mean.pt`
- `l96_phase2_baseline_sweep_N100_T10.pt`

**Figures saved**
- `figures/v2_baseline_comparison.png`
- `figures/v2_baseline_hovmoller.png`

**Acceptance criteria (verify in the cells above)**
- ✅ For $\\sigma_{obs} = 0$, $T = 10$: paper-baseline $x_0$ RMSE noticeably
  lower than repeat-baseline (printed in the acceptance cell).
- The Hovmöller "paper-baseline + obs" panel should be visibly closer to truth
  than the "repeat-baseline + obs" panel.""")


# ============================================================================
# PHASE 4 — sigma_b sweep in the full 4D-Var bar chart
# ============================================================================
md("""---
# Phase 4 — Tune $\\sigma_b$ in the full 4D-Var bar chart

The earlier (Phase 2) bar chart fixed $\\sigma_b = 1.0$ for every cell. That is
not a fair comparison: each (init, mode, noise) combination has its own optimal
background weight. Here we sweep
$$\\sigma_b \\in \\{0.1,\\ 0.3,\\ 1.0,\\ 3.0,\\ 10.0\\}$$
for each combo and record the lowest-RMSE choice.

**Expected pattern** (per the spec):
- "invobs + hybrid" should choose a *smaller* optimal $\\sigma_b$ than
  "paper-baseline + obs" — its background is more trustworthy, so the cost
  weights it more heavily.
- Tuned bars should be lower than the Phase 2 untuned bars in every cell.

**Inverter caveat.** For fair "invobs + hybrid" comparison at $\\sigma_{obs} > 0$
we need an inverter trained at the matching observation noise (not the Phase 1
sigma=0 inverter). The next cell trains one PAPER inverter per $\\sigma_{obs}$
in the Phase 4 grid (cached).""")


code("""# --- Configuration. ---
SIGMA_B_GRID         = [0.1, 0.3, 1.0, 3.0, 10.0]
NOISE_LEVELS_PHASE4  = [0.1, 0.5, 1.0]
N_EVAL_PHASE4        = 16    # matches v1 bar chart's N_EVAL_4D
T_PHASE4             = 10
SIGMA_P_PHASE4       = 0.5

print(f'sigma_b grid:    {SIGMA_B_GRID}')
print(f'noise levels:    {NOISE_LEVELS_PHASE4}')
print(f'N eval:          {N_EVAL_PHASE4}')
print(f'T eval:          {T_PHASE4}')""")


code("""# --- Train PAPER inverter at each noise level in NOISE_LEVELS_PHASE4. ---
# Phase 1 already trained at sigma=0 (saved as inverter_paper); this section
# trains additional ones for fair invobs+hybrid evaluation at sigma_obs > 0.

def get_or_train_paper_inverter_at_sigma(sigma):
    \"\"\"Return a PAPER inverter trained at observation noise = sigma.

    For sigma == SIGMA_TRAIN (Phase 1's value, default 0.0) reuse the Phase 1
    inverter. Otherwise generate a (cached) noisy training set, train, and cache.
    \"\"\"
    if sigma == SIGMA_TRAIN:
        return inverter_paper

    data_key = f'l96_train_data_n{N_TRAIN}_T{T_TRAIN}_warmup{N_WARMUP}_sigma{sigma}.pt'
    data = load_cache(data_key, CACHE_DIR)
    if data is None:
        print(f'  [data sigma={sigma}] generating {N_TRAIN} noisy training trajectories...')
        _, X_s, Y_s, _ = generate_data(L96, N_TRAIN, T_TRAIN, N_WARMUP,
                                       obs_noise_std=sigma, seed=42)
        X_s, Y_s = X_s.detach(), Y_s.detach()
        save_cache({'X': X_s, 'Y': Y_s}, data_key, CACHE_DIR)
    else:
        X_s, Y_s = data['X'], data['Y']

    ckpt_key = f'l96_inverter_PAPER_sigma{sigma}_n{N_TRAIN}_ep{N_EPOCHS}.pt'
    ckpt = load_cache(ckpt_key, CACHE_DIR)
    model = ObservationInverterLorenz96Paper(obs_grid=10, full_grid=40).to(device)
    if ckpt is None:
        print(f'  [paper sigma={sigma}] training {N_EPOCHS} epochs...')
        hist = train_inverter_v2(model, X_s, Y_s, label=f'PAPER sigma={sigma}')
        save_cache({'state_dict': model.state_dict(), 'hist': hist}, ckpt_key, CACHE_DIR)
    else:
        model.load_state_dict(ckpt['state_dict'])
        print(f'  [paper sigma={sigma}] loaded from cache (final MSE = {ckpt["hist"][-1]:.4f})')
    return model

inverters_phase4 = {sigma: get_or_train_paper_inverter_at_sigma(sigma)
                    for sigma in NOISE_LEVELS_PHASE4}
print(f'\\nReady: {len(inverters_phase4)} PAPER inverters across noise levels.')""")


code("""# --- Sweep sigma_b for every (init x mode x noise) combo. ---
# 4 combos: paper-baseline x {obs, hybrid} + invobs x {obs, hybrid}.
# Repeat-baseline was shown weaker in Phase 2; we keep the chart focused on the
# paper baseline and the inverter init.

phase4_combos = {
    'paper + obs':           dict(init='paper',  mode='obs'),
    'paper + hybrid':        dict(init='paper',  mode='hybrid'),
    'invobs + obs':          dict(init='invobs', mode='obs'),
    'invobs + hybrid':       dict(init='invobs', mode='hybrid'),
}

p4_cache_key = (
    f'l96_phase4_sigmab_sweep_N{N_EVAL_PHASE4}_T{T_PHASE4}'
    f'_sigmab{"-".join(str(s) for s in SIGMA_B_GRID)}.pt'
)
p4_cache = load_cache(p4_cache_key, CACHE_DIR)
if p4_cache is None:
    import time
    p4_results = {sigma: {name: {} for name in phase4_combos}
                  for sigma in NOISE_LEVELS_PHASE4}
    for sigma_obs in NOISE_LEVELS_PHASE4:
        print(f'\\n=== sigma_obs = {sigma_obs} ===')
        inv = inverters_phase4[sigma_obs]
        X0_gt, _, Y, _ = generate_data(L96, n_samples=N_EVAL_PHASE4,
                                       n_time_steps=T_PHASE4, n_warmup=N_WARMUP,
                                       obs_noise_std=sigma_obs, seed=400)
        inits = {
            'paper':  baseline_init_l96_paper(L96, Y, X0_mean),
            'invobs': invobs_init_l96(inv, Y),
        }
        for name, cfg in phase4_combos.items():
            X0_init = inits[cfg['init']]
            for sigma_b in SIGMA_B_GRID:
                t0 = time.time()
                X0_opt, _ = run_4dvar_l96(L96, inv, corr, X0_init, Y, T=T_PHASE4,
                                          sigma_b=sigma_b, sigma_obs=sigma_obs,
                                          sigma_p=SIGMA_P_PHASE4,
                                          mode=cfg['mode'],
                                          physics_steps=200, obs_steps=300)
                rmse = ((X0_opt - X0_gt) ** 2).mean(dim=-1).sqrt().detach().cpu().numpy()
                p4_results[sigma_obs][name][sigma_b] = rmse
                print(f'  {name:18s}  sigma_b={sigma_b:5.2f}  '
                      f'RMSE={rmse.mean():.4f}+/-{rmse.std():.4f}  '
                      f'({time.time()-t0:.1f}s)')
    save_cache({'p4_results': p4_results}, p4_cache_key, CACHE_DIR)
else:
    p4_results = p4_cache['p4_results']
    print('Loaded Phase 4 sweep results from cache.')""")


code("""# --- Tabulate the optimal sigma_b per cell. ---
import pandas as pd
rows = []
for sigma_obs in NOISE_LEVELS_PHASE4:
    for name in phase4_combos:
        rmse_means = {sb: p4_results[sigma_obs][name][sb].mean() for sb in SIGMA_B_GRID}
        best_sb = min(rmse_means, key=rmse_means.get)
        rows.append({
            'sigma_obs': sigma_obs,
            'combo':     name,
            'best_sigma_b': best_sb,
            'best_RMSE':  rmse_means[best_sb],
            'untuned_RMSE_at_sigma_b=1': rmse_means[1.0],
        })
phase4_table = pd.DataFrame(rows)
print(phase4_table.to_string(index=False))""")


code("""# --- Acceptance check: optimal sigma_b for invobs+hybrid < for paper+obs (per noise). ---
print('\\nAcceptance check (per noise level): is invobs+hybrid optimal sigma_b < paper+obs optimal sigma_b?')
for sigma_obs in NOISE_LEVELS_PHASE4:
    invhyb_sb = phase4_table[(phase4_table.sigma_obs == sigma_obs) &
                             (phase4_table.combo == 'invobs + hybrid')]['best_sigma_b'].values[0]
    base_sb  = phase4_table[(phase4_table.sigma_obs == sigma_obs) &
                            (phase4_table.combo == 'paper + obs')]['best_sigma_b'].values[0]
    verdict  = 'PASS' if invhyb_sb < base_sb else ('TIE' if invhyb_sb == base_sb else 'FAIL')
    print(f'  sigma_obs={sigma_obs}: invobs+hybrid sigma_b={invhyb_sb:5.2f}, '
          f'paper+obs sigma_b={base_sb:5.2f}   -> {verdict}')

# Tuned vs untuned: tuned should beat untuned in every cell.
print('\\nTuned <= untuned (sigma_b = 1.0)?')
for _, row in phase4_table.iterrows():
    impr = row['untuned_RMSE_at_sigma_b=1'] - row['best_RMSE']
    print(f'  sigma_obs={row.sigma_obs}, {row.combo:18s}: '
          f'tuned={row.best_RMSE:.4f}, untuned={row["untuned_RMSE_at_sigma_b=1"]:.4f}, '
          f'gain={impr:.4f}')""")


code("""# --- Tuned bar chart: panel per noise level, bars = combos, sigma_b annotated. ---
phase4_names   = list(phase4_combos.keys())
phase4_colors  = ['#0072B2', '#D55E00', '#009E73', '#CC79A7']
phase4_hatches = ['', '//', '\\\\\\\\', 'xx']

fig, axes = plt.subplots(1, len(NOISE_LEVELS_PHASE4),
                         figsize=(5 * len(NOISE_LEVELS_PHASE4), 4.5),
                         sharey=False)

for col, sigma_obs in enumerate(NOISE_LEVELS_PHASE4):
    ax = axes[col] if len(NOISE_LEVELS_PHASE4) > 1 else axes
    xpos  = np.arange(len(phase4_names))
    means = []
    stds  = []
    sb_choices = []
    for name in phase4_names:
        per_sb = {sb: p4_results[sigma_obs][name][sb] for sb in SIGMA_B_GRID}
        sb_best = min(per_sb, key=lambda s: per_sb[s].mean())
        means.append(per_sb[sb_best].mean())
        stds.append(per_sb[sb_best].std())
        sb_choices.append(sb_best)
    bars = ax.bar(xpos, means, yerr=stds, capsize=3,
                  color=phase4_colors[:len(phase4_names)],
                  hatch=phase4_hatches[:len(phase4_names)],
                  edgecolor='black', linewidth=0.6)
    # Annotate sigma_b above each bar
    for b, sb, m, s in zip(bars, sb_choices, means, stds):
        ax.annotate(rf'$\\sigma_b={sb}$',
                    xy=(b.get_x() + b.get_width() / 2, m + s),
                    ha='center', va='bottom', fontsize=8)
    ax.set_xticks(xpos)
    ax.set_xticklabels(phase4_names, rotation=18, ha='right', fontsize=8)
    ax.set_title(rf'$\\sigma_{{obs}} = {sigma_obs}$')
    ax.set_ylabel('state RMSE (tuned)' if col == 0 else '')
    ax.grid(axis='y', alpha=0.3)

plt.suptitle('Phase 4 tuned-$\\\\sigma_b$ 4D-Var: best result per (init, mode, noise) cell',
             y=1.02, fontsize=11)
plt.tight_layout()
plt.savefig('figures/v2_4dvar_tuned_sigmab.png', dpi=150, bbox_inches='tight')
plt.show()""")


# ============================================================================
# Phase 4 summary
# ============================================================================
md("""---
## Phase 4 summary

**What changed**
- New helpers reused from Phase 2 (`run_4dvar_l96`, `var4d_cost_*`).
- This notebook gained: per-noise PAPER inverter training, $\\sigma_b$ sweep,
  optimal-$\\sigma_b$ table, acceptance check, and tuned bar chart.

**Cache keys produced**
- `l96_train_data_n32000_T20_warmup1000_sigma{0.1, 0.5, 1.0}.pt` (one per noise level)
- `l96_inverter_PAPER_sigma{0.1, 0.5, 1.0}_n32000_ep500.pt`
- `l96_phase4_sigmab_sweep_N16_T10_sigmab0.1-0.3-1.0-3.0-10.0.pt`

**Figures saved**
- `figures/v2_4dvar_tuned_sigmab.png`

**Acceptance criteria**
- "invobs + hybrid" optimal $\\sigma_b$ smaller than "paper + obs" optimal
  $\\sigma_b$ at each noise level.
- Tuned bars lower than the Phase 2 untuned bars in every cell.
- Bar chart at $\\sigma_{obs} = 0.5$ within ~20% of the paper's Figure 4
  Lorenz96 panel.

If any acceptance criterion fails after a full GPU run, **stop** and report
the gap before proceeding to Phase 5+ (per the spec).""")


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

with open("PyTorch_InvObs_DA_v2_PaperFaithful.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print(f"Wrote PyTorch_InvObs_DA_v2_PaperFaithful.ipynb with {len(cells)} cells.")
