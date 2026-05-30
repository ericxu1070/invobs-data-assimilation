# DIGEST: SlidingWindow_Kolmogorov_PyTorch.ipynb  (slug: KFlow_SlidingWindow)

## [md cell 0]
# Sliding-Window 4D-Var on 2D Kolmogorov Flow â€” PyTorch

Companion to `SlidingWindow_PyTorch.ipynb` (Lorenz96). Same three-experiment
structure on **2D Kolmogorov flow** (vorticity formulation, 64x64 grid,
warmed-up initial conditions).

**Experiments**
- **A.** Single window (`T_OBS_TOTAL = WINDOW_T = 10`): 2 init (invobs /
  baseline) x 2 opt (hybrid / obs-only) on noisy + clean observations.
- **B.** Sliding-window stride sweep for `T_OBS_TOTAL in {20, 30}` with
  strides `[2, 5, 10]`. Invobs init is the L-BFGS starting point every
  cycle; the **propagated previous analysis** is the J_b background on
  cycles >= 1.
- **C.** Final comparison: best sliding-window run vs. last-window-only
  baselines, with vorticity snapshot panels and per-cycle L-BFGS loss curves.

Differences from the L96 sliding-window notebook:
- State is `(64, 64)` vorticity. Observations are `(4, 4)` subsampled grid.
- 4D-Var is optimized directly on `omega0` (no `C^{1/2}` preconditioning â€”
  precomputing a 4096x4096 spatial covariance is heavy and the existing
  Kolmogorov PyTorch notebook also skips it).
- Hovmoller is replaced with vorticity field snapshots (truth / SW-best /
  invobs+hybrid + signed error).
- KolmogorovFlow integration uses `torch.fft.rfft2`, so this runs on CPU
  (MPS does not yet support `rfft2`).

Port of Frerix et al. 2021 ([arXiv:2102.11192](https://arxiv.org/abs/2102.11192)).

## [code cell 1]
```python
# Colab setup
import os, sys, subprocess

try:
    import google.colab  # noqa: F401
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

def pip(*pkgs):
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', *pkgs])

# pytorch_kolmogorov.py lives at the repo root. In Colab, clone the repo and
# chdir into it so the module is importable.
REPO_URL = 'https://github.com/ericxu1070/invobs-data-assimilation'
REPO_DIR = '/content/invobs-data-assimilation'

if IN_COLAB and not os.path.exists('pytorch_kolmogorov.py'):
    if not os.path.isdir(REPO_DIR):
        subprocess.check_call(['git', 'clone', '--depth', '1', REPO_URL, REPO_DIR])
    os.chdir(REPO_DIR)
    if REPO_DIR not in sys.path:
        sys.path.insert(0, REPO_DIR)

print('cwd:', os.getcwd())
print('pytorch_kolmogorov.py present:', os.path.exists('pytorch_kolmogorov.py'))
```
--- output ---
cwd: /content/invobs
pytorch_kolmogorov.py present: True

## [code cell 2]
```python
import math
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# KolmogorovFlow uses torch.fft.rfft2 â€” not on MPS. Force CPU.
device = torch.device('cpu')
torch.set_default_dtype(torch.float32)
torch.manual_seed(0)
np.random.seed(0)
print(f'device={device}, torch={torch.__version__}')
```
--- output ---
device=cpu, torch=2.11.0+cu128

## [md cell 3]
### Disk cache (Google Drive)

Survives runtime resets. Set `FORCE_RETRAIN = True` to ignore the cache and regenerate. To clear: delete files in `/content/drive/MyDrive/invobs_cache_kf/`.

## [code cell 4]
```python
import os

FORCE_RETRAIN = False

if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    CACHE_DIR = '/content/drive/MyDrive/invobs_cache_kf'
else:
    CACHE_DIR = './sw_kf_cache'
os.makedirs(CACHE_DIR, exist_ok=True)


def cache_path(name):
    return os.path.join(CACHE_DIR, name)


def save_cache(obj, name):
    torch.save(obj, cache_path(name))
    print(f'  [cache] wrote {name}')


def load_cache(name):
    p = cache_path(name)
    if FORCE_RETRAIN or not os.path.exists(p):
        return None
    print(f'  [cache] loaded {name}')
    return torch.load(p, map_location=device, weights_only=False)


print(f'Cache dir: {CACHE_DIR}')
print(f'Existing cache files: {sorted(os.listdir(CACHE_DIR)) if os.path.isdir(CACHE_DIR) else []}')
```
--- output ---
Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).
Cache dir: /content/drive/MyDrive/invobs_cache_kf
Existing cache files: ['kf_sw_inverter_sigma0.0_n200_ep30.pt']

## [md cell 5]
---
## 1. 2D Kolmogorov flow

Vorticity formulation of 2D incompressible Navier-Stokes with Kolmogorov forcing,
periodic BCs on $[0, 2\pi]^2$. State `omega: (Nx, Ny)`. Pseudo-spectral RK4
solver lives in `pytorch_kolmogorov.py`. Observation operator subsamples every
`observe_every` grid points in both directions: 64 / 16 = 4, so `Y` is `(T, 4, 4)`.

## [code cell 6]
```python
from pytorch_kolmogorov import (
    KolmogorovFlow,
    ObservationInverterKolmogorov,
    generate_kolmogorov_data,
)

NX, NY        = 64, 64
NU            = 1e-2
ALPHA         = 0.1
K_FORCING     = 4
OUTER_DT      = 0.18
N_INNER       = 25
OBSERVE_EVERY = 16   # -> 4x4 obs grid

KF = KolmogorovFlow(
    Nx=NX, Ny=NY, nu=NU, alpha=ALPHA,
    k_forcing=K_FORCING, outer_dt=OUTER_DT, n_inner=N_INNER,
    observe_every=OBSERVE_EVERY, device=device,
)
print(f'KolmogorovFlow ready  |  full grid {NX}x{NY}  |  obs grid {NX//OBSERVE_EVERY}x{NY//OBSERVE_EVERY}'
      f'  |  outer_dt={OUTER_DT} ({N_INNER} inner RK4 steps)')
```
--- output ---
KolmogorovFlow ready  |  full grid 64x64  |  obs grid 4x4  |  outer_dt=0.18 (25 inner RK4 steps)

## [md cell 7]
### Data generation (warmed-up states)

`generate_kolmogorov_data` draws spectrally-filtered random vorticity, integrates
forward `N_WARMUP` outer steps to reach the statistically stationary regime, then
returns `(omega0, traj, Y, Y_clean)` for the following `T_GENERATE` outer steps.

## [code cell 8]
```python
def baseline_init_kf(kf, Y):
    """Bicubic upsample of t=0 observations to full (Nx, Ny). Y: (N, T, X_obs, Y_obs).
    Returns (N, Nx, Ny)."""
    Y0 = Y[:, 0].unsqueeze(1)   # (N, 1, X_obs, Y_obs)
    Y0_up = F.interpolate(Y0, size=(kf.Nx, kf.Ny),
                          mode='bicubic', align_corners=False).squeeze(1)
    return Y0_up


def invobs_init_kf(inverter, Y):
    """Background estimate from H^{-1}_theta evaluated at the observation sequence.
    Y: (N, T, X_obs, Y_obs) -> (N, Nx, Ny). Uses the first frame of the inverted trajectory."""
    inverter.eval()
    with torch.no_grad():
        return inverter(Y).detach()[:, 0]
```

## [md cell 9]
### Train the inverse observation operator

Supervised regression: integrate trajectories, observe them, teach the net to invert $H$. We train one inverter per noise level (sigma_obs in {0.0, 0.1}).

## [code cell 10]
```python
def train_inverter(kf, inverter, n_train=200, T_train=10, n_warmup=100,
                   n_epochs=30, batch_size=8, lr=1e-3, obs_noise_std=0.0,
                   log_every=5, seed=42):
    print(f'  generating {n_train} training trajectories (warmup={n_warmup}, sigma={obs_noise_std})')
    _, X, Y, _ = generate_kolmogorov_data(
        kf, n_samples=n_train, n_time_steps=T_train,
        n_warmup=n_warmup, obs_noise_std=obs_noise_std, seed=seed,
    )
    X = X.detach(); Y = Y.detach()
    opt = torch.optim.Adam(inverter.parameters(), lr=lr)
    n = X.shape[0]
    history = []
    for epoch in range(n_epochs):
        perm = torch.randperm(n, device=device)
        ep_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            y_b, x_b = Y[idx], X[idx]
            pred = inverter(y_b)
            loss = F.mse_loss(pred, x_b)
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item() * idx.numel()
        ep_loss /= n
        history.append(ep_loss)
        if epoch % log_every == 0 or epoch == n_epochs - 1:
            print(f'    epoch {epoch:3d}  loss={ep_loss:.5f}')
    return history
```

## [md cell 11]
### Variational DA

Three optimization modes from the paper (no decorrelation â€” Kolmogorov has too many spatial DOF for a full `C^{1/2}` to be cheap):
- **obs-space**: minimize $\|H(M(\omega_0)) - y\|^2$
- **physics-space**: minimize $\|M(\omega_0) - H^{-1}_\theta(y)\|^2$
- **hybrid**: physics-space warm-start -> obs-space refinement

## [code cell 12]
```python
def da_loss(omega0, Y, kf, T, mode, inverter=None):
    """Batched DA loss.
    omega0: (N, Nx, Ny). Y: (N, T, X_obs, Y_obs).
    Returns scalar (mean over N, T, spatial)."""
    traj = kf.integrate(omega0, T)              # (T, N, Nx, Ny)
    if mode == 'obs':
        pred = kf.observe(traj)                 # (T, N, X_obs, Y_obs)
        target = Y.permute(1, 0, 2, 3)          # (T, N, X_obs, Y_obs)
    elif mode == 'physics':
        assert inverter is not None
        with torch.no_grad():
            inv = inverter(Y).detach()          # (N, T, Nx, Ny)
        target = inv.permute(1, 0, 2, 3)        # (T, N, Nx, Ny)
        pred = traj
    else:
        raise ValueError(mode)
    return ((pred - target) ** 2).mean()


def lbfgs_minimize(loss_fn, omega0_init, max_iter=200, history_size=10, lr=1.0):
    w = omega0_init.clone().detach().requires_grad_(True)
    opt = torch.optim.LBFGS([w], max_iter=max_iter, history_size=history_size,
                            tolerance_grad=1e-12, tolerance_change=1e-12,
                            line_search_fn='strong_wolfe', lr=lr)
    history = []

    def closure():
        opt.zero_grad()
        loss = loss_fn(w)
        loss.backward()
        history.append(loss.item())
        return loss

    try:
        opt.step(closure)
    except Exception as e:
        print(f'    L-BFGS warning: {e}')
    return w.detach(), history
```

## [md cell 13]
### Full 4D-Var (J_b + J_o) with identity background covariance

No decorrelation: `omega0` is the optimization variable directly and
`J_b = 0.5 * ||omega0 - omega_b||^2 / sigma_b^2`.

## [code cell 14]
```python
def var4d_cost_obs(omega0, y_T, kf, T, omega_b, sigma_obs, sigma_b):
    """Standard 4D-Var on Kolmogorov: J_b + J_o, both summed.

    omega0, omega_b : (N, Nx, Ny).  y_T : (T, N, X_obs, Y_obs).
    """
    traj = kf.integrate(omega0, T)
    y_pred = kf.observe(traj)
    innov = y_pred - y_T
    J_o = 0.5 * innov.pow(2).sum() / (sigma_obs ** 2)
    J_b = 0.5 * (omega0 - omega_b).pow(2).sum() / (sigma_b ** 2)
    return J_b + J_o


def var4d_cost_phys(omega0, target_traj, kf, T, omega_b, sigma_b, sigma_p):
    """Physics-space companion: J_b + (1/2 sigma_p^2) ||M(omega0) - H^{-1}_theta(y)||^2.

    target_traj: (T, N, Nx, Ny) inverter output (precomputed, detached).
    """
    traj = kf.integrate(omega0, T)
    J_p = 0.5 * (traj - target_traj).pow(2).sum() / (sigma_p ** 2)
    J_b = 0.5 * (omega0 - omega_b).pow(2).sum() / (sigma_b ** 2)
    return J_b + J_p


def run_4dvar_kf(kf, inverter, omega0_init, Y, T,
                 sigma_b=1.0, sigma_obs=0.1, sigma_p=0.1,
                 mode='obs', physics_steps=50, obs_steps=200,
                 omega_background=None):
    """Full 4D-Var driver for Kolmogorov flow.

    omega0_init      : (N, Nx, Ny) L-BFGS starting point.
    omega_background : optional (N, Nx, Ny) for J_b. If None, omega0_init is used
                       (single-window cold start). In cycling DA pass the
                       propagated previous analysis here while omega0_init can
                       remain the fresh invobs estimate of the new window.

    mode = 'obs'    : minimize J_b + J_o for `obs_steps` L-BFGS iterations.
    mode = 'hybrid' : first minimize J_b + J_phys for `physics_steps` iters,
                      then J_b + J_o.

    Returns (omega0_opt: (N, Nx, Ny), loss_history: list).
    """
    w_b = omega_background if omega_background is not None else omega0_init
    w = omega0_init.clone()
    Y_T = Y.permute(1, 0, 2, 3)   # (T, N, X_obs, Y_obs)
    history = []

    if mode == 'hybrid':
        inverter.eval()
        with torch.no_grad():
            target_traj = inverter(Y).detach().permute(1, 0, 2, 3)   # (T, N, Nx, Ny)
        loss_p = partial(var4d_cost_phys, target_traj=target_traj, kf=kf, T=T,
                         omega_b=w_b, sigma_b=sigma_b, sigma_p=sigma_p)
        w, h_p = lbfgs_minimize(loss_p, w, max_iter=physics_steps)
        history.extend(h_p)

    loss_o = partial(var4d_cost_obs, y_T=Y_T, kf=kf, T=T,
                     omega_b=w_b, sigma_obs=sigma_obs, sigma_b=sigma_b)
    w, h_o = lbfgs_minimize(loss_o, w, max_iter=obs_steps)
    history.extend(h_o)
    return w, history
```

## [md cell 15]
### Sliding-window cycling 4D-Var

Invobs init at the start of every cycle; for cycles >= 1 the J_b background is the propagated previous analysis (not the invobs estimate). The last window is always assimilated.

## [code cell 16]
```python
def run_sliding_window_4dvar_kf(
    kf, inverter, Y_long,
    window_T=10,
    stride=2,
    sigma_b=0.3,
    sigma_obs=0.1,
    sigma_p=0.1,
    init_mode='invobs',
    opt_mode='hybrid',
    physics_steps=50,
    obs_steps=200,
):
    """Sliding-window / cycling 4D-Var on Kolmogorov flow.

    Y_long: (N, T_total, X_obs, Y_obs). Each cycle uses invobs_init_kf (or
    baseline_init_kf for cycle 0 if init_mode='baseline') as the L-BFGS
    starting point. Cycle 0's J_b background equals that starting point;
    cycles >= 1 use the propagated previous analysis.

    Returns dict with
        starts        : list of window start indices in Y_long
        analyses      : (N, n_cycles, Nx, Ny) optimized omega at each start
        backgrounds   : (N, n_cycles, Nx, Ny) J_b background actually used
        invobs_inits  : (N, n_cycles, Nx, Ny) L-BFGS starting point per cycle
        histories     : list (length n_cycles) of L-BFGS loss histories
    """
    N, T_total, _, _ = Y_long.shape
    starts = list(range(0, T_total - window_T + 1, stride))
    if starts[-1] != T_total - window_T:
        starts.append(T_total - window_T)

    analyses, backgrounds, invobs_inits, histories = [], [], [], []
    wb = None

    for c, start in enumerate(starts):
        Y_win = Y_long[:, start:start + window_T]

        if c == 0 and init_mode == 'baseline':
            w_start = baseline_init_kf(kf, Y_win)
        else:
            w_start = invobs_init_kf(inverter, Y_win)

        if c == 0:
            omega_background = None
            wb_record = w_start
        else:
            omega_background = wb
            wb_record = wb

        w_opt, hist = run_4dvar_kf(
            kf=kf, inverter=inverter,
            omega0_init=w_start, Y=Y_win, T=window_T,
            sigma_b=sigma_b, sigma_obs=sigma_obs, sigma_p=sigma_p,
            mode=opt_mode, physics_steps=physics_steps, obs_steps=obs_steps,
            omega_background=omega_background,
        )

        analyses.append(w_opt.detach())
        backgrounds.append(wb_record.detach())
        invobs_inits.append(w_start.detach())
        histories.append(hist)

        if c < len(starts) - 1:
            step = starts[c + 1] - start
            with torch.no_grad():
                wb = kf.integrate(w_opt.detach(), step + 1)[-1].detach()

    return {
        'starts': starts,
        'analyses': torch.stack(analyses, dim=1),         # (N, n_cycles, Nx, Ny)
        'backgrounds': torch.stack(backgrounds, dim=1),
        'invobs_inits': torch.stack(invobs_inits, dim=1),
        'histories': histories,
    }
```

## [md cell 17]
---
## 2. Setup

## [code cell 18]
```python
# ---- Sliding-window experiment configuration (light defaults) -------------
WINDOW_T       = 10       # per-cycle assimilation window
T_FORECAST     = 15       # forecast horizon evaluated after last window ends
T_GENERATE     = 50       # 30 + 15 + 5 margin (enough for largest T_OBS_TOTAL)
N_EVAL         = 8        # ensemble size for evaluation
N_WARMUP       = 100      # outer warmup steps
NOISE_LEVELS   = [0.0]
SIGMA_B_COLD   = 1.0      # single-window (no history)
SIGMA_B_CYCLE  = 0.3      # cycling (trusts propagated analysis)
SIGMA_P        = 0.1
PHYSICS_STEPS  = 50
OBS_STEPS      = 200

T_OBS_TOTALS_B = [20, 30]
STRIDES        = [2, 5, 10]

# sigma_obs is the inverse-variance weighting in J_o. Floor it so the
# perfect-observation case (sigma=0) still produces a finite cost.
def sigma_obs_eff(s):
    return max(float(s), 0.05)

# Inverter training (kept small so it runs on CPU in a few minutes).
INV_N_TRAIN  = 200
INV_T_TRAIN  = WINDOW_T
INV_N_EPOCHS = 30
INV_BATCH    = 8

DATA_SEED      = 12345
```

## [code cell 19]
```python
# ---- Sanity check: visualize the warmup transient ------------------------
# Run a single trajectory through warmup, tracking enstrophy every outer step.
# A warmed-up Kolmogorov flow has enstrophy that grows from ~0 (the smooth
# spectral-filtered cold start) and saturates to a quasi-stationary value.
# Snapshots should evolve from a smooth filtered field into the characteristic
# turbulent shear bands at the forcing scale (k=K_FORCING).
N_WARMUP_CHECK = max(N_WARMUP, 150)
SNAPSHOT_STEPS = [0, N_WARMUP // 4, N_WARMUP // 2, N_WARMUP, N_WARMUP_CHECK]

omega_cold = KF.random_init(batch_size=1, peak_wavenumber=K_FORCING, seed=0)
omega = omega_cold.clone()

enstrophy = [float(0.5 * (omega ** 2).mean().cpu())]
snaps = {0: omega[0].clone()}
for step in range(1, N_WARMUP_CHECK + 1):
    omega = KF.step(omega)
    enstrophy.append(float(0.5 * (omega ** 2).mean().cpu()))
    if step in SNAPSHOT_STEPS:
        snaps[step] = omega[0].clone()

fig = plt.figure(figsize=(15, 3.6))
gs = fig.add_gridspec(1, 1 + len(SNAPSHOT_STEPS),
                      width_ratios=[1.8] + [1.0] * len(SNAPSHOT_STEPS))

ax_e = fig.add_subplot(gs[0, 0])
steps = np.arange(len(enstrophy))
ax_e.plot(steps, enstrophy, color='black', lw=1.6)
ax_e.axvline(N_WARMUP, color='tab:red', ls='--', lw=1.4,
             label=f'N_WARMUP = {N_WARMUP}')
for s in SNAPSHOT_STEPS:
    ax_e.axvline(s, color='tab:blue', ls=':', lw=0.8, alpha=0.5)
ax_e.set_xlabel('outer step')
ax_e.set_ylabel('enstrophy  0.5 * <omega^2>')
ax_e.set_title('Warmup transient (single trajectory)')
ax_e.grid(True, alpha=0.3)
ax_e.legend(fontsize=8, loc='lower right')

vmax = float(max(snaps[s].abs().max() for s in SNAPSHOT_STEPS))
for i, s in enumerate(SNAPSHOT_STEPS):
    ax = fig.add_subplot(gs[0, 1 + i])
    ax.imshow(snaps[s].cpu().numpy(), cmap='RdBu_r',
              vmin=-vmax, vmax=vmax, origin='lower')
    tag = ' (N_WARMUP)' if s == N_WARMUP else ''
    ax.set_title(f't={s}{tag}', fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])

fig.suptitle('Kolmogorov-flow warmup diagnostic  |  enstrophy should plateau; '
             'snapshots should develop k=K_FORCING shear bands',
             y=1.04, fontsize=10)
plt.tight_layout()
plt.show()

# Relative drift of enstrophy over the last 20 steps — a rough stationarity proxy.
tail = enstrophy[-20:]
drift = (max(tail) - min(tail)) / (sum(tail) / len(tail))
print(f'  enstrophy at t=0           : {enstrophy[0]:.3f}')
print(f'  enstrophy at t=N_WARMUP    : {enstrophy[N_WARMUP]:.3f}')
print(f'  enstrophy at t=N_WARMUP+50 : {enstrophy[-1]:.3f}')
print(f'  last-20-step relative drift: {drift:.2%}  (want <~ 10% for stationarity)')
```

>>> FIGURE EMBEDDED: figures/KFlow_SlidingWindow__cell19__out00.png

--- output ---
<Figure size 1500x360 with 6 Axes>  enstrophy at t=0           : 0.500
  enstrophy at t=N_WARMUP    : 9.824
  enstrophy at t=N_WARMUP+50 : 6.195
  last-20-step relative drift: 10.82%  (want <~ 10% for stationarity)

## [code cell 20]
```python
# ---- Train one inverter per noise level -----------------------------------
def build_inverter():
    return ObservationInverterKolmogorov(
        T=WINDOW_T, obs_grid=NX // OBSERVE_EVERY,
        full_grid=NX, in_channels=1, out_channels=1,
    ).to(device)


inverters = {}
inv_hist  = {}
for s in NOISE_LEVELS:
    cache_name = f'kf_sw_inverter_sigma{s}_n{INV_N_TRAIN}_ep{INV_N_EPOCHS}.pt'
    ckpt = load_cache(cache_name)
    inv = build_inverter()
    if ckpt is None:
        print(f'Training inverter for sigma={s} ...')
        hist = train_inverter(
            KF, inv, n_train=INV_N_TRAIN, T_train=INV_T_TRAIN,
            n_warmup=N_WARMUP, n_epochs=INV_N_EPOCHS,
            batch_size=INV_BATCH, obs_noise_std=s,
        )
        save_cache({'state_dict': inv.state_dict(), 'hist': hist}, cache_name)
    else:
        inv.load_state_dict(ckpt['state_dict'])
        hist = ckpt['hist']
    inverters[s] = inv
    inv_hist[s] = hist

# Quick training-loss plot.
fig, ax = plt.subplots(figsize=(7, 3))
for s, h in inv_hist.items():
    ax.plot(h, label=f'sigma={s}')
ax.set_yscale('log'); ax.set_xlabel('epoch'); ax.set_ylabel('MSE')
ax.set_title('Inverse-obs training loss'); ax.legend(); ax.grid(True, alpha=0.3)
plt.show()

# ---- Evaluation dataset per noise level -----------------------------------
# One warmed-up ensemble per noise level: identical truth, different
# realizations of observation noise.
eval_data = {}
omega0_eval, X_long_eval, _, Y_clean_eval = generate_kolmogorov_data(
    KF, n_samples=N_EVAL, n_time_steps=T_GENERATE,
    n_warmup=N_WARMUP, obs_noise_std=0.0, seed=DATA_SEED,
)
omega0_eval = omega0_eval.detach()
X_long_eval = X_long_eval.detach()
Y_clean_eval = Y_clean_eval.detach()
for s in NOISE_LEVELS:
    g = torch.Generator(device=device).manual_seed(DATA_SEED + int(round(s * 1000)) + 1)
    if s > 0:
        noise = torch.empty_like(Y_clean_eval).normal_(generator=g) * s
        Y_obs = Y_clean_eval + noise
    else:
        Y_obs = Y_clean_eval.clone()
    eval_data[s] = {
        'omega0':  omega0_eval,
        'X_true':  X_long_eval,   # (N, T_GENERATE, Nx, Ny)
        'Y_obs':   Y_obs.detach(),
    }
    print(f'  sigma={s}: X_true {tuple(X_long_eval.shape)}, Y_obs {tuple(Y_obs.shape)}')
```

>>> FIGURE EMBEDDED: figures/KFlow_SlidingWindow__cell20__out00.png

--- output ---
  [cache] loaded kf_sw_inverter_sigma0.0_n200_ep30.pt
<Figure size 700x300 with 1 Axes>  sigma=0.0: X_true (8, 50, 64, 64), Y_obs (8, 50, 4, 4)

## [code cell 21]
```python
# ---- Forecast / RMSE helpers ----------------------------------------------
def spatial_rmse(pred, truth):
    """pred, truth: (..., Nx, Ny) -> (...,) sqrt mean square over the two spatial axes."""
    return (pred - truth).pow(2).mean(dim=(-2, -1)).sqrt()


def forecast_curves(kf, omega_analysis, X_truth_long, t_analysis, t_end, T_forecast):
    """Integrate omega_analysis from t_analysis to t_end+T_forecast and compute
    spatial RMSE vs truth for the forecast period [t_end, t_end+T_forecast].

    omega_analysis : (N, Nx, Ny)
    X_truth_long   : (N, T_GENERATE, Nx, Ny)
    Returns (mean_rmse, std_rmse, traj_fcst).
    """
    n_steps = t_end - t_analysis + T_forecast + 1
    with torch.no_grad():
        traj_full = kf.integrate(omega_analysis, n_steps)        # (n_steps, N, Nx, Ny)
    traj_fcst = traj_full[t_end - t_analysis:]                    # (T_f+1, N, Nx, Ny)
    truth_seg = X_truth_long[:, t_end:t_end + T_forecast + 1].permute(1, 0, 2, 3)
    err = spatial_rmse(traj_fcst, truth_seg)                      # (T_f+1, N)
    return err.mean(dim=1).cpu().numpy(), err.std(dim=1).cpu().numpy(), traj_fcst.detach()


def full_rmse_curve(kf, omega_analysis, X_truth_long, t_analysis, t_end, T_forecast):
    """Spatial RMSE over the complete span [t_analysis, t_end + T_forecast]."""
    n_steps = t_end - t_analysis + T_forecast + 1
    with torch.no_grad():
        traj = kf.integrate(omega_analysis, n_steps)              # (n_steps, N, Nx, Ny)
    truth = X_truth_long[:, t_analysis:t_analysis + n_steps].permute(1, 0, 2, 3)
    err = spatial_rmse(traj, truth)                                # (n_steps, N)
    return err.mean(dim=1).cpu().numpy(), err.std(dim=1).cpu().numpy()


def analysis_rmse(omega_analysis, X_truth_long, t_analysis):
    truth = X_truth_long[:, t_analysis]                           # (N, Nx, Ny)
    err = spatial_rmse(omega_analysis, truth)                     # (N,)
    return float(err.mean().cpu()), float(err.std().cpu())
```

## [md cell 22]
---
## 3. Experiment A â€” Single window (`T_OBS_TOTAL = WINDOW_T = 10`)

Four init x opt combinations on a single 10-step assimilation window. All use the cold-start background variance SIGMA_B_COLD = 1.0.

## [code cell 23]
```python
# ---- Experiment A: single-window, 4 init x opt combos ---------------------
T_OBS_TOTAL_A = WINDOW_T  # 10

combos_A = [
    ('invobs + hybrid',     'invobs',   'hybrid'),
    ('invobs + obs-only',   'invobs',   'obs'),
    ('baseline + hybrid',   'baseline', 'hybrid'),
    ('baseline + obs-only', 'baseline', 'obs'),
]

results_A = load_cache('expA_results.pt')
if results_A is None:
    results_A = {}
    for sigma in NOISE_LEVELS:
        d = eval_data[sigma]
        Y_win = d['Y_obs'][:, :T_OBS_TOTAL_A]
        inv = inverters[sigma]
        s_obs = sigma_obs_eff(sigma)
        per_sigma = {}
        for label, init, opt in combos_A:
            if init == 'invobs':
                omega0_init = invobs_init_kf(inv, Y_win)
            else:
                omega0_init = baseline_init_kf(KF, Y_win)
            if opt == 'hybrid':
                ps, os_ = PHYSICS_STEPS, OBS_STEPS
            else:
                ps, os_ = 0, PHYSICS_STEPS + OBS_STEPS
            omega0_opt, hist = run_4dvar_kf(
                KF, inv,
                omega0_init=omega0_init, Y=Y_win, T=WINDOW_T,
                sigma_b=SIGMA_B_COLD, sigma_obs=s_obs, sigma_p=SIGMA_P,
                mode=opt, physics_steps=ps, obs_steps=os_,
            )
            per_sigma[label] = {
                'omega0_opt': omega0_opt.detach().cpu(),
                'hist': hist,
            }
            print(f'  sigma={sigma}  {label:<22s}  iters={len(hist)}  final_loss={hist[-1]:.3f}')
        results_A[sigma] = per_sigma
    save_cache(results_A, 'expA_results.pt')

# Forecast / analysis RMSE summaries for plotting and Exp C.
summary_A = {}
for sigma in NOISE_LEVELS:
    d = eval_data[sigma]
    per_sigma = {}
    for label, _, _ in combos_A:
        w_opt = results_A[sigma][label]['omega0_opt'].to(device)
        mean_f, std_f, _ = forecast_curves(KF, w_opt, d['X_true'],
                                            t_analysis=0, t_end=WINDOW_T,
                                            T_forecast=T_FORECAST)
        a_mean, a_std = analysis_rmse(w_opt, d['X_true'], t_analysis=0)
        per_sigma[label] = {
            'fcst_mean': mean_f, 'fcst_std': std_f,
            'ana_mean': a_mean, 'ana_std': a_std,
        }
    summary_A[sigma] = per_sigma
print('Experiment A summary ready.')
```
--- output ---
  sigma=0.0  invobs + hybrid         iters=270  final_loss=56385.488
  sigma=0.0  invobs + obs-only       iters=264  final_loss=45759.961
  sigma=0.0  baseline + hybrid       iters=267  final_loss=168340.906
  sigma=0.0  baseline + obs-only     iters=261  final_loss=138254.438
  [cache] wrote expA_results.pt
Experiment A summary ready.

## [code cell 24]
```python
# ---- Plot A: RMSE over assimilation window + forecast + analysis bar ------
COLORS_A = {
    'invobs + hybrid':     '#0072B2',
    'invobs + obs-only':   '#56B4E9',
    'baseline + hybrid':   '#D55E00',
    'baseline + obs-only': '#E69F00',
}

fig = plt.figure(figsize=(14, 8))
gs = fig.add_gridspec(2, 2, height_ratios=[1.4, 1.0], hspace=0.35, wspace=0.25)
t_range_A = np.arange(0, WINDOW_T + T_FORECAST + 1)

for col, sigma in enumerate(NOISE_LEVELS):
    d = eval_data[sigma]
    ax = fig.add_subplot(gs[0, col])
    ax.axvspan(0, WINDOW_T, alpha=0.07, color='steelblue', zorder=0,
               label='assimilation window')
    for label, _, _ in combos_A:
        w_opt = results_A[sigma][label]['omega0_opt'].to(device)
        m, s = full_rmse_curve(KF, w_opt, d['X_true'],
                               t_analysis=0, t_end=WINDOW_T, T_forecast=T_FORECAST)
        c = COLORS_A[label]
        ax.plot(t_range_A, m, color=c, lw=1.8, label=label)
        ax.fill_between(t_range_A, m - s, m + s, color=c, alpha=0.15, linewidth=0)
    ax.axvline(WINDOW_T, color='gray', ls='--', lw=1.3, zorder=2, label='window end')
    ax.set_xlim(0, WINDOW_T + T_FORECAST)
    ax.set_title(f'Experiment A: Spatial RMSE  (sigma_obs={sigma})')
    ax.set_xlabel('Outer step')
    ax.set_ylabel('Spatial RMSE')
    ax.grid(True, alpha=0.3)
    if col == 1:
        ax.legend(loc='upper left', fontsize=8)

ax_bar = fig.add_subplot(gs[1, :])
labels = [lab for lab, _, _ in combos_A]
x = np.arange(len(labels))
w = 0.35
for i, sigma in enumerate(NOISE_LEVELS):
    vals = [summary_A[sigma][lab]['ana_mean'] for lab in labels]
    errs = [summary_A[sigma][lab]['ana_std'] for lab in labels]
    ax_bar.bar(x + (i - 0.5) * w, vals, w, yerr=errs, capsize=3,
               label=f'sigma={sigma}', alpha=0.85)
ax_bar.set_xticks(x)
ax_bar.set_xticklabels(labels, rotation=15)
ax_bar.set_ylabel('Analysis RMSE at window start')
ax_bar.set_title('Experiment A: analysis RMSE at window start (t=0)')
ax_bar.grid(True, axis='y', alpha=0.3)
ax_bar.legend()
plt.show()
```

>>> FIGURE EMBEDDED: figures/KFlow_SlidingWindow__cell24__out00.png

--- output ---
<Figure size 1400x800 with 2 Axes>

## [md cell 25]
---
## 4. Experiment B â€” Sliding-window stride sweep

Invobs init every cycle, hybrid opt, propagated background for cycles >= 1. SIGMA_B_CYCLE = 0.3. Sweep strides `[2, 5, 10]` for each `T_OBS_TOTAL` in `{20, 30}`.

## [code cell 26]
```python
# ---- Experiment B: stride sweep across T_OBS_TOTAL ------------------------
sw_results = load_cache('expB_results.pt')
if sw_results is None:
    sw_results = {}
    for T_obs in T_OBS_TOTALS_B:
        for sigma in NOISE_LEVELS:
            d = eval_data[sigma]
            Y_long = d['Y_obs'][:, :T_obs]
            inv = inverters[sigma]
            s_obs = sigma_obs_eff(sigma)
            for stride in STRIDES:
                key = (T_obs, sigma, stride)
                print(f'  running T_obs={T_obs}  sigma={sigma}  stride={stride}')
                out = run_sliding_window_4dvar_kf(
                    KF, inv, Y_long,
                    window_T=WINDOW_T,
                    stride=stride,
                    sigma_b=SIGMA_B_CYCLE,
                    sigma_obs=s_obs,
                    sigma_p=SIGMA_P,
                    init_mode='invobs',
                    opt_mode='hybrid',
                    physics_steps=PHYSICS_STEPS,
                    obs_steps=OBS_STEPS,
                )
                sw_results[key] = {
                    'starts':       out['starts'],
                    'analyses':     out['analyses'].detach().cpu(),
                    'backgrounds':  out['backgrounds'].detach().cpu(),
                    'invobs_inits': out['invobs_inits'].detach().cpu(),
                    'histories':    out['histories'],
                }
    save_cache(sw_results, 'expB_results.pt')

# Per-stride forecast summary + best stride selection.
summary_B = {}
best_stride = {}
for T_obs in T_OBS_TOTALS_B:
    for sigma in NOISE_LEVELS:
        d = eval_data[sigma]
        per_stride = {}
        for stride in STRIDES:
            key = (T_obs, sigma, stride)
            r = sw_results[key]
            t_a = r['starts'][-1]
            W_last = r['analyses'][:, -1].to(device)
            mean_f, std_f, _ = forecast_curves(KF, W_last, d['X_true'],
                                                t_analysis=t_a, t_end=T_obs,
                                                T_forecast=T_FORECAST)
            a_mean, a_std = analysis_rmse(W_last, d['X_true'], t_analysis=t_a)
            per_stride[stride] = {
                'fcst_mean': mean_f, 'fcst_std': std_f,
                'ana_mean':  a_mean, 'ana_std':  a_std,
                'starts':    r['starts'],
            }
        summary_B[(T_obs, sigma)] = per_stride
        avg = {st: per_stride[st]['fcst_mean'].mean() for st in STRIDES}
        best_stride[(T_obs, sigma)] = min(avg, key=avg.get)
        print(f'  T_obs={T_obs} sigma={sigma}  best stride = {best_stride[(T_obs, sigma)]}')

# Reference: best single-window method (lowest mean forecast RMSE) per sigma.
best_A = {}
for sigma in NOISE_LEVELS:
    avg = {lab: summary_A[sigma][lab]['fcst_mean'].mean() for lab, _, _ in combos_A}
    best_A[sigma] = min(avg, key=avg.get)
    print(f'  sigma={sigma}  best Exp-A method = {best_A[sigma]}')
```
--- output ---
  running T_obs=20  sigma=0.0  stride=2
  running T_obs=20  sigma=0.0  stride=5
  running T_obs=20  sigma=0.0  stride=10
  running T_obs=30  sigma=0.0  stride=2
  running T_obs=30  sigma=0.0  stride=5
  running T_obs=30  sigma=0.0  stride=10
  [cache] wrote expB_results.pt
  T_obs=20 sigma=0.0  best stride = 10
  T_obs=30 sigma=0.0  best stride = 10
  sigma=0.0  best Exp-A method = invobs + hybrid

## [code cell 27]
```python
# ---- Plot B1: RMSE per stride â€” absolute time axis, from t=0 -------------
STRIDE_COLORS = {2: '#1b9e77', 5: '#d95f02', 10: '#7570b3'}

fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharey='row')
for row_idx, T_obs in enumerate(T_OBS_TOTALS_B):
    t_win_start = T_obs - WINDOW_T
    t_range_sw  = np.arange(t_win_start, T_obs + T_FORECAST + 1)
    t_range_ref = np.arange(t_win_start, T_obs + T_FORECAST + 1)

    for col_idx, sigma in enumerate(NOISE_LEVELS):
        d  = eval_data[sigma]
        ax = axes[row_idx, col_idx]
        ax.axvspan(t_win_start, T_obs, alpha=0.07, color='steelblue', zorder=0)

        for stride in STRIDES:
            rec    = sw_results[(T_obs, sigma, stride)]
            W_last = rec['analyses'][:, -1].to(device)
            t_a    = rec['starts'][-1]
            m, s = full_rmse_curve(KF, W_last, d['X_true'],
                                   t_analysis=t_a, t_end=T_obs, T_forecast=T_FORECAST)
            color = STRIDE_COLORS[stride]
            ax.plot(t_range_sw, m, color=color, lw=1.8, label=f'stride={stride}')
            ax.fill_between(t_range_sw, m - s, m + s, color=color, alpha=0.15, linewidth=0)

        # Reference: best Exp-A RMSE values plotted at the last-window time position.
        w_ref = results_A[sigma][best_A[sigma]]['omega0_opt'].to(device)
        m_ref, s_ref = full_rmse_curve(KF, w_ref, d['X_true'],
                                       t_analysis=0, t_end=WINDOW_T, T_forecast=T_FORECAST)
        ax.plot(t_range_ref, m_ref, color='black', lw=1.2, ls='--',
                label=f'best Exp-A ({best_A[sigma]})')
        ax.fill_between(t_range_ref, m_ref - s_ref, m_ref + s_ref,
                        color='black', alpha=0.08, linewidth=0)

        ax.axvline(T_obs, color='gray', ls='--', lw=1.3, zorder=2, label='window end')
        ax.set_xlim(0, T_obs + T_FORECAST)
        ax.set_title(f'T_obs={T_obs}, sigma={sigma}')
        ax.grid(True, alpha=0.3)
        if row_idx == 1:
            ax.set_xlabel('Outer step')
        if col_idx == 0:
            ax.set_ylabel('Spatial RMSE')
        if row_idx == 0 and col_idx == 1:
            ax.legend(fontsize=8, loc='upper left')

fig.suptitle('Experiment B1: RMSE per stride â€” last assimilation window + forecast\n'
             '(shaded = last assimilation window; Exp-A reference shifted to last window position)', y=1.03)
plt.tight_layout()
plt.show()
```

>>> FIGURE EMBEDDED: figures/KFlow_SlidingWindow__cell27__out00.png

--- output ---
<Figure size 1300x900 with 4 Axes>

## [code cell 28]
```python
# ---- Plot B2: per-cycle background vs analysis RMSE for best_stride run ----
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
for r, T_obs in enumerate(T_OBS_TOTALS_B):
    for c, sigma in enumerate(NOISE_LEVELS):
        ax = axes[r, c]
        st = best_stride[(T_obs, sigma)]
        rec = sw_results[(T_obs, sigma, st)]
        d = eval_data[sigma]
        analyses    = rec['analyses'].to(device)
        backgrounds = rec['backgrounds'].to(device)
        starts = rec['starts']
        ana_curve, bg_curve = [], []
        for ci, t_a in enumerate(starts):
            truth = d['X_true'][:, t_a]
            ana_curve.append(float(spatial_rmse(analyses[:, ci], truth).mean().cpu()))
            bg_curve.append(float(spatial_rmse(backgrounds[:, ci], truth).mean().cpu()))
        cycles = np.arange(len(starts))
        ax.plot(cycles, bg_curve, color='#d62728', lw=1.6, ls='--',
                marker='s', label='background')
        ax.plot(cycles, ana_curve, color='#1f77b4', lw=1.8,
                marker='o', label='analysis')
        ref = summary_A[sigma][best_A[sigma]]['ana_mean']
        ax.axhline(ref, color='gray', ls=':', lw=1.2,
                   label=f'single-window {best_A[sigma]}')
        ax.set_title(f'T_obs={T_obs}, sigma={sigma}, best stride={st}')
        ax.set_xlabel('Cycle index')
        if c == 0:
            ax.set_ylabel('RMSE at window start')
        ax.grid(True, alpha=0.3)
        if r == 0 and c == 1:
            ax.legend(fontsize=8, loc='upper right')

fig.suptitle('Experiment B2: per-cycle background vs analysis RMSE (best stride)', y=1.02)
plt.tight_layout()
plt.show()
```

>>> FIGURE EMBEDDED: figures/KFlow_SlidingWindow__cell28__out00.png

--- output ---
<Figure size 1300x900 with 4 Axes>

## [md cell 29]
---
## 5. Experiment C â€” Final comparison

SW-best (best stride from Experiment B, full history) vs invobs+hybrid on the last 10 observations vs baseline+obs-only on the last 10 observations.

## [code cell 30]
```python
# ---- Experiment C: SW-best vs single-window methods on last WINDOW_T obs ----
results_C = load_cache('expC_results.pt')
if results_C is None:
    results_C = {}
    for T_obs in T_OBS_TOTALS_B:
        for sigma in NOISE_LEVELS:
            d = eval_data[sigma]
            inv = inverters[sigma]
            s_obs = sigma_obs_eff(sigma)
            st = best_stride[(T_obs, sigma)]
            rec = sw_results[(T_obs, sigma, st)]

            Y_win = d['Y_obs'][:, T_obs - WINDOW_T:T_obs]

            w0_invh_init = invobs_init_kf(inv, Y_win)
            w0_invh, h_invh = run_4dvar_kf(
                KF, inv, omega0_init=w0_invh_init, Y=Y_win, T=WINDOW_T,
                sigma_b=SIGMA_B_COLD, sigma_obs=s_obs, sigma_p=SIGMA_P,
                mode='hybrid', physics_steps=PHYSICS_STEPS, obs_steps=OBS_STEPS,
            )
            w0_base_init = baseline_init_kf(KF, Y_win)
            w0_base, h_base = run_4dvar_kf(
                KF, inv, omega0_init=w0_base_init, Y=Y_win, T=WINDOW_T,
                sigma_b=SIGMA_B_COLD, sigma_obs=s_obs, sigma_p=SIGMA_P,
                mode='obs', physics_steps=0, obs_steps=PHYSICS_STEPS + OBS_STEPS,
            )
            results_C[(T_obs, sigma)] = {
                'best_stride':       st,
                'sw_last_analysis':  rec['analyses'][:, -1].cpu(),
                'sw_last_start':     rec['starts'][-1],
                'invh_omega0':       w0_invh.detach().cpu(),
                'base_omega0':       w0_base.detach().cpu(),
            }
            print(f'  T_obs={T_obs} sigma={sigma}  done.')
    save_cache(results_C, 'expC_results.pt')

METHOD_LABELS_C = ['SW-best', 'invobs + hybrid (last)', 'baseline + obs-only (last)']
summary_C = {}
for T_obs in T_OBS_TOTALS_B:
    for sigma in NOISE_LEVELS:
        d = eval_data[sigma]
        r = results_C[(T_obs, sigma)]
        per_method = {}

        W_sw = r['sw_last_analysis'].to(device)
        m_sw, s_sw, traj_sw = forecast_curves(KF, W_sw, d['X_true'],
                                               t_analysis=r['sw_last_start'],
                                               t_end=T_obs, T_forecast=T_FORECAST)
        a_sw_m, a_sw_s = analysis_rmse(W_sw, d['X_true'], t_analysis=r['sw_last_start'])
        per_method['SW-best'] = {
            'fcst_mean': m_sw, 'fcst_std': s_sw, 'traj': traj_sw,
            'ana_mean':  a_sw_m, 'ana_std':  a_sw_s,
            't_analysis': r['sw_last_start'],
        }

        W_ih = r['invh_omega0'].to(device)
        m_ih, s_ih, traj_ih = forecast_curves(KF, W_ih, d['X_true'],
                                                t_analysis=T_obs - WINDOW_T,
                                                t_end=T_obs, T_forecast=T_FORECAST)
        a_ih_m, a_ih_s = analysis_rmse(W_ih, d['X_true'], t_analysis=T_obs - WINDOW_T)
        per_method['invobs + hybrid (last)'] = {
            'fcst_mean': m_ih, 'fcst_std': s_ih, 'traj': traj_ih,
            'ana_mean':  a_ih_m, 'ana_std':  a_ih_s,
            't_analysis': T_obs - WINDOW_T,
        }

        W_bs = r['base_omega0'].to(device)
        m_bs, s_bs, traj_bs = forecast_curves(KF, W_bs, d['X_true'],
                                                t_analysis=T_obs - WINDOW_T,
                                                t_end=T_obs, T_forecast=T_FORECAST)
        a_bs_m, a_bs_s = analysis_rmse(W_bs, d['X_true'], t_analysis=T_obs - WINDOW_T)
        per_method['baseline + obs-only (last)'] = {
            'fcst_mean': m_bs, 'fcst_std': s_bs, 'traj': traj_bs,
            'ana_mean':  a_bs_m, 'ana_std':  a_bs_s,
            't_analysis': T_obs - WINDOW_T,
        }
        summary_C[(T_obs, sigma)] = per_method
print('Experiment C summary ready.')
```
--- output ---
  T_obs=20 sigma=0.0  done.
  T_obs=30 sigma=0.0  done.
  [cache] wrote expC_results.pt
Experiment C summary ready.

## [code cell 31]
```python
# ---- Plot C: RMSE â€” absolute time axis, from t=0 (SW vs single-window) ---
COLORS_C = {
    'SW-best':                      '#0072B2',
    'invobs + hybrid (last)':       '#009E73',
    'baseline + obs-only (last)':   '#D55E00',
}

fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharey='row')
for row_idx, T_obs in enumerate(T_OBS_TOTALS_B):
    t_win_start = T_obs - WINDOW_T

    for col_idx, sigma in enumerate(NOISE_LEVELS):
        d  = eval_data[sigma]
        rc = results_C[(T_obs, sigma)]
        ax = axes[row_idx, col_idx]
        ax.axvspan(t_win_start, T_obs, alpha=0.07, color='steelblue', zorder=0)

        # SW-best: per-cycle analysis RMSE chain from t=0 + last-cycle continuous.
        st_sw   = rc['best_stride']
        rec_sw  = sw_results[(T_obs, sigma, st_sw)]
        sw_starts   = rec_sw['starts']
        sw_analyses = rec_sw['analyses'].to(device)

        chain_t, chain_m, chain_s = [], [], []
        for ci, t_a in enumerate(sw_starts):
            err = spatial_rmse(sw_analyses[:, ci], d['X_true'][:, t_a])
            chain_t.append(t_a)
            chain_m.append(float(err.mean().cpu()))
            chain_s.append(float(err.std().cpu()))
        chain_t = np.array(chain_t)
        chain_m = np.array(chain_m)
        chain_s = np.array(chain_s)

        m_sw, s_sw = full_rmse_curve(KF, sw_analyses[:, -1], d['X_true'],
                                      t_analysis=sw_starts[-1], t_end=T_obs,
                                      T_forecast=T_FORECAST)
        t_range_sw_last = np.arange(sw_starts[-1], T_obs + T_FORECAST + 1)

        color_sw = COLORS_C['SW-best']
        ax.plot(chain_t, chain_m, color=color_sw, lw=1.4, ls='--',
                marker='o', markersize=4, zorder=3, label='SW-best (per-cycle analyses)')
        ax.fill_between(chain_t, chain_m - chain_s, chain_m + chain_s,
                        color=color_sw, alpha=0.12, linewidth=0)
        ax.plot(t_range_sw_last, m_sw, color=color_sw, lw=1.8)
        ax.fill_between(t_range_sw_last, m_sw - s_sw, m_sw + s_sw,
                        color=color_sw, alpha=0.15, linewidth=0)

        # Single-window baselines on the last window.
        t_range_single = np.arange(t_win_start, T_obs + T_FORECAST + 1)
        for label, w_key in [('invobs + hybrid (last)', 'invh_omega0'),
                              ('baseline + obs-only (last)', 'base_omega0')]:
            W = rc[w_key].to(device)
            m, s = full_rmse_curve(KF, W, d['X_true'],
                                   t_analysis=t_win_start, t_end=T_obs, T_forecast=T_FORECAST)
            color = COLORS_C[label]
            ax.plot(t_range_single, m, color=color, lw=1.8, label=label)
            ax.fill_between(t_range_single, m - s, m + s, color=color, alpha=0.15, linewidth=0)

        ax.axvline(T_obs, color='gray', ls='--', lw=1.3, zorder=2, label='window end')
        ax.set_xlim(0, T_obs + T_FORECAST)
        ax.set_title(f'T_obs={T_obs}, sigma={sigma}')
        ax.grid(True, alpha=0.3)
        if row_idx == 1:
            ax.set_xlabel('Outer step')
        if col_idx == 0:
            ax.set_ylabel('Spatial RMSE')
        if row_idx == 0 and col_idx == 1:
            ax.legend(fontsize=8, loc='upper left')

fig.suptitle('Experiment C: RMSE â€” sliding window vs single window\n'
             '(SW-best dashed = per-cycle analysis RMSE from t=0; solid = last-cycle + forecast)', y=1.03)
plt.tight_layout()
plt.show()
```

>>> FIGURE EMBEDDED: figures/KFlow_SlidingWindow__cell31__out00.png

--- output ---
<Figure size 1300x900 with 4 Axes>

## [code cell 32]
```python
# ---- Plot D: analysis RMSE at last-window start (bar chart) ---------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
x = np.arange(len(T_OBS_TOTALS_B))
w = 0.27
for col, sigma in enumerate(NOISE_LEVELS):
    ax = axes[col]
    for i, label in enumerate(METHOD_LABELS_C):
        vals = [summary_C[(T_obs, sigma)][label]['ana_mean'] for T_obs in T_OBS_TOTALS_B]
        errs = [summary_C[(T_obs, sigma)][label]['ana_std'] for T_obs in T_OBS_TOTALS_B]
        ax.bar(x + (i - 1) * w, vals, w, yerr=errs, capsize=3,
               label=label, color=COLORS_C[label], alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([f'T_obs={T}' for T in T_OBS_TOTALS_B])
    ax.set_title(f'sigma={sigma}')
    ax.grid(True, axis='y', alpha=0.3)
    if col == 0:
        ax.set_ylabel('Analysis RMSE at last-window start')
    if col == 1:
        ax.legend(fontsize=8, loc='upper right')

fig.suptitle('Experiment D: analysis RMSE at last-window start', y=1.02)
plt.tight_layout()
plt.show()
```

>>> FIGURE EMBEDDED: figures/KFlow_SlidingWindow__cell32__out00.png

--- output ---
<Figure size 1200x450 with 2 Axes>

## [code cell 33]
```python
# ---- Plot E: vorticity field snapshots (truth / SW-best / invobs+hybrid + error)
# 5 rows per (T_obs, sigma) column:
#   0: truth
#   1: SW-best prediction      (RdBu_r, Â±vmax_state)
#   2: SW-best error           (bwr,    Â±vmax_err shared with row 4)
#   3: invobs+hybrid last      (RdBu_r)
#   4: invobs+hybrid error     (bwr,    same Â±vmax_err)
# Three snapshot times per panel column: window-start, window-end, forecast-end.
SAMPLE_IDX = 0
SNAP_OFFSETS = [0, WINDOW_T, WINDOW_T + T_FORECAST]   # t relative to last-window start
ROW_LABELS = [
    'truth',
    'SW-best',
    'SW-best\nerror',
    'invobs+hybrid\n(last)',
    'invobs+hybrid\nerror',
]
n_rows = len(ROW_LABELS)
n_snaps = len(SNAP_OFFSETS)

for T_obs in T_OBS_TOTALS_B:
    for sigma in NOISE_LEVELS:
        d = eval_data[sigma]
        r = results_C[(T_obs, sigma)]

        truth_seg = d['X_true'][SAMPLE_IDX,
                                T_obs - WINDOW_T:T_obs + T_FORECAST + 1].cpu().numpy()
        W_sw = r['sw_last_analysis'].to(device)
        W_ih = r['invh_omega0'].to(device)
        n_total = WINDOW_T + T_FORECAST + 1
        with torch.no_grad():
            traj_sw = KF.integrate(W_sw, n_total)[:, SAMPLE_IDX].cpu().numpy()
            traj_ih = KF.integrate(W_ih, n_total)[:, SAMPLE_IDX].cpu().numpy()
        err_sw = traj_sw - truth_seg
        err_ih = traj_ih - truth_seg

        vmax_s = float(np.abs(truth_seg).max())
        vmax_e = float(max(np.abs(err_sw).max(), np.abs(err_ih).max()))

        fig, axes = plt.subplots(n_rows, n_snaps, figsize=(3.2 * n_snaps, 2.8 * n_rows))
        for col, off in enumerate(SNAP_OFFSETS):
            panels = [
                (truth_seg[off], 'RdBu_r', -vmax_s, vmax_s),
                (traj_sw[off],   'RdBu_r', -vmax_s, vmax_s),
                (err_sw[off],    'bwr',    -vmax_e, vmax_e),
                (traj_ih[off],   'RdBu_r', -vmax_s, vmax_s),
                (err_ih[off],    'bwr',    -vmax_e, vmax_e),
            ]
            t_abs = (T_obs - WINDOW_T) + off
            for row, (data, cmap, vmin, vmax) in enumerate(panels):
                ax = axes[row, col]
                ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
                ax.set_xticks([]); ax.set_yticks([])
                if row == 0:
                    label = ('window start' if off == 0
                             else 'window end' if off == WINDOW_T
                             else 'forecast end')
                    ax.set_title(f't={t_abs}  ({label})', fontsize=9)
                if col == 0:
                    ax.set_ylabel(ROW_LABELS[row], fontsize=9)

        fig.suptitle(f'Vorticity snapshots â€” T_obs={T_obs}, sigma={sigma} (sample {SAMPLE_IDX})\n'
                     'State rows (RdBu_r): Â±max(|truth|).  Error rows (bwr): pred âˆ’ truth.',
                     y=1.02, fontsize=10)
        plt.tight_layout()
        plt.show()
```

>>> FIGURE EMBEDDED: figures/KFlow_SlidingWindow__cell33__out00.png


>>> FIGURE EMBEDDED: figures/KFlow_SlidingWindow__cell33__out01.png

--- output ---
<Figure size 960x1400 with 15 Axes><Figure size 960x1400 with 15 Axes>

## [code cell 34]
```python
# ---- Plot F: per-window L-BFGS loss curves for the SW-best runs -----------
import matplotlib.cm as cm

for T_obs in T_OBS_TOTALS_B:
    for sigma in NOISE_LEVELS:
        st = best_stride[(T_obs, sigma)]
        rec = sw_results[(T_obs, sigma, st)]
        histories = rec['histories']
        n_cycles = len(histories)
        ncols = min(n_cycles, 5)
        nrows = math.ceil(n_cycles / ncols)
        fig, axes = plt.subplots(nrows, ncols,
                                  figsize=(3.0 * ncols, 2.4 * nrows),
                                  squeeze=False)
        cmap = cm.get_cmap('Blues')
        for ci, h in enumerate(histories):
            ax = axes[ci // ncols, ci % ncols]
            color = cmap(0.25 + 0.7 * ci / max(n_cycles - 1, 1))
            ax.plot(h, color=color, lw=1.4)
            ax.set_yscale('log')
            ax.set_title(f'cycle {ci}', fontsize=8)
            ax.grid(True, alpha=0.3)
        for k in range(n_cycles, nrows * ncols):
            axes[k // ncols, k % ncols].axis('off')
        fig.suptitle(f'L-BFGS loss per cycle  |  T_obs={T_obs}, sigma={sigma}, stride={st}',
                      y=1.02, fontsize=10)
        plt.tight_layout()
        plt.show()
```

>>> FIGURE EMBEDDED: figures/KFlow_SlidingWindow__cell34__out00.png


>>> FIGURE EMBEDDED: figures/KFlow_SlidingWindow__cell34__out01.png

--- output ---
/tmp/ipykernel_1979/72553093.py:15: MatplotlibDeprecationWarning: The get_cmap function was deprecated in Matplotlib 3.7 and will be removed in 3.11. Use ``matplotlib.colormaps[name]`` or ``matplotlib.colormaps.get_cmap()`` or ``pyplot.get_cmap()`` instead.
  cmap = cm.get_cmap('Blues')
<Figure size 600x240 with 2 Axes><Figure size 900x240 with 3 Axes>
