# DIGEST: SlidingWindow_PyTorch.ipynb  (slug: L96_SlidingWindow)

## [md cell 0]
# Sliding-Window 4D-Var on Lorenz96 — PyTorch

Compares **cycling sliding-window 4D-Var** against single-window DA across
different observation-history lengths and assimilation strides.

**Experiments**
- **A.** Single window (`T_OBS_TOTAL = 8`): 2 init (invobs / baseline) x 2 opt
  (hybrid / obs-only) baselines, two noise levels.
- **B.** Sliding-window stride sweep for `T_OBS_TOTAL in {24, 48}` with
  strides `[2, 4, 8]`. Invobs init is used as the L-BFGS starting point every
  cycle; the **propagated previous analysis** is used as the J_b background
  on cycles >= 1.
- **C.** Final comparison: best sliding-window run vs. last-window-only
  baselines, with forecast Hovmoller panels and per-cycle L-BFGS loss curves.

Port of Frerix et al. 2021 ([arXiv:2102.11192](https://arxiv.org/abs/2102.11192))
infrastructure, with sliding-window cycling added on top.

## [code cell 1]
```python
# Colab setup
try:
    import google.colab  # noqa: F401
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

import sys, subprocess
def pip(*pkgs):
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', *pkgs])

# torchdiffeq is optional — we use a hand-written RK4 below. Uncomment if you want adaptive solvers.
# pip('torchdiffeq')
```

## [code cell 2]
```python
import math
from functools import partial
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float32)
torch.manual_seed(0)
np.random.seed(0)
print(f'device={device}, torch={torch.__version__}')
```
--- output ---
device=cuda, torch=2.11.0+cu128

## [md cell 3]
### Disk cache (Google Drive)

Survives runtime resets. Set `FORCE_RETRAIN = True` to ignore the cache and regenerate. To clear: delete files in `/content/drive/MyDrive/invobs_cache/`.

## [code cell 4]
```python
import os

FORCE_RETRAIN = False

if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    CACHE_DIR = '/content/drive/MyDrive/invobs_cache'
else:
    CACHE_DIR = './sw_cache'
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
Cache dir: /content/drive/MyDrive/invobs_cache
Existing cache files: ['.ipynb_checkpoints', 'expA_results.pt', 'expB_results.pt', 'expC_results.pt', 'l96_corr.pt', 'l96_corrected_da_v2_batched_ckptl96_inverter_sigma0.0_n32000_ep500_T10_methodbaseline_init___hybrid_opt_N16_forecast40_sigma0.0_seed1706_p50_o150.pt', 'l96_corrected_da_v2_batched_ckptl96_inverter_sigma0.0_n32000_ep500_T10_methodbaseline_init___hybrid_opt_N8_forecast40_sigma0.0_seed1706_p20_o60.pt', 'l96_corrected_da_v2_batched_ckptl96_inverter_sigma0.0_n32000_ep500_T10_methodbaseline_init___observation_opt_N16_forecast40_sigma0.0_seed1706_p0_o200.pt', 'l96_corrected_da_v2_batched_ckptl96_inverter_sigma0.0_n32000_ep500_T10_methodbaseline_init___observation_opt_N8_forecast40_sigma0.0_seed1706_p0_o80.pt', 'l96_corrected_da_v2_batched_ckptl96_inverter_sigma0.0_n32000_ep500_T10_methodinverse_init___hybrid_opt_N16_forecast40_sigma0.0_seed1706_p50_o150.pt', 'l96_corrected_da_v2_batched_ckptl96_inverter_sigma0.0_n32000_ep500_T10_methodinverse_init___hybrid_opt_N8_forecast40_sigma0.0_seed1706_p20_o60.pt', 'l96_corrected_da_v2_batched_ckptl96_inverter_sigma0.0_n32000_ep500_T10_methodinverse_init___observation_opt_N16_forecast40_sigma0.0_seed1706_p0_o200.pt', 'l96_corrected_da_v2_batched_ckptl96_inverter_sigma0.0_n32000_ep500_T10_methodinverse_init___observation_opt_N8_forecast40_sigma0.0_seed1706_p0_o80.pt', 'l96_corrected_da_v2_batched_ckptl96_inverter_sigma0.0_n32000_ep500_T11_methodbaseline_init___hybrid_opt_N16_forecast40_sigma0.0_seed1706_p50_o150.pt', 'l96_corrected_da_v2_batched_ckptl96_inverter_sigma0.0_n32000_ep500_T11_methodbaseline_init___observation_opt_N16_forecast40_sigma0.0_seed1706_p0_o200.pt', 'l96_corrected_da_v2_batched_ckptl96_inverter_sigma0.0_n32000_ep500_T11_methodinverse_init___hybrid_opt_N16_forecast40_sigma0.0_seed1706_p50_o150.pt', 'l96_corrected_da_v2_batched_ckptl96_inverter_sigma0.0_n32000_ep500_T11_methodinverse_init___observation_opt_N16_forecast40_sigma0.0_seed1706_p0_o200.pt', 'l96_corrected_da_v2_batched_ckptl96_inverter_sigma0.0_n32000_ep500_T12_methodbaseline_init___hybrid_opt_N16_forecast40_sigma0.0_seed1706_p50_o150.pt', 'l96_corrected_da_v2_batched_ckptl96_inverter_sigma0.0_n32000_ep500_T12_methodbaseline_init___observation_opt_N16_forecast40_sigma0.0_seed1706_p0_o200.pt', 'l96_corrected_da_v2_batched_ckptl96_inverter_sigma0.0_n32000_ep500_T12_methodinverse_init___hybrid_opt_N16_forecast40_sigma0.0_seed1706_p50_o150.pt', 'l96_corrected_da_v2_batched_ckptl96_inverter_sigma0.0_n32000_ep500_T12_methodinverse_init___observation_opt_N16_forecast40_sigma0.0_seed1706_p0_o200.pt', 'l96_corrected_da_v2_batched_ckptl96_inverter_sigma0.0_n32000_ep500_T13_methodbaseline_init___hybrid_opt_N16_forecast40_sigma0.0_seed1706_p50_o150.pt', 'l96_corrected_da_v2_batched_ckptl96_inverter_sigma0.0_n32000_ep500_T13_methodbaseline_init___observation_o
...[truncated]...

## [md cell 5]
---
## 1. Lorenz96

$\dot x_k = -x_{k-1}(x_{k-2} - x_{k+1}) - x_k + F$ with periodic BCs. Observation operator: subsample every `observe_every` grid points.

## [code cell 6]
```python
def rk4_step(rhs, x, dt):
    k1 = rhs(x)
    k2 = rhs(x + 0.5 * dt * k1)
    k3 = rhs(x + 0.5 * dt * k2)
    k4 = rhs(x + dt * k3)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


class Lorenz96:
    def __init__(self, grid_size=40, F=8.0, dt=0.01, observe_every=4, n_inner=10):
        # n_inner internal RK4 steps per "outer" step of size dt*n_inner.
        self.grid_size = grid_size
        self.F = F
        self.dt = dt
        self.n_inner = n_inner
        self.outer_dt = dt * n_inner
        self.observe_every = observe_every

    def rhs(self, x):
        xp1 = torch.roll(x, -1, dims=-1)
        xm1 = torch.roll(x,  1, dims=-1)
        xm2 = torch.roll(x,  2, dims=-1)
        return (xp1 - xm2) * xm1 - x + self.F

    def step(self, x):
        for _ in range(self.n_inner):
            x = rk4_step(self.rhs, x, self.dt)
        return x

    def integrate(self, x0, n_steps, start_with_input=True):
        traj = [x0] if start_with_input else []
        x = x0
        n = n_steps if start_with_input else n_steps
        for t in range(n_steps - (1 if start_with_input else 0)):
            x = self.step(x)
            traj.append(x)
        return torch.stack(traj, dim=0)  # (T, ..., grid_size)

    def warmup(self, x0, total_inner_steps):
        x = x0
        for _ in range(total_inner_steps):
            x = rk4_step(self.rhs, x, self.dt)
        return x

    def observe(self, x):
        return x[..., ::self.observe_every]


L96 = Lorenz96(grid_size=40, dt=0.01, n_inner=10, observe_every=4)  # outer_dt = 0.1
```

## [md cell 7]
### Data generation and spatial correlation

`generate_data` returns `(X0, X_true, Y)` where `Y = H(X_true) + noise`. Also computes a long trajectory for estimating the spatial correlation matrix `C` used to precondition the optimizer.

## [code cell 8]
```python
def generate_data(dyn_sys, n_samples, n_time_steps, n_warmup, obs_noise_std=0.0, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    X0_cold = torch.randn(n_samples, dyn_sys.grid_size, device=device, generator=g) * 0.5
    # Spin up each trajectory independently.
    X0 = torch.stack([dyn_sys.warmup(x, n_warmup) for x in X0_cold], dim=0)
    # Forward integration.
    X = torch.stack([dyn_sys.integrate(x, n_time_steps) for x in X0], dim=0)  # (N, T, grid)
    Y_clean = dyn_sys.observe(X)                                               # (N, T, obs_grid)
    noise = torch.empty_like(Y_clean).normal_(generator=g) * obs_noise_std
    Y = Y_clean + noise
    return X0, X, Y, Y_clean


def estimate_correlation(dyn_sys, n_samples=2000, n_warmup=1000, seed=1):
    # One long ensemble of warmed-up states. C is estimated over the ensemble.
    g = torch.Generator(device=device).manual_seed(seed)
    X0 = torch.randn(n_samples, dyn_sys.grid_size, device=device, generator=g) * 0.5
    X = torch.stack([dyn_sys.warmup(x, n_warmup) for x in X0], dim=0)  # (N, grid)
    X = X - X.mean(dim=0, keepdim=True)
    C = (X.T @ X) / (X.shape[0] - 1)  # (grid, grid)
    # Symmetric matrix square root via eigendecomposition.
    eigvals, eigvecs = torch.linalg.eigh(C)
    eigvals = torch.clamp(eigvals, min=1e-8)
    C_sqrt = eigvecs @ torch.diag(eigvals.sqrt()) @ eigvecs.T
    C_inv_sqrt = eigvecs @ torch.diag(eigvals.rsqrt()) @ eigvecs.T
    C_inv = eigvecs @ torch.diag(1.0 / eigvals) @ eigvecs.T
    return dict(C=C, C_sqrt=C_sqrt, C_inv_sqrt=C_inv_sqrt, C_inv=C_inv)


corr = load_cache('l96_corr.pt')
if corr is None:
    corr = estimate_correlation(L96, n_samples=1000, n_warmup=500)
    save_cache(corr, 'l96_corr.pt')
print('C shape:', corr['C'].shape, 'cond=', torch.linalg.cond(corr['C']).item())
```
--- output ---
  [cache] loaded l96_corr.pt
C shape: torch.Size([40, 40]) cond= 8.72168254852295

## [md cell 9]
### Inverse observation operator $H^{-1}_\theta$

Maps an *observation sequence* $Y \in \mathbb{R}^{T \times n_{obs}}$ back to full physical space $X \in \mathbb{R}^{T \times n_{grid}}$. CNN with **periodic** padding in the grid dimension and **zero** padding in time (matches the paper's `PeriodicSpaceConv`).

## [code cell 10]
```python
class PeriodicSpaceConv2d(nn.Module):
    """Conv2d over (time, space). Space gets periodic padding, time gets zero padding."""
    def __init__(self, in_ch, out_ch, k_t=3, k_x=3):
        super().__init__()
        self.k_t = k_t
        self.k_x = k_x
        # Inner conv has 'valid' behavior; we pad manually.
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=(k_t, k_x), padding=0)

    def forward(self, x):  # x: (B, C, T, X)
        pt = (self.k_t - 1) // 2
        px = (self.k_x - 1) // 2
        x = F.pad(x, (px, px, 0, 0), mode='circular')  # wrap space
        x = F.pad(x, (0, 0, pt, pt), mode='constant')  # zero time
        return self.conv(x)


class InverseObsLorenz96(nn.Module):
    def __init__(self, obs_grid=10, full_grid=40, hidden=32, n_layers=6):
        super().__init__()
        self.obs_grid = obs_grid
        self.full_grid = full_grid
        self.in_proj = PeriodicSpaceConv2d(1, hidden, 3, 3)
        self.blocks = nn.ModuleList(
            [PeriodicSpaceConv2d(hidden, hidden, 3, 3) for _ in range(n_layers)]
        )
        self.out_proj = PeriodicSpaceConv2d(hidden, 1, 3, 3)

    def forward(self, y):  # y: (B, T, obs_grid) -> (B, T, full_grid)
        B, T, _ = y.shape
        x = y.unsqueeze(1)  # (B, 1, T, obs_grid)
        # Spatial upsample with periodic-aware linear interp:
        x = F.interpolate(x, size=(T, self.full_grid), mode='bilinear', align_corners=False)
        x = F.gelu(self.in_proj(x))
        for blk in self.blocks:
            x = x + F.gelu(blk(x))
        x = self.out_proj(x)
        return x.squeeze(1)
```

## [md cell 11]
### Train the inverse observation operator

Supervised regression: integrate trajectories, observe them, teach the net to invert $H$.

## [code cell 12]
```python
def train_inverter(dyn_sys, inverter, n_train=4000, T_train=20, n_warmup=1000,
                   n_epochs=500, batch_size=8, lr=1e-3, obs_noise_std=0.0, log_every=5):
    # Build a dataset once.
    _, X, Y, _ = generate_data(dyn_sys, n_train, T_train, n_warmup,
                                obs_noise_std=obs_noise_std, seed=42)
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
            print(f'  epoch {epoch:3d}  loss={ep_loss:.4f}')
    return history


inverter = InverseObsLorenz96(obs_grid=10, full_grid=40, hidden=32, n_layers=6).to(device)
ckpt = load_cache('l96_inverter_sigma0.0.pt')
if ckpt is None:
    hist = train_inverter(L96, inverter, n_train=400, T_train=20, n_epochs=500,
                          batch_size=8, obs_noise_std=0.0)
    save_cache({'state_dict': inverter.state_dict(), 'hist': hist}, 'l96_inverter_sigma0.0.pt')
else:
    inverter.load_state_dict(ckpt['state_dict'])
    hist = ckpt['hist']

plt.plot(hist); plt.yscale('log'); plt.xlabel('epoch'); plt.ylabel('MSE')
plt.title('Inverse-obs training'); plt.show()
```

>>> FIGURE EMBEDDED: figures/L96_SlidingWindow__cell12__out00.png

--- output ---
  [cache] loaded l96_inverter_sigma0.0.pt
<Figure size 640x480 with 1 Axes>

## [md cell 13]
### Variational DA

Three optimization modes from the paper:
- **obs-space**: minimize $\|H(M(x_0)) - y\|^2$
- **physics-space**: minimize $\|M(x_0) - H^{-1}_\theta(y)\|^2$
- **hybrid**: physics-space warm-start → obs-space refinement

All operate in decorrelated coordinates $z = C^{-1/2} x$ (classic 4D-Var preconditioning).

## [code cell 14]
```python
def decorrelate(x, C_inv_sqrt):
    # x: (..., grid) -> z: (..., grid)
    return x @ C_inv_sqrt  # C is symmetric so left/right multiply is equivalent


def correlate(z, C_sqrt):
    return z @ C_sqrt


def da_loss(Z0, Y, dyn_sys, C_sqrt, T, mode, inverter=None):
    """Batched DA loss.
    Z0: (N, grid) decorrelated initial states.
    Y:  (N, T, obs_grid) observations.
    Returns scalar (mean over N samples, T steps, and obs components).
    """
    X0 = correlate(Z0, C_sqrt)           # (N, grid)
    traj = dyn_sys.integrate(X0, T)      # (T, N, grid) — rhs broadcasts over leading dims
    if mode == 'obs':
        pred = dyn_sys.observe(traj)     # (T, N, obs_grid)
        target = Y.transpose(0, 1)       # (T, N, obs_grid)
    elif mode == 'physics':
        assert inverter is not None
        inv = inverter(Y).detach()       # (N, T, grid)
        target = inv.transpose(0, 1)     # (T, N, grid)
        pred = traj
    else:
        raise ValueError(mode)
    return ((pred - target) ** 2).mean()


def lbfgs_minimize(loss_fn, z0_init, max_iter=200, history_size=20, lr=1.0):
    z = z0_init.clone().detach().requires_grad_(True)
    opt = torch.optim.LBFGS([z], max_iter=max_iter, history_size=history_size,
                            tolerance_grad=1e-12, tolerance_change=1e-12,
                            line_search_fn='strong_wolfe', lr=lr)
    history = []

    def closure():
        opt.zero_grad()
        loss = loss_fn(z)
        loss.backward()
        history.append(loss.item())
        return loss

    opt.step(closure)
    return z.detach(), history
```

## [code cell 15]
```python
def run_da(dyn_sys, inverter, corr, X0_init, Y, T,
           physics_steps=0, obs_steps=500):
    """Batched DA: all N samples optimized in a single L-BFGS call.
    X0_init: (N, grid), Y: (N, T, obs_grid). Returns X0_opt: (N, grid) and loss history."""
    C_sqrt, C_inv_sqrt = corr['C_sqrt'], corr['C_inv_sqrt']
    Z0 = decorrelate(X0_init, C_inv_sqrt)
    hist = []
    if physics_steps > 0:
        loss_p = partial(da_loss, Y=Y, dyn_sys=dyn_sys, C_sqrt=C_sqrt, T=T,
                         mode='physics', inverter=inverter)
        Z0, h1 = lbfgs_minimize(loss_p, Z0, max_iter=physics_steps)
        hist.extend(h1)
    if obs_steps > 0:
        loss_o = partial(da_loss, Y=Y, dyn_sys=dyn_sys, C_sqrt=C_sqrt, T=T,
                         mode='obs')
        Z0, h2 = lbfgs_minimize(loss_o, Z0, max_iter=obs_steps)
        hist.extend(h2)
    X0_opt = correlate(Z0, C_sqrt)
    return X0_opt, hist


def baseline_init_l96(dyn_sys, Y):
    """Batched baseline init: nearest-neighbor upsample from t=0 observations.
    Y: (N, T, obs_grid) or (T, obs_grid). Returns (N, grid) or (grid,)."""
    if Y.ndim == 3:
        return Y[:, 0].repeat_interleave(dyn_sys.observe_every, dim=-1)
    return Y[0].repeat_interleave(dyn_sys.observe_every)
```

## [md cell 16]
### Full 4D-Var (J_b + J_o) with separable background

## [code cell 17]
```python
# 4D-Var cost functions in decorrelated z-coordinates.
# Both operate on batched z0 of shape (N, grid); each row is independently optimized
# (its gradient depends only on its own observations), so L-BFGS over the flat (N*grid)
# vector is equivalent to N independent 4D-Var problems.

def var4d_cost_obs(z0, y_T, dyn_sys, C_sqrt, T, z_b, R_inv_diag, sigma_b):
    """Standard 4D-Var: J_b + J_o.

    z0, z_b: (N, grid).  y_T: (T, N, obs_grid) (already transposed).
    With B = sigma_b^2 * C, decorrelating gives J_b = ||z0 - z_b||^2 / (2 sigma_b^2).
    """
    x0 = correlate(z0, C_sqrt)
    traj = dyn_sys.integrate(x0, T)            # (T, N, grid)
    y_pred = dyn_sys.observe(traj)             # (T, N, obs_grid)
    innov = y_pred - y_T
    J_o = 0.5 * (innov.pow(2) * R_inv_diag).sum()
    J_b = 0.5 * ((z0 - z_b).pow(2)).sum() / (sigma_b ** 2)
    return J_b + J_o


def var4d_cost_phys(z0, target_traj, dyn_sys, C_sqrt, T, z_b, sigma_b, sigma_p):
    """Physics-space companion cost: J_b + (1/2 sigma_p^2) ||M(x_0) - H^{-1}_theta(y)||^2.

    target_traj: precomputed (T, N, grid) = inverter(Y).detach().transpose(0,1).
    """
    x0 = correlate(z0, C_sqrt)
    traj = dyn_sys.integrate(x0, T)            # (T, N, grid)
    J_phys = 0.5 * ((traj - target_traj).pow(2)).sum() / (sigma_p ** 2)
    J_b = 0.5 * ((z0 - z_b).pow(2)).sum() / (sigma_b ** 2)
    return J_b + J_phys
```

## [code cell 18]
```python
def invobs_init_l96(inverter, Y):
    """Background estimate from H^{-1}_theta evaluated at the observation sequence.
    Y: (N, T, obs_grid) -> (N, grid). Uses the first frame of the inverted trajectory."""
    return inverter(Y).detach()[:, 0]


def run_4dvar_l96(dyn_sys, inverter, corr, X0_init, Y, T,
                  sigma_b=1.0, sigma_obs=0.5, sigma_p=0.5,
                  mode='obs', physics_steps=200, obs_steps=300,
                  X_background=None):
    """Full 4D-Var driver.

    X0_init   : (N, grid) starting point for the L-BFGS optimizer.
    X_background : optional (N, grid) background state used in J_b. If None,
                   X0_init is used as the background (single-window default).
                   In cycling DA, pass the propagated previous analysis here
                   while X0_init can be a separate (e.g. invobs) starting point.

    mode = 'obs'    : minimize J_b + J_o for `obs_steps` L-BFGS iterations.
    mode = 'hybrid' : first minimize J_b + J_phys for `physics_steps` iters,
                      then J_b + J_o.

    Returns (X0_opt: (N, grid), loss_history: list).
    """
    C_sqrt, C_inv_sqrt = corr['C_sqrt'], corr['C_inv_sqrt']
    x_b = X_background if X_background is not None else X0_init
    z_b = decorrelate(x_b, C_inv_sqrt)             # background for cost function
    z = decorrelate(X0_init, C_inv_sqrt).clone()   # starting point for optimizer

    R_inv_diag = torch.full((dyn_sys.grid_size // dyn_sys.observe_every,),
                            1.0 / (sigma_obs ** 2), device=device)
    Y_T = Y.transpose(0, 1)                                 # (T, N, obs_grid)
    history = []

    if mode == 'hybrid':
        target_traj = inverter(Y).detach().transpose(0, 1)  # (T, N, grid)
        loss_phys = partial(var4d_cost_phys, target_traj=target_traj,
                            dyn_sys=dyn_sys, C_sqrt=C_sqrt, T=T,
                            z_b=z_b, sigma_b=sigma_b, sigma_p=sigma_p)
        z, h_p = lbfgs_minimize(loss_phys, z, max_iter=physics_steps)
        history.extend(h_p)

    loss_obs = partial(var4d_cost_obs, y_T=Y_T, dyn_sys=dyn_sys, C_sqrt=C_sqrt, T=T,
                       z_b=z_b, R_inv_diag=R_inv_diag, sigma_b=sigma_b)
    z, h_o = lbfgs_minimize(loss_obs, z, max_iter=obs_steps)
    history.extend(h_o)

    X0_opt = correlate(z, C_sqrt)
    return X0_opt, history
```

## [md cell 19]
### Sliding-window cycling 4D-Var

Invobs init at the start of every cycle; for cycles >= 1 the J_b background is the propagated previous analysis (not the invobs estimate). The last window is always assimilated.

## [code cell 20]
```python
def run_sliding_window_4dvar_l96(
    dyn_sys, inverter, corr, Y_long,
    window_T=8,
    stride=1,
    sigma_b=0.3,
    sigma_obs=0.5,
    sigma_p=0.5,
    init_mode='invobs',
    opt_mode='hybrid',
    physics_steps=100,
    obs_steps=200,
):
    """Sliding-window / cycling 4D-Var.

    Each cycle uses invobs_init_l96 (or baseline_init_l96 for cycle 0 if
    init_mode='baseline') as the **starting point** for L-BFGS. For cycle 0
    the J_b background equals that starting point. For cycles >= 1 the J_b
    background is the propagated previous analysis (`X_background=xb`), while
    the optimizer still starts from the fresh invobs estimate of the new
    window. This decouples the two roles X0_init used to play in
    run_4dvar_l96.

    Returns dict with
        starts        : list of window start indices in Y_long
        analyses      : (N, n_cycles, grid)  optimized x at each window start
        backgrounds   : (N, n_cycles, grid)  J_b background actually used
        invobs_inits  : (N, n_cycles, grid)  L-BFGS starting point per cycle
        histories     : list (length n_cycles) of L-BFGS loss histories
    """
    N, T_total, _ = Y_long.shape
    starts = list(range(0, T_total - window_T + 1, stride))
    # Guarantee the most recent window is always assimilated, even when
    # (T_total - window_T) is not divisible by stride.
    if starts[-1] != T_total - window_T:
        starts.append(T_total - window_T)

    analyses = []
    backgrounds = []
    invobs_inits = []
    histories = []

    xb = None  # propagated analysis from previous cycle

    for c, start in enumerate(starts):
        Y_win = Y_long[:, start:start + window_T]

        # Starting point for L-BFGS: always invobs (or baseline on cycle 0
        # when init_mode requests it).
        if c == 0 and init_mode == 'baseline':
            x_start = baseline_init_l96(dyn_sys, Y_win)
        else:
            x_start = invobs_init_l96(inverter, Y_win)

        # Background for J_b
        if c == 0:
            X_background = None     # == x_start (cold start)
            x_b_record = x_start
        else:
            X_background = xb       # propagated previous analysis
            x_b_record = xb

        X0_opt, hist = run_4dvar_l96(
            dyn_sys=dyn_sys,
            inverter=inverter,
            corr=corr,
            X0_init=x_start,
            Y=Y_win,
            T=window_T,
            sigma_b=sigma_b,
            sigma_obs=sigma_obs,
            sigma_p=sigma_p,
            mode=opt_mode,
            physics_steps=physics_steps,
            obs_steps=obs_steps,
            X_background=X_background,
        )

        analyses.append(X0_opt.detach())
        backgrounds.append(x_b_record.detach())
        invobs_inits.append(x_start.detach())
        histories.append(hist)

        # Forecast analysis forward to next window's start time.
        if c < len(starts) - 1:
            step = starts[c + 1] - start
            xb = dyn_sys.integrate(X0_opt.detach(), step + 1)[-1].detach()

    return {
        'starts': starts,
        'analyses': torch.stack(analyses, dim=1),       # (N, n_cycles, grid)
        'backgrounds': torch.stack(backgrounds, dim=1),  # (N, n_cycles, grid)
        'invobs_inits': torch.stack(invobs_inits, dim=1),
        'histories': histories,
    }
```

## [md cell 21]
---
## 2. Setup

## [code cell 22]
```python
# ---- Sliding-window experiment configuration -------------------------------
WINDOW_T       = 8        # per-cycle assimilation window (always)
T_FORECAST     = 50       # forecast horizon evaluated after last window ends
T_GENERATE     = 103      # 48 + 50 + 5 margin
N_EVAL         = 50       # ensemble size for evaluation
N_WARMUP       = 1000
NOISE_LEVELS   = [0.0, 0.5]
SIGMA_B_COLD   = 1.0      # single-window (no history)
SIGMA_B_CYCLE  = 0.3      # cycling (trusts propagated analysis)
PHYSICS_STEPS  = 100
OBS_STEPS      = 400

T_OBS_TOTALS_B = [24, 48]
STRIDES        = [2, 4, 8]

# Sigma_obs is used as the inverse-variance weighting in J_o. Floor it so the
# perfect-observation case (sigma=0) still produces a finite cost.
def sigma_obs_eff(s):
    return max(float(s), 0.1)

DATA_SEED      = 12345
```

## [code cell 23]
```python
# ---- Train sigma=0.5 inverter ---------------------------------------------
inverter_05 = InverseObsLorenz96(obs_grid=10, full_grid=40, hidden=32, n_layers=6).to(device)
ckpt_05 = load_cache('l96_inverter_sigma0.5.pt')
if ckpt_05 is None:
    hist_05 = train_inverter(L96, inverter_05, n_train=400, T_train=20, n_epochs=500,
                              batch_size=8, obs_noise_std=0.5)
    save_cache({'state_dict': inverter_05.state_dict(), 'hist': hist_05},
               'l96_inverter_sigma0.5.pt')
else:
    inverter_05.load_state_dict(ckpt_05['state_dict'])
    hist_05 = ckpt_05['hist']

inverters = {0.0: inverter, 0.5: inverter_05}
print(f'Inverters ready for noise levels: {sorted(inverters)}')

# ---- Evaluation dataset per noise level ------------------------------------
# One long trajectory per noise level (length T_GENERATE) covering all
# T_OBS_TOTAL values up to 48 plus the T_FORECAST horizon.
eval_data = {}
for s in NOISE_LEVELS:
    X0, X_long, Y_long, Y_clean = generate_data(
        L96, n_samples=N_EVAL, n_time_steps=T_GENERATE, n_warmup=N_WARMUP,
        obs_noise_std=s, seed=DATA_SEED,
    )
    eval_data[s] = {
        'X0': X0.detach(),           # (N, grid)
        'X_true': X_long.detach(),    # (N, T_GENERATE, grid)
        'Y_obs': Y_long.detach(),     # (N, T_GENERATE, obs_grid)
    }
    print(f'  sigma={s}: X_true {tuple(X_long.shape)}, Y_obs {tuple(Y_long.shape)}')
```
--- output ---
  [cache] loaded l96_inverter_sigma0.5.pt
Inverters ready for noise levels: [0.0, 0.5]
  sigma=0.0: X_true (50, 103, 40), Y_obs (50, 103, 10)
  sigma=0.5: X_true (50, 103, 40), Y_obs (50, 103, 10)

## [code cell 24]
```python
# ---- Forecast / RMSE helpers ----------------------------------------------
def spatial_rmse(pred, truth):
    """pred, truth: (..., grid) -> (...,) sqrt mean square over grid axis."""
    return (pred - truth).pow(2).mean(dim=-1).sqrt()


def forecast_curves(dyn_sys, X_analysis, X_truth_long, t_analysis, t_end, T_forecast):
    """Integrate X_analysis from t_analysis to t_end+T_forecast and compute
    spatial RMSE vs truth for the forecast period [t_end, t_end+T_forecast].

    X_analysis  : (N, grid)
    X_truth_long: (N, T_GENERATE, grid)
    Returns (mean_rmse, std_rmse) of shape (T_forecast + 1,) keyed by lead time.
    """
    n_steps = t_end - t_analysis + T_forecast + 1
    traj_full = dyn_sys.integrate(X_analysis, n_steps)        # (n_steps, N, grid)
    traj_fcst = traj_full[t_end - t_analysis:]                # (T_forecast+1, N, grid)
    truth_seg = X_truth_long[:, t_end:t_end + T_forecast + 1].transpose(0, 1)
    err = spatial_rmse(traj_fcst, truth_seg)                  # (T_forecast+1, N)
    return err.mean(dim=1).cpu().numpy(), err.std(dim=1).cpu().numpy(), traj_fcst.detach()


def full_rmse_curve(dyn_sys, X_analysis, X_truth_long, t_analysis, t_end, T_forecast):
    """Spatial RMSE over the complete span [t_analysis, t_end + T_forecast].

    Returns (mean, std) arrays of length (t_end - t_analysis + T_forecast + 1).
    Index 0  = t_analysis (window start).
    Index (t_end - t_analysis) = t_end (window end / lead = 0).
    Last index = t_end + T_forecast.

    Pair with x = np.arange(-(t_end - t_analysis), T_forecast + 1) so x=0
    is the window end, negative x = inside the assimilation window.
    """
    n_steps = t_end - t_analysis + T_forecast + 1
    with torch.no_grad():
        traj = dyn_sys.integrate(X_analysis, n_steps)         # (n_steps, N, grid)
    truth = X_truth_long[:, t_analysis:t_analysis + n_steps].transpose(0, 1)
    err = spatial_rmse(traj, truth)                           # (n_steps, N)
    return err.mean(dim=1).cpu().numpy(), err.std(dim=1).cpu().numpy()


def analysis_rmse(X_analysis, X_truth_long, t_analysis):
    """Spatial RMSE between X_analysis and the truth state at t_analysis.
    Returns scalar mean and std across the ensemble.
    """
    truth = X_truth_long[:, t_analysis]                       # (N, grid)
    err = spatial_rmse(X_analysis, truth)                     # (N,)
    return float(err.mean().cpu()), float(err.std().cpu())
```

## [md cell 25]
---
## 3. Experiment A — Single window (`T_OBS_TOTAL = 8`)

Four init x opt combinations on a single 8-step assimilation window. All use the cold-start background variance SIGMA_B_COLD = 1.0.

## [code cell 26]
```python
# ---- Experiment A: single-window, 4 init x opt combos ---------------------
T_OBS_TOTAL_A = WINDOW_T  # 8

combos_A = [
    ('invobs + hybrid',   'invobs',   'hybrid'),
    ('invobs + obs-only', 'invobs',   'obs'),
    ('baseline + hybrid', 'baseline', 'hybrid'),
    ('baseline + obs-only','baseline','obs'),
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
                X0_init = invobs_init_l96(inv, Y_win)
            else:
                X0_init = baseline_init_l96(L96, Y_win)
            if opt == 'hybrid':
                ps, os_ = PHYSICS_STEPS, OBS_STEPS
            else:
                ps, os_ = 0, PHYSICS_STEPS + OBS_STEPS
            X0_opt, hist = run_4dvar_l96(
                L96, inv, corr,
                X0_init=X0_init, Y=Y_win, T=WINDOW_T,
                sigma_b=SIGMA_B_COLD, sigma_obs=s_obs, sigma_p=0.5,
                mode=opt, physics_steps=ps, obs_steps=os_,
            )
            per_sigma[label] = {
                'X0_opt': X0_opt.detach().cpu(),
                'hist': hist,
            }
            print(f'  sigma={sigma}  {label:<22s}  iters={len(hist)}  final_loss={hist[-1]:.4f}')
        results_A[sigma] = per_sigma
    save_cache(results_A, 'expA_results.pt')

# Compute forecast / analysis RMSE summaries for plotting and Exp C.
summary_A = {}
for sigma in NOISE_LEVELS:
    d = eval_data[sigma]
    per_sigma = {}
    for label, _, _ in combos_A:
        X0_opt = results_A[sigma][label]['X0_opt'].to(device)
        mean_f, std_f, _ = forecast_curves(L96, X0_opt, d['X_true'],
                                            t_analysis=0, t_end=WINDOW_T,
                                            T_forecast=T_FORECAST)
        a_mean, a_std = analysis_rmse(X0_opt, d['X_true'], t_analysis=0)
        per_sigma[label] = {
            'fcst_mean': mean_f, 'fcst_std': std_f,
            'ana_mean': a_mean, 'ana_std': a_std,
        }
    summary_A[sigma] = per_sigma
print('Experiment A summary ready.')
```
--- output ---
  [cache] loaded expA_results.pt
Experiment A summary ready.

## [code cell 27]
```python
# ---- Plot A: RMSE over assimilation window + forecast + analysis bar ------
# x-axis is absolute time step (t=0 = window start, t=WINDOW_T = window end).
COLORS_A = {
    'invobs + hybrid':    '#0072B2',
    'invobs + obs-only':  '#56B4E9',
    'baseline + hybrid':  '#D55E00',
    'baseline + obs-only':'#E69F00',
}

fig = plt.figure(figsize=(14, 8))
gs = fig.add_gridspec(2, 2, height_ratios=[1.4, 1.0], hspace=0.35, wspace=0.25)
t_range_A = np.arange(0, WINDOW_T + T_FORECAST + 1)   # 0 … 58

for col, sigma in enumerate(NOISE_LEVELS):
    d = eval_data[sigma]
    ax = fig.add_subplot(gs[0, col])
    ax.axvspan(0, WINDOW_T, alpha=0.07, color='steelblue', zorder=0,
               label='assimilation window')
    for label, _, _ in combos_A:
        X0_opt = results_A[sigma][label]['X0_opt'].to(device)
        m, s = full_rmse_curve(L96, X0_opt, d['X_true'],
                               t_analysis=0, t_end=WINDOW_T, T_forecast=T_FORECAST)
        c = COLORS_A[label]
        ax.plot(t_range_A, m, color=c, lw=1.8, label=label)
        ax.fill_between(t_range_A, m - s, m + s, color=c, alpha=0.15, linewidth=0)
    ax.axvline(WINDOW_T, color='gray', ls='--', lw=1.3, zorder=2, label='window end')
    ax.set_xlim(0, WINDOW_T + T_FORECAST)
    ax.set_title(f'Experiment A: RMSE  (sigma_obs={sigma})')
    ax.set_xlabel('Time step')
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

>>> FIGURE EMBEDDED: figures/L96_SlidingWindow__cell27__out00.png

--- output ---
<Figure size 1400x800 with 3 Axes>

## [md cell 28]
---
## 4. Experiment B — Sliding-window stride sweep

Invobs init every cycle, hybrid opt, propagated background for cycles >= 1. SIGMA_B_CYCLE = 0.3. Sweep strides [2, 4, 8] for each T_OBS_TOTAL in {24, 48}.

## [code cell 29]
```python
# ---- Experiment B: stride sweep across T_OBS_TOTAL in {24, 48} ------------
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
                out = run_sliding_window_4dvar_l96(
                    L96, inv, corr, Y_long,
                    window_T=WINDOW_T,
                    stride=stride,
                    sigma_b=SIGMA_B_CYCLE,
                    sigma_obs=s_obs,
                    sigma_p=0.5,
                    init_mode='invobs',
                    opt_mode='hybrid',
                    physics_steps=PHYSICS_STEPS,
                    obs_steps=OBS_STEPS,
                )
                sw_results[key] = {
                    'starts': out['starts'],
                    'analyses': out['analyses'].detach().cpu(),
                    'backgrounds': out['backgrounds'].detach().cpu(),
                    'invobs_inits': out['invobs_inits'].detach().cpu(),
                    'histories': out['histories'],
                }
    save_cache(sw_results, 'expB_results.pt')

# Build forecast summary per (T_obs, sigma, stride) and select best_stride.
summary_B = {}
best_stride = {}
for T_obs in T_OBS_TOTALS_B:
    for sigma in NOISE_LEVELS:
        d = eval_data[sigma]
        per_stride = {}
        for stride in STRIDES:
            key = (T_obs, sigma, stride)
            r = sw_results[key]
            t_analysis = r['starts'][-1]
            X_last_an = r['analyses'][:, -1].to(device)
            mean_f, std_f, _ = forecast_curves(L96, X_last_an, d['X_true'],
                                                t_analysis=t_analysis,
                                                t_end=T_obs,
                                                T_forecast=T_FORECAST)
            a_mean, a_std = analysis_rmse(X_last_an, d['X_true'], t_analysis=t_analysis)
            per_stride[stride] = {
                'fcst_mean': mean_f, 'fcst_std': std_f,
                'ana_mean': a_mean, 'ana_std': a_std,
                'starts': r['starts'],
            }
        summary_B[(T_obs, sigma)] = per_stride
        # Pick stride that minimizes mean forecast RMSE averaged over lead times.
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
  [cache] loaded expB_results.pt
  T_obs=24 sigma=0.0  best stride = 8
  T_obs=24 sigma=0.5  best stride = 2
  T_obs=48 sigma=0.0  best stride = 8
  T_obs=48 sigma=0.5  best stride = 8
  sigma=0.0  best Exp-A method = baseline + hybrid
  sigma=0.5  best Exp-A method = invobs + hybrid

## [code cell 30]
```python
# ---- Plot B1: RMSE per stride — absolute time axis, from t=0 -------------
# Blue shading = last assimilation window [T_obs-WINDOW_T, T_obs].
# Dashed bar = T_obs (window end). xlim starts at 0 for all subplots.
# Reference (best Exp-A) is shifted to align with the last-window time position
# [t_win_start, T_obs+T_FORECAST] so its window end coincides with the SW runs'.
STRIDE_COLORS = {2: '#1b9e77', 4: '#d95f02', 8: '#7570b3'}

fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharey='row')
for row_idx, T_obs in enumerate(T_OBS_TOTALS_B):
    t_win_start = T_obs - WINDOW_T          # e.g. 16 (T_obs=24) or 40 (T_obs=48)
    t_range_sw  = np.arange(t_win_start, T_obs + T_FORECAST + 1)
    # Exp-A was run on a single 8-step window; shift its time axis so the window
    # start maps to t_win_start and the window end maps to T_obs — matching the
    # last SW window's position for an apples-to-apples comparison.
    t_range_ref = np.arange(t_win_start, T_obs + T_FORECAST + 1)

    for col_idx, sigma in enumerate(NOISE_LEVELS):
        d  = eval_data[sigma]
        ax = axes[row_idx, col_idx]
        ax.axvspan(t_win_start, T_obs, alpha=0.07, color='steelblue', zorder=0)

        for stride in STRIDES:
            rec    = sw_results[(T_obs, sigma, stride)]
            X_last = rec['analyses'][:, -1].to(device)
            t_a    = rec['starts'][-1]      # guaranteed == T_obs - WINDOW_T
            m, s = full_rmse_curve(L96, X_last, d['X_true'],
                                   t_analysis=t_a, t_end=T_obs, T_forecast=T_FORECAST)
            color = STRIDE_COLORS[stride]
            ax.plot(t_range_sw, m, color=color, lw=1.8, label=f'stride={stride}')
            ax.fill_between(t_range_sw, m - s, m + s, color=color, alpha=0.15, linewidth=0)

        # Reference: best Exp-A RMSE values plotted at the last window's time position.
        X0_ref = results_A[sigma][best_A[sigma]]['X0_opt'].to(device)
        m_ref, s_ref = full_rmse_curve(L96, X0_ref, d['X_true'],
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
            ax.set_xlabel('Time step')
        if col_idx == 0:
            ax.set_ylabel('Spatial RMSE')
        if row_idx == 0 and col_idx == 1:
            ax.legend(fontsize=8, loc='upper left')

fig.suptitle('Experiment B1: RMSE per stride — assimilation window + forecast\n'
             '(shaded = last assimilation window; Exp-A reference shifted to last window position)', y=1.03)
plt.tight_layout()
plt.show()
```

>>> FIGURE EMBEDDED: figures/L96_SlidingWindow__cell30__out00.png

--- output ---
<Figure size 1300x900 with 4 Axes>

## [code cell 31]
```python
# ---- Plot B2: per-cycle background vs analysis RMSE for best_stride run ----
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
for r, T_obs in enumerate(T_OBS_TOTALS_B):
    for c, sigma in enumerate(NOISE_LEVELS):
        ax = axes[r, c]
        st = best_stride[(T_obs, sigma)]
        rec = sw_results[(T_obs, sigma, st)]
        d = eval_data[sigma]
        analyses = rec['analyses'].to(device)        # (N, n_cycles, grid)
        backgrounds = rec['backgrounds'].to(device)
        starts = rec['starts']
        ana_curve = []
        bg_curve = []
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

>>> FIGURE EMBEDDED: figures/L96_SlidingWindow__cell31__out00.png

--- output ---
<Figure size 1300x900 with 4 Axes>

## [md cell 32]
---
## 5. Experiment C — Final comparison

SW-best (best stride from Experiment B, full history) vs invobs+hybrid on the last 8 observations vs baseline+obs-only on the last 8 observations.

## [code cell 33]
```python
# ---- Experiment C: SW-best vs single-window methods on last 8 obs ----------
results_C = load_cache('expC_results.pt')
if results_C is None:
    results_C = {}
    for T_obs in T_OBS_TOTALS_B:
        for sigma in NOISE_LEVELS:
            d = eval_data[sigma]
            inv = inverters[sigma]
            s_obs = sigma_obs_eff(sigma)
            # SW-best: reuse from sw_results
            st = best_stride[(T_obs, sigma)]
            rec = sw_results[(T_obs, sigma, st)]

            # Last-window-only methods: window = Y[:, T_obs-WINDOW_T:T_obs]
            Y_win = d['Y_obs'][:, T_obs - WINDOW_T:T_obs]

            # invobs + hybrid (single window, cold start)
            X0_init_invh = invobs_init_l96(inv, Y_win)
            X0_invh, h_invh = run_4dvar_l96(
                L96, inv, corr,
                X0_init=X0_init_invh, Y=Y_win, T=WINDOW_T,
                sigma_b=SIGMA_B_COLD, sigma_obs=s_obs, sigma_p=0.5,
                mode='hybrid', physics_steps=PHYSICS_STEPS, obs_steps=OBS_STEPS,
            )
            # baseline + obs-only
            X0_init_base = baseline_init_l96(L96, Y_win)
            X0_base, h_base = run_4dvar_l96(
                L96, inv, corr,
                X0_init=X0_init_base, Y=Y_win, T=WINDOW_T,
                sigma_b=SIGMA_B_COLD, sigma_obs=s_obs, sigma_p=0.5,
                mode='obs', physics_steps=0, obs_steps=PHYSICS_STEPS + OBS_STEPS,
            )
            results_C[(T_obs, sigma)] = {
                'best_stride': st,
                'sw_last_analysis': rec['analyses'][:, -1].cpu(),
                'sw_last_start': rec['starts'][-1],
                'invh_X0': X0_invh.detach().cpu(),
                'base_X0': X0_base.detach().cpu(),
            }
            print(f'  T_obs={T_obs} sigma={sigma}  done.')
    save_cache(results_C, 'expC_results.pt')

# Summary curves keyed by (T_obs, sigma) and method label.
METHOD_LABELS_C = ['SW-best', 'invobs + hybrid (last 8)', 'baseline + obs-only (last 8)']
summary_C = {}
for T_obs in T_OBS_TOTALS_B:
    for sigma in NOISE_LEVELS:
        d = eval_data[sigma]
        r = results_C[(T_obs, sigma)]
        per_method = {}

        # SW-best
        X_sw = r['sw_last_analysis'].to(device)
        m_sw, s_sw, traj_sw = forecast_curves(L96, X_sw, d['X_true'],
                                               t_analysis=r['sw_last_start'],
                                               t_end=T_obs, T_forecast=T_FORECAST)
        a_sw_m, a_sw_s = analysis_rmse(X_sw, d['X_true'], t_analysis=r['sw_last_start'])
        per_method['SW-best'] = {
            'fcst_mean': m_sw, 'fcst_std': s_sw, 'traj': traj_sw,
            'ana_mean': a_sw_m, 'ana_std': a_sw_s,
            't_analysis': r['sw_last_start'],
        }

        # invobs + hybrid (last 8)
        X_ih = r['invh_X0'].to(device)
        m_ih, s_ih, traj_ih = forecast_curves(L96, X_ih, d['X_true'],
                                                t_analysis=T_obs - WINDOW_T,
                                                t_end=T_obs, T_forecast=T_FORECAST)
        a_ih_m, a_ih_s = analysis_rmse(X_ih, d['X_true'], t_analysis=T_obs - WINDOW_T)
        per_method['invobs + hybrid (last 8)'] = {
            'fcst_mean': m_ih, 'fcst_std': s_ih, 'traj': traj_ih,
            'ana_mean': a_ih_m, 'ana_std': a_ih_s,
            't_analysis': T_obs - WINDOW_T,
        }

        # baseline + obs-only (last 8)
        X_bs = r['base_X0'].to(device)
        m_bs, s_bs, traj_bs = forecast_curves(L96, X_bs, d['X_true'],
                                                t_analysis=T_obs - WINDOW_T,
                                                t_end=T_obs, T_forecast=T_FORECAST)
        a_bs_m, a_bs_s = analysis_rmse(X_bs, d['X_true'], t_analysis=T_obs - WINDOW_T)
        per_method['baseline + obs-only (last 8)'] = {
            'fcst_mean': m_bs, 'fcst_std': s_bs, 'traj': traj_bs,
            'ana_mean': a_bs_m, 'ana_std': a_bs_s,
            't_analysis': T_obs - WINDOW_T,
        }
        summary_C[(T_obs, sigma)] = per_method
print('Experiment C summary ready.')
```
--- output ---
  [cache] loaded expC_results.pt
Experiment C summary ready.

## [code cell 34]
```python
# ---- Plot C: RMSE — absolute time axis, from t=0 (SW vs single-window) ---
# Blue shading = last assimilation window [T_obs-WINDOW_T, T_obs].
# SW-best: per-cycle analysis RMSE chain from t=0 (dashed + markers) leads into
#          the continuous RMSE curve for the last cycle (solid), extending into
#          the forecast.  Single-window baselines still start at t_win_start.
COLORS_C = {
    'SW-best':                      '#0072B2',
    'invobs + hybrid (last 8)':     '#009E73',
    'baseline + obs-only (last 8)': '#D55E00',
}

fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharey='row')
for row_idx, T_obs in enumerate(T_OBS_TOTALS_B):
    t_win_start = T_obs - WINDOW_T          # e.g. 16 or 40

    for col_idx, sigma in enumerate(NOISE_LEVELS):
        d  = eval_data[sigma]
        rc = results_C[(T_obs, sigma)]
        ax = axes[row_idx, col_idx]
        ax.axvspan(t_win_start, T_obs, alpha=0.07, color='steelblue', zorder=0)

        # ---- SW-best: per-cycle RMSE chain from t=0 + last-cycle continuous ----
        st_sw   = rc['best_stride']
        rec_sw  = sw_results[(T_obs, sigma, st_sw)]
        sw_starts   = rec_sw['starts']
        sw_analyses = rec_sw['analyses'].to(device)   # (N, n_cycles, grid)

        # Compute RMSE of each cycle's analysis vs truth at that cycle's window start.
        chain_t, chain_m, chain_s = [], [], []
        for ci, t_a in enumerate(sw_starts):
            err = spatial_rmse(sw_analyses[:, ci], d['X_true'][:, t_a])  # (N,)
            chain_t.append(t_a)
            chain_m.append(float(err.mean().cpu()))
            chain_s.append(float(err.std().cpu()))
        chain_t = np.array(chain_t)
        chain_m = np.array(chain_m)
        chain_s = np.array(chain_s)

        # Continuous RMSE for the last cycle (window + forecast).
        m_sw, s_sw = full_rmse_curve(L96, sw_analyses[:, -1], d['X_true'],
                                      t_analysis=sw_starts[-1], t_end=T_obs,
                                      T_forecast=T_FORECAST)
        t_range_sw_last = np.arange(sw_starts[-1], T_obs + T_FORECAST + 1)

        color_sw = COLORS_C['SW-best']
        # Dashed chain: per-cycle analysis RMSE from t=0 up to (and including) last window start.
        ax.plot(chain_t, chain_m, color=color_sw, lw=1.4, ls='--',
                marker='o', markersize=4, zorder=3, label='SW-best (per-cycle analyses)')
        ax.fill_between(chain_t, chain_m - chain_s, chain_m + chain_s,
                        color=color_sw, alpha=0.12, linewidth=0)
        # Solid line: continuous RMSE for the last cycle and forecast.
        ax.plot(t_range_sw_last, m_sw, color=color_sw, lw=1.8)
        ax.fill_between(t_range_sw_last, m_sw - s_sw, m_sw + s_sw,
                        color=color_sw, alpha=0.15, linewidth=0)

        # ---- Single-window baselines: start at t_win_start (last window only) ----
        t_range_single = np.arange(t_win_start, T_obs + T_FORECAST + 1)
        for label, X_key in [('invobs + hybrid (last 8)', 'invh_X0'),
                              ('baseline + obs-only (last 8)', 'base_X0')]:
            X = rc[X_key].to(device)
            m, s = full_rmse_curve(L96, X, d['X_true'],
                                   t_analysis=t_win_start, t_end=T_obs, T_forecast=T_FORECAST)
            color = COLORS_C[label]
            ax.plot(t_range_single, m, color=color, lw=1.8, label=label)
            ax.fill_between(t_range_single, m - s, m + s, color=color, alpha=0.15, linewidth=0)

        ax.axvline(T_obs, color='gray', ls='--', lw=1.3, zorder=2, label='window end')
        ax.set_xlim(0, T_obs + T_FORECAST)
        ax.set_title(f'T_obs={T_obs}, sigma={sigma}')
        ax.grid(True, alpha=0.3)
        if row_idx == 1:
            ax.set_xlabel('Time step')
        if col_idx == 0:
            ax.set_ylabel('Spatial RMSE')
        if row_idx == 0 and col_idx == 1:
            ax.legend(fontsize=8, loc='upper left')

fig.suptitle('Experiment C: RMSE — sliding window vs single window\n'
             '(SW-best dashed = per-cycle analysis RMSE from t=0; solid = last-cycle + forecast)', y=1.03)
plt.tight_layout()
plt.show()
```

>>> FIGURE EMBEDDED: figures/L96_SlidingWindow__cell34__out00.png

--- output ---
<Figure size 1300x900 with 4 Axes>

## [code cell 35]
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

>>> FIGURE EMBEDDED: figures/L96_SlidingWindow__cell35__out00.png

--- output ---
<Figure size 1200x450 with 2 Axes>

## [code cell 36]
```python
# ---- Plot E: Hovmöller — truth / prediction / error -------------------
# 5 rows per column:
#   0: truth
#   1: SW-best prediction      (RdBu_r, same ±vmax as truth)
#   2: SW-best error           (bwr, symmetric ±vmax_err shared with row 4)
#   3: invobs+hybrid last-8    (RdBu_r)
#   4: invobs+hybrid error     (bwr, same ±vmax_err as row 2)
# Error scale is the max |error| across both methods in each column so the
# two error rows are directly comparable within a panel.
SAMPLE_IDX = 0
n_total = WINDOW_T + T_FORECAST + 1

ROW_LABELS = [
    'truth',
    'SW-best',
    'SW-best\nerror',
    'invobs+hybrid\n(last 8)',
    'invobs+hybrid\nerror',
]
n_rows = len(ROW_LABELS)

fig, axes = plt.subplots(n_rows, 4, figsize=(17, 13))
col = 0

for T_obs in T_OBS_TOTALS_B:
    for sigma in NOISE_LEVELS:
        d = eval_data[sigma]
        r = results_C[(T_obs, sigma)]

        # Truth segment: last assimilation window + forecast.
        truth_seg = d['X_true'][
            SAMPLE_IDX, T_obs - WINDOW_T : T_obs + T_FORECAST + 1
        ].cpu().numpy()                                     # (n_total, 40)

        # Re-integrate from stored analysis states.
        X_sw = r['sw_last_analysis'].to(device)             # (N, grid)
        X_ih = r['invh_X0'].to(device)
        with torch.no_grad():
            traj_sw = L96.integrate(X_sw, n_total)[:, SAMPLE_IDX].cpu().numpy()
            traj_ih = L96.integrate(X_ih, n_total)[:, SAMPLE_IDX].cpu().numpy()

        err_sw = traj_sw - truth_seg                        # (n_total, 40)
        err_ih = traj_ih - truth_seg

        # Colour scales: state symmetric around truth range; error shared between
        # both methods in this column so magnitudes are directly comparable.
        vmax_s = float(np.abs(truth_seg).max())
        vmax_e = float(max(np.abs(err_sw).max(), np.abs(err_ih).max()))

        extent = [T_obs - WINDOW_T, T_obs + T_FORECAST, 0, L96.grid_size]

        panels = [
            (truth_seg, 'RdBu_r', -vmax_s, vmax_s),
            (traj_sw,   'RdBu_r', -vmax_s, vmax_s),
            (err_sw,    'bwr',    -vmax_e, vmax_e),
            (traj_ih,   'RdBu_r', -vmax_s, vmax_s),
            (err_ih,    'bwr',    -vmax_e, vmax_e),
        ]

        for row, (data, cmap, vmin, vmax) in enumerate(panels):
            ax = axes[row, col]
            ax.imshow(data.T, aspect='auto', origin='lower',
                      cmap=cmap, vmin=vmin, vmax=vmax, extent=extent)
            # Window-end marker.
            ax.axvline(T_obs, color='white', lw=2.5, zorder=3)
            ax.axvline(T_obs, color='black', lw=1.2, ls='--', zorder=4)
            if row == 0:
                ax.set_title(f'T_obs={T_obs}, σ={sigma}', fontsize=9)
                ax.text(T_obs + 0.4, L96.grid_size * 0.96, 'window end',
                        fontsize=6.5, color='black', ha='left', va='top',
                        bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7, lw=0))
            if col == 0:
                ax.set_ylabel(f'{ROW_LABELS[row]}\ngrid k', fontsize=8)
            if row == n_rows - 1:
                ax.set_xlabel('time step')

        col += 1

fig.suptitle(
    'Experiment E: Hovmöller (sample 0)\n'
    'State rows (RdBu_r): symmetric ±max(|truth|)  |  '
    'Error rows (bwr): pred − truth, blue=under, white=exact, red=over\n'
    'left of bar = last assimilation window  |  right of bar = forecast',
    y=1.02, fontsize=9,
)
plt.tight_layout()
plt.show()
```

>>> FIGURE EMBEDDED: figures/L96_SlidingWindow__cell36__out00.png

--- output ---
<Figure size 1700x1300 with 20 Axes>

## [code cell 37]
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
        # Hide spare axes
        for k in range(n_cycles, nrows * ncols):
            axes[k // ncols, k % ncols].axis('off')
        fig.suptitle(f'L-BFGS loss per cycle  |  T_obs={T_obs}, sigma={sigma}, stride={st}',
                      y=1.02, fontsize=10)
        plt.tight_layout()
        plt.show()
```

>>> FIGURE EMBEDDED: figures/L96_SlidingWindow__cell37__out00.png


>>> FIGURE EMBEDDED: figures/L96_SlidingWindow__cell37__out01.png


>>> FIGURE EMBEDDED: figures/L96_SlidingWindow__cell37__out02.png


>>> FIGURE EMBEDDED: figures/L96_SlidingWindow__cell37__out03.png

--- output ---
/tmp/ipykernel_3191/3055502080.py:15: MatplotlibDeprecationWarning: The get_cmap function was deprecated in Matplotlib 3.7 and will be removed in 3.11. Use ``matplotlib.colormaps[name]`` or ``matplotlib.colormaps.get_cmap()`` or ``pyplot.get_cmap()`` instead.
  cmap = cm.get_cmap('Blues')
<Figure size 900x240 with 3 Axes><Figure size 1500x480 with 10 Axes><Figure size 1500x480 with 10 Axes><Figure size 1500x480 with 10 Axes>
