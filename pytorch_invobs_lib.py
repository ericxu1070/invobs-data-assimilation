"""Shared library for the v2 PyTorch InvObs DA notebooks.

Helpers lifted verbatim from the original notebooks:
  - PyTorch_InvObs_DA.ipynb
      Lorenz96, rk4_step, generate_data, estimate_correlation,
      PeriodicSpaceConv2d, InverseObsLorenz96, baseline_init_l96,
      decorrelate, correlate
  - PyTorch_InvObs_DA_WindowSweep_Corrected.ipynb
      lbfgs_minimize, make_da_loss, invobs_init_l96

Cache helpers (save_cache, load_cache) take ``cache_dir`` explicitly so this
module has no hidden globals. Functions that need a default device read the
module-level ``device`` (CPU by default; notebooks override it after detecting
the runtime).

Usage:
    import pytorch_invobs_lib as pil
    pil.device = device                # set once after notebook setup
"""
import os
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Module-level device. Notebooks override this via:
#     import pytorch_invobs_lib as pil
#     pil.device = ns.setup_device()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ---------------------------------------------------------------------------
# Cache helpers (refactored: explicit cache_dir, optional force flag).
# ---------------------------------------------------------------------------


def save_cache(obj, name, cache_dir):
    """Write ``obj`` to ``cache_dir/name`` via torch.save."""
    p = os.path.join(cache_dir, name)
    torch.save(obj, p)
    print(f'  [cache] wrote {name}')


def load_cache(name, cache_dir, force=False):
    """Load ``cache_dir/name`` if present (and not ``force``); else return None."""
    p = os.path.join(cache_dir, name)
    if force or not os.path.exists(p):
        return None
    print(f'  [cache] loaded {name}')
    return torch.load(p, map_location=device, weights_only=False)


# ---------------------------------------------------------------------------
# Lorenz96 (lifted verbatim from PyTorch_InvObs_DA.ipynb).
# ---------------------------------------------------------------------------


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

    def integrate_adaptive(self, x0, n_steps, rtol=1e-7, atol=1e-9, method='dopri5'):
        """Integrate via torchdiffeq's adaptive ``odeint``, returning samples on the
        same outer-step grid (`outer_dt = dt * n_inner`) as :meth:`integrate`.

        Mirrors the JAX paper's ``jax.experimental.ode.odeint(...)`` with
        ``ts = arange(n_steps) * outer_dt`` — variable internal step, fixed
        output spacing. Use this for fair comparison against the paper's
        integrator; ``integrate`` (fixed RK4) remains untouched and is what
        the DA pipeline normally uses.

        Args:
            x0: initial state(s), shape (..., grid_size).
            n_steps: number of outer time samples to produce (including t=0).
            rtol, atol: solver tolerances.
            method: torchdiffeq solver. 'dopri5' = 5(4) Dormand-Prince (default).

        Returns:
            Trajectory of shape (n_steps, ..., grid_size), matching
            :meth:`integrate(..., start_with_input=True)`.
        """
        import torchdiffeq                    # imported lazily so the lib has no hard dep
        ts = torch.arange(n_steps, device=x0.device, dtype=x0.dtype) * self.outer_dt
        # torchdiffeq expects f(t, y) -> dy/dt; our rhs is autonomous (no t).
        return torchdiffeq.odeint(lambda t, y: self.rhs(y), x0, ts,
                                  rtol=rtol, atol=atol, method=method)


# ---------------------------------------------------------------------------
# Data generation + spatial correlation.
# ---------------------------------------------------------------------------


def generate_data(dyn_sys, n_samples, n_time_steps, n_warmup, obs_noise_std=0.0, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    X0_cold = torch.randn(n_samples, dyn_sys.grid_size, device=device, generator=g) * 0.5
    # Batched warmup: all N samples spun up simultaneously as (N, grid) — no Python loop.
    X0 = dyn_sys.warmup(X0_cold, n_warmup)                          # (N, grid)
    # Batched integrate: returns (T, N, grid); permute to (N, T, grid).
    X = dyn_sys.integrate(X0, n_time_steps).permute(1, 0, 2)        # (N, T, grid)
    Y_clean = dyn_sys.observe(X)                                    # (N, T, obs_grid)
    noise = torch.empty_like(Y_clean).normal_(generator=g) * obs_noise_std
    Y = Y_clean + noise
    return X0, X, Y, Y_clean


def estimate_correlation(dyn_sys, n_samples=2000, n_warmup=1000, seed=1):
    g = torch.Generator(device=device).manual_seed(seed)
    X0 = torch.randn(n_samples, dyn_sys.grid_size, device=device, generator=g) * 0.5
    # Batched warmup.
    X = dyn_sys.warmup(X0, n_warmup)                                # (N, grid)
    X = X - X.mean(dim=0, keepdim=True)
    C = (X.T @ X) / (X.shape[0] - 1)
    eigvals, eigvecs = torch.linalg.eigh(C)
    eigvals = torch.clamp(eigvals, min=1e-8)
    C_sqrt     = eigvecs @ torch.diag(eigvals.sqrt())  @ eigvecs.T
    C_inv_sqrt = eigvecs @ torch.diag(eigvals.rsqrt()) @ eigvecs.T
    C_inv      = eigvecs @ torch.diag(1.0 / eigvals)   @ eigvecs.T
    return dict(C=C, C_sqrt=C_sqrt, C_inv_sqrt=C_inv_sqrt, C_inv=C_inv)


# ---------------------------------------------------------------------------
# Original PyTorch port of the inverse-observation CNN.
# (The paper-faithful port lives in pytorch_paper_inverter.py.)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# DA initializers + 4D-Var loss + LBFGS driver.
# ---------------------------------------------------------------------------


def decorrelate(x, C_inv_sqrt):
    return x @ C_inv_sqrt


def correlate(z, C_sqrt):
    return z @ C_sqrt


def baseline_init_l96(dyn_sys, Y):
    """Repeat-interleave baseline init: copies the t=0 observation to each unobserved
    grid point in its block. Cheap, deterministic, ignores the dataset mean.
    """
    if Y.ndim == 3:
        return Y[:, 0].repeat_interleave(dyn_sys.observe_every, dim=-1)
    return Y[0].repeat_interleave(dyn_sys.observe_every)


def estimate_climatological_mean(dyn_sys, n_samples=2000, n_warmup=1000, seed=2):
    """Per-grid-point mean over a long warmed-up ensemble.

    Mirrors the spirit of paper_scripts/lorenz96_methods.py::average_da_init_lorenz96,
    where the dataset mean is precomputed and then used as the unobserved-grid-point
    fill value. We compute it from a separate warmed-up ensemble (parameterized by
    its own seed) so it generalizes across different evaluation sets.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    X0 = torch.randn(n_samples, dyn_sys.grid_size, device=device, generator=g) * 0.5
    X = dyn_sys.warmup(X0, n_warmup)                               # (N, grid)
    return X.mean(dim=0)                                           # (grid,)


def baseline_init_l96_paper(dyn_sys, Y, X0_mean):
    """Paper-style "average" baseline init from paper_scripts/lorenz96_methods.py::
    average_da_init_lorenz96.

    Recipe: tile the climatological mean across the batch, then overwrite the
    observed grid points with the t=0 observations. This is a fair prior because
    the unobserved positions are filled with the dataset's expectation rather
    than a copy of an adjacent observation.

    Args:
        dyn_sys: Lorenz96 instance (provides ``observe_every``).
        Y: observations of shape (N, T, obs_grid) or (T, obs_grid).
        X0_mean: per-grid-point climatological mean, shape (grid,).

    Returns:
        Initial conditions of shape (N, grid) or (grid,) matching Y.
    """
    if Y.ndim == 3:
        N = Y.shape[0]
        X0_init = X0_mean.unsqueeze(0).expand(N, -1).clone()
        X0_init[:, ::dyn_sys.observe_every] = Y[:, 0]
        return X0_init
    X0_init = X0_mean.clone()
    X0_init[::dyn_sys.observe_every] = Y[0]
    return X0_init


def invobs_init_l96(inverter, Y):
    with torch.no_grad():
        return inverter(Y).detach()[:, 0]


def make_da_loss(Y, dyn_sys, C_sqrt, T, mode, inverter=None):
    """Closure factory for the per-mode DA loss in decorrelated z-coordinates.

    mode='obs'     : ||H(M(x_0)) - y||^2 (mean over batch / time / obs).
    mode='physics' : ||M(x_0) - H^{-1}_theta(y)||^2 with the inverter's output frozen.
    """
    if mode == 'physics':
        assert inverter is not None
        with torch.no_grad():
            target = inverter(Y).detach().transpose(0, 1)
    elif mode == 'obs':
        target = Y.transpose(0, 1)
    else:
        raise ValueError(mode)

    def loss_fn(Z0):
        X0 = correlate(Z0, C_sqrt)
        traj = dyn_sys.integrate(X0, T)
        if mode == 'obs':
            pred = dyn_sys.observe(traj)
        else:
            pred = traj
        return ((pred - target) ** 2).mean()

    return loss_fn


def lbfgs_minimize(loss_fn, z0_init, max_iter=200, max_eval=None, history_size=20, lr=1.0):
    """L-BFGS driver with strong-Wolfe line search and per-step diagnostics.

    Captures loss / gradient norm / gradient max per closure call. If a closure
    sees a non-finite loss or gradient it raises; the outer try/except logs the
    failure into ``diag['status']`` rather than propagating.
    """
    z = z0_init.clone().detach().requires_grad_(True)
    kwargs = dict(
        max_iter=max_iter,
        history_size=history_size,
        tolerance_grad=1e-12,
        tolerance_change=1e-12,
        line_search_fn='strong_wolfe',
        lr=lr,
    )
    if max_eval is not None:
        kwargs['max_eval'] = max_eval
    opt = torch.optim.LBFGS([z], **kwargs)
    diag = {
        'loss': [],
        'grad_norm': [],
        'grad_abs_max': [],
        'status': 'ok',
        'exception': None,
        'optimizer_n_iter': None,
        'optimizer_func_evals': None,
    }

    def closure():
        opt.zero_grad()
        loss = loss_fn(z)
        if not torch.isfinite(loss):
            raise FloatingPointError(f'nonfinite loss: {loss.item()}')
        loss.backward()
        grad = z.grad.detach()
        grad_norm = torch.linalg.vector_norm(grad).item()
        grad_abs_max = grad.abs().max().item()
        diag['loss'].append(loss.item())
        diag['grad_norm'].append(grad_norm)
        diag['grad_abs_max'].append(grad_abs_max)
        if not np.isfinite(grad_norm) or not np.isfinite(grad_abs_max):
            raise FloatingPointError(f'nonfinite gradient: norm={grad_norm}, max={grad_abs_max}')
        return loss

    try:
        opt.step(closure)
    except Exception as exc:
        diag['status'] = 'failed'
        diag['exception'] = repr(exc)
    state = opt.state.get(z, {})
    diag['optimizer_n_iter'] = int(state.get('n_iter', -1)) if state else None
    diag['optimizer_func_evals'] = int(state.get('func_evals', -1)) if state else None
    return z.detach(), diag


# ---------------------------------------------------------------------------
# Full 4D-Var (lifted from PyTorch_InvObs_DA.ipynb, adapted to the new
# lbfgs_minimize diag-dict return value). The cost functions use .sum() over
# the batch; this makes J_b, J_o decomposable per sample and L-BFGS over the
# stacked Z is equivalent to N independent 4D-Var problems.
# ---------------------------------------------------------------------------


def var4d_cost_obs(z0, y_T, dyn_sys, C_sqrt, T, z_b, R_inv_diag, sigma_b):
    """Standard 4D-Var: J = J_b + J_o.

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
    """Physics-space companion cost: J = J_b + (1/(2 sigma_p^2)) ||M(x_0) - H^{-1}_theta(y)||^2.

    target_traj: precomputed (T, N, grid) = inverter(Y).detach().transpose(0,1).
    """
    x0 = correlate(z0, C_sqrt)
    traj = dyn_sys.integrate(x0, T)            # (T, N, grid)
    J_phys = 0.5 * ((traj - target_traj).pow(2)).sum() / (sigma_p ** 2)
    J_b = 0.5 * ((z0 - z_b).pow(2)).sum() / (sigma_b ** 2)
    return J_b + J_phys


def run_4dvar_l96(dyn_sys, inverter, corr, X0_init, Y, T,
                  sigma_b=1.0, sigma_obs=0.5, sigma_p=0.5,
                  mode='obs', physics_steps=200, obs_steps=300,
                  max_eval=None):
    """Full 4D-Var driver. ``X0_init`` acts as both starting point and background x_b.

    mode='obs'    : minimize J_b + J_o for ``obs_steps`` L-BFGS iterations.
    mode='hybrid' : first minimize J_b + J_phys for ``physics_steps`` iters,
                    then continue with J_b + J_o.

    Returns (X0_opt: (N, grid), stage_diags: list of diag dicts).
    """
    C_sqrt, C_inv_sqrt = corr['C_sqrt'], corr['C_inv_sqrt']
    z_b = decorrelate(X0_init, C_inv_sqrt)
    R_inv_diag = torch.full(
        (dyn_sys.grid_size // dyn_sys.observe_every,),
        1.0 / (sigma_obs ** 2),
        device=z_b.device,
    )
    Y_T = Y.transpose(0, 1)                                 # (T, N, obs_grid)
    z = z_b.clone()
    stage_diags = []

    if mode == 'hybrid':
        target_traj = inverter(Y).detach().transpose(0, 1)  # (T, N, grid)
        loss_phys = partial(var4d_cost_phys, target_traj=target_traj,
                            dyn_sys=dyn_sys, C_sqrt=C_sqrt, T=T,
                            z_b=z_b, sigma_b=sigma_b, sigma_p=sigma_p)
        z, diag_p = lbfgs_minimize(loss_phys, z, max_iter=physics_steps, max_eval=max_eval)
        diag_p['stage'] = 'physics'
        stage_diags.append(diag_p)

    loss_obs = partial(var4d_cost_obs, y_T=Y_T, dyn_sys=dyn_sys, C_sqrt=C_sqrt, T=T,
                       z_b=z_b, R_inv_diag=R_inv_diag, sigma_b=sigma_b)
    z, diag_o = lbfgs_minimize(loss_obs, z, max_iter=obs_steps, max_eval=max_eval)
    diag_o['stage'] = 'obs'
    stage_diags.append(diag_o)

    X0_opt = correlate(z, C_sqrt)
    return X0_opt, stage_diags
