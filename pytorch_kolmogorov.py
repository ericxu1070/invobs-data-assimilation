"""
PyTorch pseudo-spectral solver for 2D Kolmogorov flow.
Mirrors paper_scripts/dynamical_system.py::KolmogorovFlow but in PyTorch.

State variable: vorticity field omega, shape (batch, Nx, Ny).
Domain: [0, 2*pi]^2 with periodic boundary conditions.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class KolmogorovFlow:
    """
    2D incompressible Navier-Stokes with Kolmogorov forcing.

    Paper parameters:
      Nx = Ny = 64
      nu = 1e-2
      alpha = 0.1
      k_forcing = 4
      outer_dt ~ 0.18  (25 internal RK4 steps per outer step)
      observe_every = 16  -> 4x4 observation grid
      assimilation window T = 10
    """

    def __init__(self, Nx=64, Ny=64, nu=1e-2, alpha=0.1,
                 k_forcing=4, outer_dt=0.18, n_inner=25,
                 observe_every=16, device=None):
        self.Nx = Nx
        self.Ny = Ny
        self.nu = nu
        self.alpha = alpha
        self.k_forcing = k_forcing
        self.outer_dt = outer_dt
        self.n_inner = n_inner
        self.dt_inner = outer_dt / n_inner
        self.observe_every = observe_every
        self.device = device or torch.device('cpu')

        # Wavenumber grids — shape (Nx, Ny)
        kx = torch.fft.fftfreq(Nx, d=1.0 / Nx).to(self.device)
        ky = torch.fft.fftfreq(Ny, d=1.0 / Ny).to(self.device)
        self.KX, self.KY = torch.meshgrid(kx, ky, indexing='ij')
        self.K2 = self.KX ** 2 + self.KY ** 2
        # Avoid division by zero at k=0 (mean mode stays zero)
        self.K2_safe = self.K2.clone()
        self.K2_safe[0, 0] = 1.0

        # Forcing: curl of F = sin(k_forcing * y) x-hat
        # => vorticity forcing = -d(sin(k*y))/dy = -k*cos(k*y)
        # (matches paper: linear damping + Kolmogorov body force)
        y = torch.linspace(0, 2 * math.pi, Ny + 1, device=self.device)[:-1]
        x = torch.linspace(0, 2 * math.pi, Nx + 1, device=self.device)[:-1]
        _, Y = torch.meshgrid(x, y, indexing='ij')
        # Vorticity source from Kolmogorov force sin(k*x)*x_hat:
        # omega forcing = dFy/dx - dFx/dy = 0 - d(sin(k*y))/dy = -k*cos(k*y)
        self.forcing = -k_forcing * torch.cos(k_forcing * Y)  # (Nx, Ny)

    # ------------------------------------------------------------------
    # Core physics
    # ------------------------------------------------------------------

    def _rhs(self, omega):
        """Compute d(omega)/dt for a batch of vorticity fields.

        Args:
            omega: (batch, Nx, Ny) real tensor

        Returns:
            d_omega_dt: (batch, Nx, Ny) real tensor
        """
        omega_hat = torch.fft.rfft2(omega)  # (B, Nx, Ny//2+1) complex

        # Stream function: psi_hat = -omega_hat / K2
        K2_r = self.K2_safe[..., :self.Ny // 2 + 1].unsqueeze(0)
        KX_r = self.KX[..., :self.Ny // 2 + 1].unsqueeze(0)
        KY_r = self.KY[..., :self.Ny // 2 + 1].unsqueeze(0)

        psi_hat = -omega_hat / K2_r

        # Velocity field: u = dpsi/dy, v = -dpsi/dx
        u_hat = 1j * KY_r * psi_hat
        v_hat = -1j * KX_r * psi_hat

        # Vorticity gradients
        domega_dx_hat = 1j * KX_r * omega_hat
        domega_dy_hat = 1j * KY_r * omega_hat

        # Back to physical space
        u = torch.fft.irfft2(u_hat, s=(self.Nx, self.Ny))
        v = torch.fft.irfft2(v_hat, s=(self.Nx, self.Ny))
        domega_dx = torch.fft.irfft2(domega_dx_hat, s=(self.Nx, self.Ny))
        domega_dy = torch.fft.irfft2(domega_dy_hat, s=(self.Nx, self.Ny))

        # Diffusion term: nu * Laplacian(omega)
        K2_r_full = self.K2[..., :self.Ny // 2 + 1].unsqueeze(0)
        lap_omega = torch.fft.irfft2(
            -K2_r_full * omega_hat, s=(self.Nx, self.Ny)
        )

        # Full RHS: nonlinear advection + diffusion + forcing - damping
        rhs = (-(u * domega_dx + v * domega_dy)
               + self.nu * lap_omega
               + self.forcing.unsqueeze(0)
               - self.alpha * omega)
        return rhs

    def _rk4_step(self, omega, dt):
        k1 = self._rhs(omega)
        k2 = self._rhs(omega + 0.5 * dt * k1)
        k3 = self._rhs(omega + 0.5 * dt * k2)
        k4 = self._rhs(omega + dt * k3)
        return omega + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def step(self, omega):
        """One outer step = n_inner internal RK4 steps."""
        for _ in range(self.n_inner):
            omega = self._rk4_step(omega, self.dt_inner)
        return omega

    # ------------------------------------------------------------------
    # Trajectory utilities
    # ------------------------------------------------------------------

    def integrate(self, omega0, n_steps, start_with_input=True):
        """Integrate forward.

        Args:
            omega0: (batch, Nx, Ny)
            n_steps: number of outer steps in the returned trajectory
            start_with_input: if True, trajectory[0] = omega0

        Returns:
            (n_steps, batch, Nx, Ny)
        """
        traj = [omega0] if start_with_input else []
        omega = omega0
        n_advance = n_steps - 1 if start_with_input else n_steps
        for _ in range(n_advance):
            omega = self.step(omega)
            traj.append(omega)
        return torch.stack(traj, dim=0)

    def warmup(self, omega0, n_outer_steps):
        """Spin up from a cold start to the statistically stationary regime.

        Args:
            omega0: (batch, Nx, Ny)
            n_outer_steps: number of outer steps to discard

        Returns:
            (batch, Nx, Ny) warmed-up state
        """
        omega = omega0
        for _ in range(n_outer_steps):
            omega = self.step(omega)
        return omega

    def observe(self, omega):
        """Subsample every observe_every grid points in x and y.

        Args:
            omega: (..., Nx, Ny)

        Returns:
            (..., Nx//observe_every, Ny//observe_every)
        """
        return omega[..., ::self.observe_every, ::self.observe_every]

    # ------------------------------------------------------------------
    # Initial condition helper
    # ------------------------------------------------------------------

    def random_init(self, batch_size, peak_wavenumber=4, seed=None):
        """Spectrally filtered random vorticity field.

        Mirrors the paper: random field filtered at peak wavenumber k0,
        then integrated to stationary regime.

        Args:
            batch_size: number of independent initial conditions
            peak_wavenumber: spectral filter peak (paper uses 4)
            seed: optional int for reproducibility

        Returns:
            (batch_size, Nx, Ny) unwarmed vorticity
        """
        g = None
        if seed is not None:
            g = torch.Generator(device=self.device).manual_seed(seed)

        noise = torch.randn(batch_size, self.Nx, self.Ny,
                            device=self.device, generator=g)
        noise_hat = torch.fft.rfft2(noise)

        # Gaussian spectral filter centered at peak_wavenumber
        K2_r = self.K2[..., :self.Ny // 2 + 1]
        K_mag = K2_r.sqrt()
        filt = torch.exp(-0.5 * ((K_mag - peak_wavenumber) / peak_wavenumber) ** 2)
        noise_hat = noise_hat * filt.unsqueeze(0)

        omega0 = torch.fft.irfft2(noise_hat, s=(self.Nx, self.Ny))
        # Normalise to unit RMS
        rms = omega0.std(dim=(-2, -1), keepdim=True).clamp(min=1e-8)
        return omega0 / rms


# -----------------------------------------------------------------------
# Inverse observation operator for Kolmogorov flow
# -----------------------------------------------------------------------

class PeriodicSpaceConv3d(nn.Module):
    """Conv3d over (time, x, y). Space dims get periodic padding, time gets zero padding."""

    def __init__(self, in_ch, out_ch, k_t=3, k_x=3, k_y=3):
        super().__init__()
        self.k_t = k_t
        self.k_x = k_x
        self.k_y = k_y
        self.conv = nn.Conv3d(in_ch, out_ch,
                              kernel_size=(k_t, k_x, k_y),
                              padding=0)

    def forward(self, x):
        # x: (B, C, T, X, Y)
        pt = (self.k_t - 1) // 2
        px = (self.k_x - 1) // 2
        py = (self.k_y - 1) // 2
        # Periodic in both spatial dims
        x = F.pad(x, (py, py, px, px, 0, 0), mode='circular')
        # Zero-pad time
        x = F.pad(x, (0, 0, 0, 0, pt, pt), mode='constant')
        return self.conv(x)


class ObservationInverterKolmogorov(nn.Module):
    """
    Fully-convolutional inverse observation operator for Kolmogorov flow.

    Mirrors paper Table 2:
      Input:  (T=10, X_obs=4, Y_obs=4, C=2)  [u, v velocity components]
      Output: (T=10, X=64, Y=64, C=2)

    Architecture: 5 upsample blocks, each doubling spatial resolution,
    using Conv3d + BatchNorm + SiLU. Filter size (3,3,3) throughout.
    """

    def __init__(self, T=10, obs_grid=4, full_grid=64, in_channels=1, out_channels=1):
        """
        Args:
            in_channels: 1 for vorticity-only input, 2 for (u,v) velocity
            out_channels: 1 for vorticity output, 2 for (u,v)
        """
        super().__init__()
        self.T = T
        self.obs_grid = obs_grid
        self.full_grid = full_grid
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Channel widths matching paper Table 2: 64, 32, 16, 8, 4
        channels = [64, 32, 16, 8, 4]

        self.input_conv = nn.Sequential(
            PeriodicSpaceConv3d(in_channels, channels[0]),
            nn.BatchNorm3d(channels[0]),
            nn.SiLU(),
        )

        self.upsample_blocks = nn.ModuleList()
        for i in range(len(channels) - 1):
            block = nn.Sequential(
                PeriodicSpaceConv3d(channels[i], channels[i + 1]),
                nn.BatchNorm3d(channels[i + 1]),
                nn.SiLU(),
            )
            self.upsample_blocks.append(block)

        self.output_conv = PeriodicSpaceConv3d(channels[-1], out_channels)

    def forward(self, y):
        """
        Args:
            y: (B, T, X_obs, Y_obs) or (B, T, X_obs, Y_obs, C)

        Returns:
            x_pred: (B, T, X_full, Y_full) or (B, T, X_full, Y_full, C)
        """
        if y.dim() == 4:
            # Add channel dim: (B, T, Xo, Yo) -> (B, 1, T, Xo, Yo)
            y = y.unsqueeze(1)
        else:
            # (B, T, Xo, Yo, C) -> (B, C, T, Xo, Yo)
            y = y.permute(0, 4, 1, 2, 3)

        B, C, T, Xo, Yo = y.shape

        # Upsample spatially to full grid using bicubic interpolation
        # Reshape to (B*C*T, 1, Xo, Yo) for 2D interpolation
        y_flat = y.reshape(B * C * T, 1, Xo, Yo)
        # bicubic interpolation not supported on MPS — do it on CPU then move back
        orig_device = y_flat.device
        y_up = F.interpolate(y_flat.cpu(), size=(self.full_grid, self.full_grid),
                            mode='bicubic', align_corners=False).to(orig_device)
        y_up = y_up.reshape(B, C, T, self.full_grid, self.full_grid)

        x = self.input_conv(y_up)

        for block in self.upsample_blocks:
            x = block(x)

        x = self.output_conv(x)  # (B, out_ch, T, X, Y)

        if self.out_channels == 1:
            # (B, 1, T, X, Y) -> (B, T, X, Y)
            return x.squeeze(1)
        else:
            # (B, C, T, X, Y) -> (B, T, X, Y, C)
            return x.permute(0, 2, 3, 4, 1)


# -----------------------------------------------------------------------
# Data generation
# -----------------------------------------------------------------------

def generate_kolmogorov_data(dyn_sys, n_samples, n_time_steps,
                              n_warmup, obs_noise_std=0.0, seed=0):
    """Generate a batch of (state, observation) trajectory pairs.

    Args:
        dyn_sys: KolmogorovFlow instance
        n_samples: number of independent trajectories
        n_time_steps: length of each trajectory (outer steps)
        n_warmup: outer steps to discard during warm-up
        obs_noise_std: std of Gaussian noise added to observations
        seed: random seed

    Returns:
        omega0: (N, Nx, Ny) initial vorticity
        traj:   (N, T, Nx, Ny) full trajectory
        Y:      (N, T, X_obs, Y_obs) noisy observations
        Y_clean:(N, T, X_obs, Y_obs) clean observations
    """
    omega0_cold = dyn_sys.random_init(n_samples, seed=seed)
    omega0 = dyn_sys.warmup(omega0_cold, n_warmup)

    # integrate returns (T, N, Nx, Ny); permute to (N, T, Nx, Ny)
    traj = dyn_sys.integrate(omega0, n_time_steps).permute(1, 0, 2, 3)

    Y_clean = dyn_sys.observe(traj)  # (N, T, X_obs, Y_obs)

    g = torch.Generator(device=dyn_sys.device).manual_seed(seed + 1)
    noise = torch.empty_like(Y_clean).normal_(generator=g) * obs_noise_std
    Y = Y_clean + noise

    return omega0, traj, Y, Y_clean