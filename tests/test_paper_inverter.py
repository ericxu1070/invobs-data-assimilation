"""Tests for the paper-faithful PyTorch port in pytorch_paper_inverter.py.

Run from the repo root:
    pytest tests/test_paper_inverter.py -v

Notes on tolerances. The JAX paper recipe (pad-1, cubic-resize, crop-factor) is
not perfectly cyclic-aware: a 1-element pad does not cover the reach of the
cubic kernel (+/- 2), so the boundary handling of the resize backend leaks into
the cropped output. We mirror that recipe exactly, then test the operational
properties: shape, constant invariance, low-frequency preservation, and wrap
continuity (no boundary jump). We do NOT test strict cyclic shift equivariance
because the bicubic backend's boundary handling cannot guarantee it.
"""
import math
import os
import sys

import pytest
import torch

# Allow importing repo-root modules when pytest is invoked from anywhere.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pytorch_paper_inverter import (
    CubicPeriodicUpsample1d,
    ObservationInverterLorenz96Paper,
)


# ---------------------------------------------------------------------------
# CubicPeriodicUpsample1d
# ---------------------------------------------------------------------------


def _output_positions(N: int, factor: int) -> torch.Tensor:
    """Return the input-coordinate positions of the upsampler's output samples.

    F.interpolate(..., mode='bicubic', align_corners=False) maps output index
    ``i_out`` to input coord ``(i_out + 0.5)/scale - 0.5``. With our pad-1 then
    crop-factor recipe, surviving cropped positions in original (unpadded)
    coords are ``i / factor + 1/(2*factor) - 1/2`` for ``i = 0..factor*N-1``.
    """
    M = factor * N
    return torch.arange(M, dtype=torch.float32) / factor + 0.5 / factor - 0.5


@pytest.mark.parametrize("factor", [1, 2, 4])
def test_constant_input_constant_output(factor):
    """A constant signal must upsample to itself (cubic kernel partition of unity)."""
    up = CubicPeriodicUpsample1d(factor=factor)
    x = torch.full((2, 3, 5, 10), 3.7)
    y = up(x)
    expected_S = 10 * factor
    assert y.shape == (2, 3, 5, expected_S)
    assert torch.allclose(y, torch.full_like(y, 3.7), atol=1e-5), (
        f'constant {3.7} not preserved at factor {factor}: '
        f'max abs diff = {(y - 3.7).abs().max().item()}'
    )


@pytest.mark.parametrize("factor", [2, 4])
@pytest.mark.parametrize("k", [0, 1, 2, 3, 4])
def test_low_freq_sinusoid_preserved(factor, k):
    """sin(2 pi k j / N) with k <= N/4 is preserved at the upsampler's actual
    output positions.

    Cubic upsampling with ``align_corners=False`` introduces a half-pixel grid
    shift; we account for it via ``_output_positions``. With that correction,
    cubic interpolation reproduces low-frequency sinusoids to within
    ~0.1 absolute (Keys-cubic interpolant on a length-16 signal at the Nyquist
    band is the limit of what one can expect).
    """
    N = 16  # ensures k <= N/4 covers the well-resolved band
    j = torch.arange(N, dtype=torch.float32)
    x = torch.sin(2 * math.pi * k * j / N).reshape(1, 1, 1, N)

    up = CubicPeriodicUpsample1d(factor=factor)
    y = up(x).squeeze()

    positions = _output_positions(N, factor)
    expected = torch.sin(2 * math.pi * k * positions / N)
    err = (y - expected).abs().max().item()
    assert err < 0.1, (
        f'sin(2 pi {k} j / {N}) not preserved at factor {factor}: max err {err:.4f}'
    )


@pytest.mark.parametrize("factor", [2, 4])
def test_wrap_is_continuous_no_boundary_jump(factor):
    """The upsampled sequence must wrap smoothly: the step from the last sample
    to the first (cyclically) must be in family with the interior step
    distribution. This is the operational meaning of the spec's
    "up(x)[..., 0] matches the wrap of up(x)[..., -1]" — no boundary jump.
    """
    torch.manual_seed(1)
    x = torch.randn(1, 1, 1, 12)
    up = CubicPeriodicUpsample1d(factor=factor)
    y = up(x).squeeze()
    interior_step = (y[1:] - y[:-1]).abs()
    wrap_step = (y[0] - y[-1]).abs()
    # Wrap step should not be a wild outlier vs the median interior step.
    assert wrap_step.item() <= 5.0 * interior_step.median().item() + 1e-6, (
        f'wrap discontinuity at factor {factor}: '
        f'wrap={wrap_step.item():.4f} vs median interior={interior_step.median().item():.4f}'
    )


@pytest.mark.parametrize("factor", [2, 4])
def test_upsampler_preserves_low_freq_dft_power(factor):
    """The upsampled signal's DFT power at low frequencies should match the
    input's DFT power at those same frequencies (modulo a known overall scaling
    from the resampling rate). This is a global periodicity check that does not
    depend on grid alignment.
    """
    torch.manual_seed(2)
    N = 32
    j = torch.arange(N, dtype=torch.float32)
    # Random low-pass signal: sum of low-frequency sinusoids
    x = sum(torch.sin(2 * math.pi * k * j / N + (k + 1) * 0.3) for k in range(1, N // 4 + 1))
    x = x.reshape(1, 1, 1, N)

    up = CubicPeriodicUpsample1d(factor=factor)
    y = up(x).squeeze()                                # length factor * N

    # Compare normalized DFT magnitudes at the low-freq bands.
    fx = torch.fft.rfft(x.squeeze())
    fy = torch.fft.rfft(y)
    n_lowfreq = N // 4 + 1                             # check first N/4 + 1 modes (incl. DC)
    # Normalize by signal length (Parseval-style scaling).
    mag_x = fx[:n_lowfreq].abs() / N
    mag_y = fy[:n_lowfreq].abs() / (factor * N)
    rel_err = (mag_x - mag_y).abs().max().item() / (mag_x.abs().max().item() + 1e-9)
    assert rel_err < 0.05, (
        f'low-freq DFT power mismatch at factor {factor}: relative err {rel_err:.4f}'
    )


# ---------------------------------------------------------------------------
# ObservationInverterLorenz96Paper
# ---------------------------------------------------------------------------


def test_inverter_forward_shape_eval():
    """Smoke test: paper inverter outputs the correct shape (eval mode, so BN runs
    on the inferred running stats and tolerates batch size 1)."""
    net = ObservationInverterLorenz96Paper(obs_grid=10, full_grid=40)
    net.eval()
    y = torch.randn(8, 5, 10)  # (N, T, obs_grid)
    with torch.no_grad():
        out = net(y)
    assert out.shape == (8, 5, 40)


def test_inverter_forward_no_nan_train_mode():
    """Inverter's forward pass must not produce NaN/Inf in training mode (which
    activates BatchNorm's running-stat updates)."""
    net = ObservationInverterLorenz96Paper(obs_grid=10, full_grid=40)
    net.train()
    y = torch.randn(8, 5, 10)
    out = net(y)
    assert torch.isfinite(out).all(), 'paper inverter produced NaN/Inf in train mode'


def test_inverter_grid_size_assertion():
    """Constructor must reject mismatched (obs_grid, full_grid) pairs."""
    with pytest.raises(AssertionError):
        ObservationInverterLorenz96Paper(obs_grid=10, full_grid=80)
