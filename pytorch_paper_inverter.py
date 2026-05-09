"""Paper-faithful PyTorch port of paper_scripts/lorenz96_ml.py::ObservationInverterLorenz96.

Differences from the original PyTorch port (pytorch_invobs_lib.InverseObsLorenz96):
- Activation: SiLU (paper) vs GELU (port).
- Channel widths: [128, 64, 32, 16] across 4 blocks (paper) vs flat hidden=32 (port).
- Cubic periodic upsampling at factors (1, 2, 2, 1) (paper) vs one-shot bilinear at the start (port).
- BatchNorm after every conv (paper) vs none (port).
- No skip connections (paper) vs residual blocks (port).

This module is independent of pytorch_invobs_lib so tests can import it without
pulling in the heavier helpers.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CubicPeriodicUpsample1d(nn.Module):
    """Cubic periodic upsampling along the last axis (the spatial dim) of a 4D tensor.

    Mirrors paper_scripts/lorenz96_methods.py::interpolate_periodic_lorenz96, which
    uses jax.image.resize(method='cubic') after a one-element wrap-pad on each side
    and crops ``factor`` elements from each end of the resized output.

    Implementation note: torch.nn.functional.interpolate exposes 1D 'linear' and 2D
    'bicubic' but no 1D 'cubic' kernel. We therefore feed the 4D tensor to bicubic
    and arrange that the time dim is unchanged: with align_corners=False and
    H_in == H_out, output coordinates coincide with input grid points and the
    cubic kernel reduces to identity along that axis (kernel is 1 at offset 0 and
    0 at all other integer offsets, by construction).

    Input shape:  [N, C, T, S]
    Output shape: [N, C, T, factor * S]
    """

    def __init__(self, factor: int):
        super().__init__()
        self.factor = int(factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.factor
        if f == 1:
            # JAX equivalent at factor=1 reduces to identity on the unpadded interior:
            # pad to S+2, resize to 1*(S+2), crop arange(1, S+1) -> S elements.
            return x
        N, C, T, S = x.shape
        # Wrap one element of context onto each end of the spatial axis.
        # F.pad(circular) on a 4D tensor requires a 4-tuple covering the last
        # two dims; we leave the time dim untouched.
        x_pad = F.pad(x, (1, 1, 0, 0), mode='circular')    # [N, C, T, S+2]
        Sp = S + 2
        # Bicubic 2D with target H == T leaves the time axis untouched and
        # cubic-resamples the space axis to f * Sp.
        out = F.interpolate(
            x_pad,
            size=(T, f * Sp),
            mode='bicubic',
            align_corners=False,
        )
        # Crop f elements from each side of the space axis. Result length:
        #     f * Sp - 2 * f = f * S.
        return out[..., f:-f]


class _PeriodicSpaceConv2dPaper(nn.Module):
    """Mirror of paper_scripts/lorenz96_ml.py::PeriodicSpaceConv:
    space gets periodic padding, time gets zero padding.

    Duplicated from pytorch_invobs_lib.PeriodicSpaceConv2d so that this module
    has no cross-imports.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size=(3, 3)):
        super().__init__()
        self.kt, self.ks = kernel_size
        self.t_pad = (self.kt - 1) // 2
        self.s_pad = (self.ks - 1) // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.s_pad, self.s_pad, 0, 0), mode='circular')          # wrap space
        x = F.pad(x, (0, 0, self.t_pad, self.t_pad), mode='constant', value=0)  # zero time
        return self.conv(x)


class ObservationInverterLorenz96Paper(nn.Module):
    """PyTorch port of paper_scripts/lorenz96_ml.py::ObservationInverterLorenz96.

    Forward input shape:  [N, T, obs_grid]
    Forward output shape: [N, T, full_grid]
    """
    FEATURE_SIZES = (128, 64, 32, 16)
    RESIZES = (1, 2, 2, 1)
    KERNEL_SIZE = (3, 3)

    def __init__(self, obs_grid: int = 10, full_grid: int = 40):
        super().__init__()
        total_resize = 1
        for r in self.RESIZES:
            total_resize *= r
        assert obs_grid * total_resize == full_grid, (
            f'product(RESIZES)={total_resize} must map obs_grid={obs_grid} '
            f'to full_grid={full_grid}; got {obs_grid * total_resize}'
        )
        self.obs_grid = obs_grid
        self.full_grid = full_grid

        in_ch = 1
        self.upsamples = nn.ModuleList()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for fs, rs in zip(self.FEATURE_SIZES, self.RESIZES):
            self.upsamples.append(CubicPeriodicUpsample1d(rs))
            self.convs.append(_PeriodicSpaceConv2dPaper(in_ch, fs, self.KERNEL_SIZE))
            self.bns.append(nn.BatchNorm2d(fs))
            in_ch = fs
        self.out_conv = _PeriodicSpaceConv2dPaper(in_ch, 1, self.KERNEL_SIZE)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        # y: [N, T, obs_grid] -> [N, 1, T, obs_grid] (NCHW with H=time, W=space).
        x = y.unsqueeze(1)
        for up, conv, bn in zip(self.upsamples, self.convs, self.bns):
            x = up(x)
            x = conv(x)
            x = bn(x)
            x = F.silu(x)
        x = self.out_conv(x)
        # Drop channel: [N, 1, T, full_grid] -> [N, T, full_grid].
        return x.squeeze(1)
