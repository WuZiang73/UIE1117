import torch
import torch.nn as nn

from psf_recovery.kernels import kernel_bank
from psf_recovery.operators import _conv3


class WavelengthPreFilter(nn.Module):
    """Global per-channel wavelength PSF prefilter, identity at init."""

    def __init__(self, channels=3, radius=8):
        super().__init__()
        self.register_buffer("kernels", kernel_bank((0.0, 0.6, 1.2, 2.4), radius=radius))
        self.coeff = nn.Parameter(torch.zeros(channels, 4))
        with torch.no_grad():
            self.coeff[:, 0] = 1.0
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        blurred = _conv3(x, self.kernels)  # B,C,4,H,W
        y = (blurred * self.coeff[None, :, :, None, None]).sum(2) + self.bias[None, :, None, None]
        return (x + 0.25 * (y - x)).clamp(0, 1)


class ISPProxy(nn.Module):
    """Per-channel gamma/gain/bias proxy, identity at init."""

    def __init__(self, channels=3):
        super().__init__()
        self.log_gamma = nn.Parameter(torch.zeros(channels))
        self.gain = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        g = torch.exp(self.log_gamma)[None, :, None, None]
        y = x.clamp(1e-4, 1.0).pow(g) * self.gain[None, :, None, None] + self.bias[None, :, None, None]
        return (x + 0.25 * (y - x)).clamp(0, 1)
