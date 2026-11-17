import torch
import torch.nn as nn

from psf_recovery.kernels import kernel_bank
from psf_recovery.operators import _conv3


def _elliptical_bank(radius=8):
    banks = [kernel_bank((0.0,), radius=radius)]
    for s in (0.6, 1.2, 2.4):
        for aspect in ((1.0, 1.0), (1.5, 2.0 / 3.0), (2.0 / 3.0, 1.5)):
            banks.append(kernel_bank((s,), radius=radius, aspect=aspect))
    return torch.cat(banks, 0)


class TrainablePSF(nn.Module):
    """Global per-channel PSF residual. Identity at initialization."""

    def __init__(self, elliptical=False, channels=3, radius=8, residual_scale=0.25):
        super().__init__()
        kernels = _elliptical_bank(radius) if elliptical else kernel_bank((0.0, 0.6, 1.2, 2.4), radius=radius)
        self.register_buffer("kernels", kernels)
        self.coeff = nn.Parameter(torch.zeros(channels, kernels.shape[0]))
        with torch.no_grad():
            self.coeff[:, 0] = 1.0
        self.bias = nn.Parameter(torch.zeros(channels))
        self.residual_scale = residual_scale

    def forward(self, pred):
        blurred = _conv3(pred, self.kernels)
        y = (blurred * self.coeff[None, :, :, None, None]).sum(2) + self.bias[None, :, None, None]
        return pred + self.residual_scale * (y - pred)
