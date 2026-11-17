import torch
import torch.nn as nn


class ISPOutput(nn.Module):
    """Per-channel output ISP proxy, identity at init."""

    def __init__(self, channels=3):
        super().__init__()
        self.log_gamma = nn.Parameter(torch.zeros(channels))
        self.gain = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, pred):
        g = torch.exp(self.log_gamma)[None, :, None, None]
        p = pred.clamp(0.0, 1.0)
        delta = p.pow(g) * self.gain[None, :, None, None] + self.bias[None, :, None, None] - p
        return pred + 0.25 * delta
