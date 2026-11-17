import torch
import torch.nn as nn
import torch.nn.functional as F


class LowFreqResidual(nn.Module):
    """Bounded low-resolution color/tone residual."""

    def __init__(self, size=64, width=32, channels=3):
        super().__init__()
        self.size = int(size)
        self.head = nn.Sequential(
            nn.Conv2d(2 * channels, width, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(width, width, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(width, channels, 3, 1, 1),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, pred, x):
        p = F.adaptive_avg_pool2d(pred, self.size)
        xr = F.adaptive_avg_pool2d(x, self.size)
        d = self.head(torch.cat([p, xr], dim=1))
        d = F.interpolate(d, size=pred.shape[-2:], mode="bilinear", align_corners=False)
        return pred + 0.25 * torch.tanh(d)


class LowFreqResidualGated(nn.Module):
    """Low-resolution residual with a scalar gate."""

    def __init__(self, size=64, width=32, channels=3):
        super().__init__()
        self.size = int(size)
        self.head = nn.Sequential(
            nn.Conv2d(2 * channels, width, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(width, width, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(width, channels, 3, 1, 1),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        self.gate = nn.Parameter(torch.ones(1))

    def forward(self, pred, x):
        p = F.adaptive_avg_pool2d(pred, self.size)
        xr = F.adaptive_avg_pool2d(x, self.size)
        d = self.head(torch.cat([p, xr], dim=1))
        d = F.interpolate(d, size=pred.shape[-2:], mode="bilinear", align_corners=False)
        return pred + 0.25 * torch.tanh(d) * torch.tanh(self.gate)
