import torch
import torch.nn as nn
import torch.nn.functional as F

from .operators import (
    from_linear,
    highpass_adjoint,
    highpass_state,
    make_kernels,
    render_state,
    rgb_to_state,
    state_adjoint,
    state_to_rgb,
    to_linear,
)


class Calibrator(nn.Module):
    def __init__(self, channels=3, nk=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, 32, 3, stride=4, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 32, 3, stride=4, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(32, channels + channels * nk + channels)
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
        with torch.no_grad():
            self.fc.bias[:channels] = 2.0
            self.fc.bias[channels:channels + channels * nk] = 0.0
            for c in range(channels):
                self.fc.bias[channels + c * nk] = 3.0
            self.fc.bias[-channels:] = -4.0
        self.channels = channels
        self.nk = nk

    def forward(self, x):
        v = self.fc(self.net(x).flatten(1))
        c, nk = self.channels, self.nk
        t = 0.05 + 0.95 * torch.sigmoid(v[:, :c])[:, :, None, None]
        pi = torch.softmax(v[:, c:c + c * nk].view(-1, c, nk), -1)
        b = torch.sigmoid(v[:, -c:])[:, :, None, None]
        return t, pi, b


class InternalOpticalUpdate(nn.Module):
    def __init__(self, feat_channels=64, mode="full", nk=4, eta=0.05):
        super().__init__()
        self.mode = mode
        self.nk = nk
        self.eta = eta
        self.register_buffer("kernels", make_kernels())
        self.rin = nn.Sequential(
            nn.Conv2d(2 * feat_channels, 32, 1),
            nn.GELU(),
            nn.Conv2d(32, 12, 1),
        )
        self.cal = Calibrator(3, nk)
        self.rout = nn.Conv2d(12, feat_channels, 1)
        nn.init.zeros_(self.rout.weight)
        nn.init.zeros_(self.rout.bias)
        self.adapter = None
        if mode == "adapter":
            self.adapter = nn.Conv2d(2 * feat_channels, feat_channels, 1)
            nn.init.zeros_(self.adapter.weight)
            nn.init.zeros_(self.adapter.bias)

    def forward(self, x, e2, q1):
        if self.mode == "adapter":
            return self.adapter(torch.cat([e2, q1], 1)), {}
        with torch.enable_grad():
            x_lin = to_linear(x)
            y_obs = rgb_to_state(x_lin)
            t, pi, b = self.cal(x)
            if self.mode == "delta":
                pi = torch.zeros_like(pi)
                pi[:, :, 0] = 1.0
            z0 = y_obs + 0.1 * torch.tanh(self.rin(torch.cat([e2, q1], 1)))
            beta = rgb_to_state(b.expand(-1, -1, x.shape[-2], x.shape[-1]))
            r = y_obs - render_state(z0, self.kernels, t, pi, b) - beta
            qr = highpass_state(r)
            g = state_adjoint(z0, self.kernels, t, pi, b, highpass_adjoint(qr, qr))
            z1 = z0 + self.eta * g
            dz = z1 - z0
            delta = self.rout(dz)
        return delta, {"t": t.detach().mean(), "pi": pi.detach().mean(), "dz": dz.detach().abs().mean()}
