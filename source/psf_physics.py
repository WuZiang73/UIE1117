import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gaussian_kernels(sigma, size=9):
    # sigma: B,C,M; returns B,C,M,size,size normalized
    b, c, m = sigma.shape
    r = size // 2
    y = torch.arange(-r, r + 1, device=sigma.device, dtype=sigma.dtype)
    yy, xx = torch.meshgrid(y, y, indexing="ij")
    d = (xx * xx + yy * yy).view(1, 1, 1, size, size)
    s = sigma.clamp(0.4, 4.0).view(b, c, m, 1, 1)
    k = torch.exp(-d / (2.0 * s * s))
    k = k / k.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
    return k


def _apply_kernel(x, kernel):
    # x: B,C,H,W; kernel: B,C,k,k
    b, c, h, w = x.shape
    k = kernel.shape[-1]
    return F.conv2d(
        x.reshape(1, b * c, h, w),
        kernel.reshape(b * c, 1, k, k),
        padding=k // 2,
        groups=b * c,
    ).reshape(b, c, h, w)


class PSFCalibrator(nn.Module):
    def __init__(self, e2_ch=64, b0_ch=128, bins=4, hidden=64, kernel_size=9):
        super().__init__()
        self.bins = bins
        self.kernel_size = kernel_size
        self.l_conv = nn.Sequential(
            nn.Conv2d(b0_ch, 32, 1), nn.GELU(), nn.Conv2d(32, bins, 1)
        )
        self.global_mlp = nn.Sequential(
            nn.Linear(6 + e2_ch + b0_ch, hidden), nn.GELU(), nn.Linear(hidden, 33)
        )
        self.register_buffer("range_bins", torch.tensor([0.25, 0.5, 0.75, 1.0]), persistent=False)

    def forward(self, x, e2, b0):
        b = x.shape[0]
        stats = torch.cat([
            x.mean(dim=(2, 3)), x.std(dim=(2, 3), unbiased=False),
            e2.mean(dim=(2, 3)), b0.mean(dim=(2, 3)),
        ], dim=1)
        raw = self.global_mlp(stats)
        l_logits = self.l_conv(F.adaptive_avg_pool2d(b0, 16))
        m = torch.softmax(l_logits, dim=1)                       # B,4,16,16
        l_map = (m * self.range_bins.view(1, 4, 1, 1)).sum(1, keepdim=True)
        beta = 0.05 + 1.95 * torch.sigmoid(raw[:, 0:3])
        back = torch.sigmoid(raw[:, 3:6])
        pi = torch.softmax(raw[:, 6:15].view(b, 3, 3), dim=2)    # B,3ch,3basis
        a = raw[:, 15:24].view(b, 3, 3)
        bcoef = raw[:, 24:33].view(b, 3, 3)
        sig = F.softplus(a[:, :, :, None] + bcoef[:, :, :, None] * self.range_bins.view(1, 1, 1, 4)) + 0.5
        sig = sig.clamp(0.5, 4.0)                                 # B,3,3,4
        kb = _gaussian_kernels(sig.reshape(b, 3, 12), self.kernel_size)
        kernels = kb.view(b, 3, 3, 4, self.kernel_size, self.kernel_size)
        pi4 = pi[:, :, :, None, None, None]
        kernels = (kernels * pi4).sum(2)                          # B,3,4,k,k
        kernel_sum = kernels.sum(dim=(-2, -1))
        # effective kernel at mean relative range for P4 / diagnostics
        sig_eff = F.softplus(a + bcoef * self.range_bins.mean()) + 0.5
        sig_eff = sig_eff.clamp(0.5, 4.0)
        k_eff = _gaussian_kernels(sig_eff, self.kernel_size)
        k_eff = (k_eff * pi[:, :, :, None, None]).sum(2)
        spread = (pi * sig.mean(dim=3)).sum(2)                   # B,3
        conf = torch.sigmoid(raw[:, 6:7]).mean(dim=1, keepdim=True)
        return {
            "L": l_map, "m": m, "beta": beta, "B": back, "pi": pi,
            "sigma": sig, "kernel": kernels, "kernel_eff": k_eff,
            "kernel_sum": kernel_sum, "spread": spread, "conf": conf,
            "raw": raw,
        }


def render_underwater(j, aux, size=256):
    b, c, h, w = j.shape
    L = F.interpolate(aux["L"], size=(h, w), mode="bilinear", align_corners=False)
    T = torch.exp(-aux["beta"][:, :, None, None] * L)
    Jt = j * T
    I = 0.0
    for bi in range(4):
        mb = F.interpolate(aux["m"][:, bi:bi + 1], size=(h, w), mode="bilinear", align_corners=False)
        I = I + mb * _apply_kernel(Jt, aux["kernel"][:, :, bi])
    I = I + aux["B"][:, :, None, None] * (1.0 - T)
    return I


def psf_stats(aux):
    return {
        "L_mean": float(aux["L"].mean()),
        "beta_mean": float(aux["beta"].mean()),
        "kernel_sum_mean": float(aux["kernel_sum"].mean()),
        "spread_mean": float(aux["spread"].mean()),
        "conf_mean": float(aux["conf"].mean()),
    }
