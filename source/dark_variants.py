import torch
import torch.nn as nn
import torch.nn.functional as F


def _norm_map(x, eps=1e-6):
    m = x.mean(dim=(-2, -1), keepdim=True)
    s = x.std(dim=(-2, -1), keepdim=True)
    return (x - m) / (s + eps)


def _sobel(y):
    kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
                      device=y.device, dtype=y.dtype).view(1, 1, 3, 3)
    ky = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]],
                      device=y.device, dtype=y.dtype).view(1, 1, 3, 3)
    gx = F.conv2d(y, kx, padding=1)
    gy = F.conv2d(y, ky, padding=1)
    return torch.sqrt(gx * gx + gy * gy + 1e-8)


class DarkPrior(nn.Module):
    """Input-level dark descriptors used by M1-M4."""
    def forward(self, x):
        min_rgb = x.min(dim=1, keepdim=True).values
        max_rgb = x.max(dim=1, keepdim=True).values
        D = 1.0 - F.max_pool2d(min_rgb, 3, 1, 1)
        C = max_rgb - min_rgb
        y = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        mu = F.avg_pool2d(y, 7, 1, 3)
        var = F.avg_pool2d(y * y, 7, 1, 3) - mu * mu
        std = torch.sqrt(var.clamp_min(1e-8))
        snr = std / (mu + 1e-4)
        E = _sobel(y)
        E = E / (E.mean(dim=(-2, -1), keepdim=True) + 1e-4)
        R = E / (std + 1e-4)
        low = F.avg_pool2d(y, 3, 1, 1)
        high = y - low
        hf = high.abs() / (low.abs() + 1e-4)
        q = torch.quantile(D.flatten(1), 0.65, dim=1).view(-1, 1, 1, 1)
        mask = torch.sigmoid((D - q) / 0.05)
        return {"D": D, "C": C, "E": E, "N": snr, "R": R, "HF": hf, "mask": mask}


class SoftClusterRouter(nn.Module):
    def __init__(self, zdim, num_clusters, temperature=0.15):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(num_clusters, zdim) * 0.1)
        self.temperature = temperature

    def forward(self, z):
        z = F.normalize(z, dim=1)
        p = F.normalize(self.prototypes, dim=1).view(1, -1, z.shape[1], 1, 1)
        d = (z.unsqueeze(1) - p).pow(2).sum(dim=2)
        return F.softmax(-d / self.temperature, dim=1)


def _features(dark, names, size):
    zs = []
    for name in names:
        z = dark[name]
        if z.shape[-2:] != size:
            z = F.interpolate(z, size=size, mode="bilinear", align_corners=False)
        zs.append(_norm_map(z))
    return torch.cat(zs, dim=1)


class _BaseDarkModule(nn.Module):
    zdim_names = ()
    def __init__(self, channels, num_clusters=3, temperature=0.15):
        super().__init__()
        self.channels = channels
        self.num_clusters = num_clusters
        self.router = SoftClusterRouter(len(self.zdim_names), num_clusters, temperature)

    def _mask(self, dark, size):
        m = dark["mask"]
        if m.shape[-2:] != size:
            m = F.interpolate(m, size=size, mode="bilinear", align_corners=False)
        return m


class M1DarkFiLM(_BaseDarkModule):
    zdim_names = ("D", "C")
    def __init__(self, channels, num_clusters=3, temperature=0.15):
        super().__init__(channels, num_clusters, temperature)
        self.norm = nn.GroupNorm(1, channels, affine=False)
        self.gamma = nn.Parameter(torch.zeros(num_clusters, channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(num_clusters, channels, 1, 1))
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, f, dark):
        a = self.router(_features(dark, self.zdim_names, f.shape[-2:]))
        g = (a.unsqueeze(2) * self.gamma.view(1, -1, self.channels, 1, 1)).sum(1)
        b = (a.unsqueeze(2) * self.beta.view(1, -1, self.channels, 1, 1)).sum(1)
        delta = g * self.norm(f) + b
        return f + self.alpha * self._mask(dark, f.shape[-2:]) * delta


class _LocalExpert(nn.Module):
    def __init__(self, channels, kernel=3):
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, kernel, 1, kernel // 2, groups=channels, bias=False)
        self.act = nn.GELU()
        self.pw = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, x):
        return self.pw(self.act(self.dw(x)))


class M2DarkMoE(_BaseDarkModule):
    zdim_names = ("D", "C", "E", "N")
    def __init__(self, channels, num_clusters=4, temperature=0.15):
        super().__init__(channels, num_clusters, temperature)
        self.experts = nn.ModuleList([_LocalExpert(channels, 3) for _ in range(num_clusters)])
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, f, dark):
        z = self.zdim_names
        feats = _features(dark, z, f.shape[-2:])
        feats = torch.cat([feats[:, :-1], -feats[:, -1:]], dim=1)
        a = self.router(feats)
        delta = 0
        for k, expert in enumerate(self.experts):
            delta = delta + a[:, k:k + 1] * expert(f)
        return f + self.alpha * self._mask(dark, f.shape[-2:]) * delta


class _FreqExpert(nn.Module):
    def __init__(self, channels, kernel):
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, kernel, 1, kernel // 2, groups=channels, bias=False)
        self.act = nn.GELU()
        self.pw = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, x):
        return self.pw(self.act(self.dw(x)))


class M3DarkFreqDual(_BaseDarkModule):
    zdim_names = ("D", "N", "R", "HF")
    def __init__(self, channels, num_clusters=4, temperature=0.15):
        super().__init__(channels, num_clusters, temperature)
        self.exp_l = nn.ModuleList([_FreqExpert(channels, 7) for _ in range(num_clusters)])
        self.exp_h = nn.ModuleList([_FreqExpert(channels, 3) for _ in range(num_clusters)])
        self.alpha_l = nn.Parameter(torch.zeros(num_clusters))
        self.alpha_h = nn.Parameter(torch.zeros(num_clusters))

    def forward(self, f, dark):
        a = self.router(_features(dark, self.zdim_names, f.shape[-2:]))
        f_l = F.avg_pool2d(f, 3, 1, 1)
        f_h = f - f_l
        delta_l = 0
        delta_h = 0
        for k in range(self.num_clusters):
            ak = a[:, k:k + 1]
            delta_l = delta_l + ak * self.alpha_l[k] * self.exp_l[k](f_l)
            delta_h = delta_h + ak * self.alpha_h[k] * self.exp_h[k](f_h)
        return f + self._mask(dark, f.shape[-2:]) * (delta_l + delta_h)


class _ConvGRUCell(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gates = nn.Sequential(
            nn.Conv2d(2 * channels, 2 * channels, 3, 1, 1, groups=2 * channels, bias=False),
            nn.Conv2d(2 * channels, 2 * channels, 1, bias=True),
        )
        self.candidate = nn.Sequential(
            nn.Conv2d(2 * channels, 2 * channels, 3, 1, 1, groups=2 * channels, bias=False),
            nn.Conv2d(2 * channels, channels, 1, bias=True),
        )

    def forward(self, x, h):
        zr = torch.sigmoid(self.gates(torch.cat([x, h], dim=1)))
        z, r = zr.chunk(2, dim=1)
        h_tilde = torch.tanh(self.candidate(torch.cat([x, r * h], dim=1)))
        return (1.0 - z) * h + z * h_tilde


class M4DarkRecurrentGate(_BaseDarkModule):
    zdim_names = ("D", "C", "N", "E")
    def __init__(self, channels, num_clusters=3, temperature=0.15, steps=2):
        super().__init__(channels, num_clusters, temperature)
        self.cell = _ConvGRUCell(channels)
        self.steps = steps
        self.gate = nn.Parameter(torch.zeros(num_clusters, channels, 1, 1))
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, f, dark):
        h = f
        for _ in range(self.steps):
            h = self.cell(f, h)
        a = self.router(_features(dark, self.zdim_names, f.shape[-2:]))
        g = (a.unsqueeze(2) * torch.tanh(self.gate).view(1, -1, self.channels, 1, 1)).sum(1)
        return f + self.alpha * self._mask(dark, f.shape[-2:]) * g * (h - f)


def build_dark_module(variant, channels, num_clusters=3, temperature=0.15):
    if variant == "M1":
        return M1DarkFiLM(channels, num_clusters, temperature)
    if variant == "M2":
        return M2DarkMoE(channels, 4, temperature)
    if variant == "M3":
        return M3DarkFreqDual(channels, 4, temperature)
    if variant == "M4":
        return M4DarkRecurrentGate(channels, num_clusters, temperature)
    raise ValueError(variant)
