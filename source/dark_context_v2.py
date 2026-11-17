import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_CTX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analysis", "dark_context_v2_uieb.npz")
EPS = 1e-6

def reflect_avg(x, k):
    p = k // 2
    return F.avg_pool2d(F.pad(x, (p, p, p, p), mode="reflect"), k, 1, 0)

def median3(x):
    p = F.pad(x, (1, 1, 1, 1), mode="reflect")
    patches = F.unfold(p, kernel_size=3).view(x.shape[0], x.shape[1], 9, -1)
    return patches.median(dim=2).values.view_as(x)

def sobel_mag(y):
    kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
                      device=y.device, dtype=y.dtype).view(1, 1, 3, 3)
    ky = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]],
                      device=y.device, dtype=y.dtype).view(1, 1, 3, 3)
    p = F.pad(y, (1, 1, 1, 1), mode="reflect")
    gx = F.conv2d(p, kx)
    gy = F.conv2d(p, ky)
    return torch.sqrt(gx * gx + gy * gy + 1e-8)

def wavg(x, w):
    return F.avg_pool2d(x * w, 2) / (F.avg_pool2d(w, 2) + EPS)

def full_descriptors(x):
    """x: [B,3,H,W] in [0,1]. Returns full-resolution descriptors + dark mask."""
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    mu = reflect_avg(y, 7)
    mdark = ((0.45 - mu) / 0.30).clamp(0.0, 1.0)
    var = (reflect_avg(y * y, 7) - mu * mu).clamp_min(0.0)
    std = torch.sqrt(var + 1e-8)
    cg = torch.log((r + EPS) / (g + EPS)).clamp(-3.0, 3.0)
    cb = torch.log((b + EPS) / (g + EPS)).clamp(-3.0, 3.0)
    ys = reflect_avg(y, 3)
    grad = sobel_mag(ys)
    hp = median3((y - ys).abs())
    return {"mu": mu, "std": std, "cg": cg, "cb": cb, "grad": grad, "hp": hp, "mdark": mdark}

class DarkContextV2(nn.Module):
    """Absolute-luminance dark candidates + fixed-stats soft clustering at 1/2 scale."""
    def __init__(self, npz_path=DEFAULT_CTX, temperature=1.0):
        super().__init__()
        data = np.load(npz_path)
        self.register_buffer("dmean", torch.tensor(data["mean"], dtype=torch.float32).view(1, -1, 1, 1))
        self.register_buffer("dscale", torch.tensor(data["scale"], dtype=torch.float32).view(1, -1, 1, 1))
        self.register_buffer("prototypes", torch.tensor(data["prototypes"], dtype=torch.float32))
        self.num_clusters = int(self.prototypes.shape[0])
        self.temperature = temperature
        self.tau_max = float(data["tau_max"]) if "tau_max" in data else 0.05

    def forward(self, x):
        d = full_descriptors(x)
        w = d.pop("mdark")
        w2 = F.avg_pool2d(w, 2)
        feats = [wavg(d[k], w) for k in ("mu", "std", "cg", "cb", "grad", "hp")]
        z = torch.cat(feats, dim=1)
        z = ((z - self.dmean) / self.dscale.clamp_min(1e-3)).clamp(-5.0, 5.0)
        # reliability: structure vs structure+high-pass
        r = d["grad"] / (d["grad"] + d["hp"] + 1e-3)
        r2 = wavg(r, w).clamp(0.0, 1.0)
        k = self.prototypes.shape[0]
        diff = z.unsqueeze(1) - self.prototypes.view(1, k, 6, 1, 1)
        dist = (diff * diff).mean(dim=2)  # B,K,h,w
        a = torch.softmax(-dist / self.temperature, dim=1)
        return {"z": z, "assign": a, "mask": w2, "w": w2 * a, "r": r2, "tau_max": self.tau_max}
