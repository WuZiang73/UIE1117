import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from global_semantic_modules import LayerGate, SemanticBasis
from semantic_backbone import DEFAULT_DINO_WEIGHTS, FrozenDINOFeatures


def _mha(q, k, v, heads, bias=None):
    b, nq, d = q.shape
    nk = k.shape[1]
    hd = d // heads
    q = q.view(b, nq, heads, hd).transpose(1, 2)
    k = k.view(b, nk, heads, hd).transpose(1, 2)
    v = v.view(b, nk, heads, hd).transpose(1, 2)
    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(hd)
    if bias is not None:
        scores = scores + bias
    a = torch.softmax(scores, dim=-1)
    out = torch.matmul(a, v).transpose(1, 2).reshape(b, nq, d)
    return out


class F1Bridge(nn.Module):
    """F1: B1-style dual semantic branches fused with a C0 anchor at B0."""

    def __init__(self, channels=128, weights_path=DEFAULT_DINO_WEIGHTS, width=32):
        super().__init__()
        self.teacher = FrozenDINOFeatures(weights_path=DEFAULT_DINO_WEIGHTS if weights_path is None else weights_path)
        self.basis = SemanticBasis(width=width, num_layers=3)
        self.layout_gate = LayerGate()
        self.local_gate = LayerGate()
        self.b_layout = nn.Conv2d(channels, width, 1)
        self.b_local = nn.Conv2d(channels, width, 1)
        self.layout_film = nn.Linear(width, 2 * width)
        self.local_film = nn.Conv2d(width, 2 * width, 1)
        nn.init.zeros_(self.layout_film.weight); nn.init.zeros_(self.layout_film.bias)
        nn.init.zeros_(self.local_film.weight); nn.init.zeros_(self.local_film.bias)
        self.fuse = nn.Conv2d(channels + 3 * width, channels, 1)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        patches, cls = self.teacher(x)
        spatial, cls_out, z = self.basis(patches, cls, x)
        gl = self.layout_gate(z)
        layout = sum(cls_out[i] * gl[:, i:i + 1] for i in range(3))
        gm = self.local_gate(z)
        local = sum(spatial[i] * gm[:, i].view(-1, 1, 1, 1) for i in range(3))
        return {"layout": layout, "local": local}

    def apply(self, b0, cond):
        bl = self.b_layout(b0)
        gl = self.layout_film(cond["layout"])[:, :, None, None]
        g1, b1 = gl.chunk(2, dim=1)
        bl = bl * (1.0 + 0.1 * torch.tanh(g1)) + 0.1 * torch.tanh(b1)

        bloc = self.b_local(b0)
        t = F.interpolate(cond["local"], size=b0.shape[-2:], mode="bilinear", align_corners=False)
        gloc = self.local_film(t)
        g2, b2 = gloc.chunk(2, dim=1)
        bloc = bloc * (1.0 + 0.1 * torch.tanh(g2)) + 0.1 * torch.tanh(b2)

        delta = self.fuse(torch.cat([b0, bl, bloc, bl * bloc], dim=1))
        return b0 + self.gate * delta


class F2Affinity(nn.Module):
    """F2: DINO provides affinity K, C0 B0 provides V and output values."""

    def __init__(self, channels=128, dim=64, heads=4, weights_path=DEFAULT_DINO_WEIGHTS):
        super().__init__()
        self.teacher = FrozenDINOFeatures(weights_path=DEFAULT_DINO_WEIGHTS if weights_path is None else weights_path)
        self.heads = heads
        self.q = nn.Linear(channels, dim)
        self.k = nn.Linear(768, dim)
        self.v = nn.Linear(channels, dim)
        self.out = nn.Conv2d(dim, channels, 1)
        self.bias = nn.Parameter(torch.zeros(1, 1, 256, 256))
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        patches, _ = self.teacher(x)
        return torch.stack(patches, dim=0).mean(dim=0)

    def apply(self, b0, dino_patch):
        b, c, h, w = b0.shape
        base = F.avg_pool2d(b0, 4)                       # B,128,16,16
        qt = base.flatten(2).transpose(1, 2)
        vt = qt
        kt = dino_patch.flatten(2).transpose(1, 2)
        q = self.q(qt)
        k = self.k(kt)
        v = self.v(vt)
        r = _mha(q, k, v, self.heads, self.bias)
        r = r.transpose(1, 2).reshape(b, -1, 16, 16)
        r = F.interpolate(r, size=(h, w), mode="bilinear", align_corners=False)
        r = self.out(r)
        base_up = F.interpolate(base, size=(h, w), mode="bilinear", align_corners=False)
        return b0 + self.gate * (r - base_up)


class F3Slots(nn.Module):
    """F3: DINO queries fill slots from C0 B0; slots reconstruct C0 values."""

    def __init__(self, channels=128, dim=64, heads=4, num_slots=8, grid=64, weights_path=DEFAULT_DINO_WEIGHTS):
        super().__init__()
        self.teacher = FrozenDINOFeatures(weights_path=DEFAULT_DINO_WEIGHTS if weights_path is None else weights_path)
        self.grid = grid
        self.heads = heads
        self.q_base = nn.Parameter(torch.randn(num_slots, dim) * 0.02)
        self.cls_proj = nn.Linear(3 * 768, dim)
        nn.init.zeros_(self.cls_proj.weight); nn.init.zeros_(self.cls_proj.bias)
        self.kv = nn.Linear(channels, 2 * dim)
        self.q2 = nn.Linear(channels, dim)
        self.k2 = nn.Linear(dim, dim)
        self.v2 = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, channels)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        _, cls = self.teacher(x)
        return self.cls_proj(torch.cat(cls, dim=1))

    def apply(self, b0, q_off):
        b, c, h, w = b0.shape
        tokens = b0.flatten(2).transpose(1, 2)             # B,4096,128
        kv = self.kv(tokens)
        k, v = kv.chunk(2, dim=-1)
        q = self.q_base[None] + 0.1 * torch.tanh(q_off)[:, None, :]
        slots = _mha(q, k, v, self.heads)                  # B,8,64
        q2 = self.q2(tokens)
        k2 = self.k2(slots)
        v2 = self.v2(slots)
        rec = self.out(_mha(q2, k2, v2, self.heads))       # B,4096,128
        rec = rec.transpose(1, 2).reshape(b, c, h, w)
        return b0 + self.gate * (rec - b0)


class F4SkipRoute(nn.Module):
    """F4: DINO selects among C0 skip paths; values stay in E2 feature space."""

    def __init__(self, channels=64, weights_path=DEFAULT_DINO_WEIGHTS):
        super().__init__()
        self.teacher = FrozenDINOFeatures(weights_path=DEFAULT_DINO_WEIGHTS if weights_path is None else weights_path)
        self.patch_proj = nn.Conv2d(768, 32, 1)
        self.route = nn.Sequential(
            nn.Conv2d(32 + channels + channels, 64, 3, 1, 1), nn.GELU(),
            nn.Conv2d(64, 3, 1),
        )
        nn.init.zeros_(self.route[-1].weight); nn.init.zeros_(self.route[-1].bias)
        self.local = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, groups=channels), nn.GELU(),
            nn.Conv2d(channels, channels, 1),
        )
        self.context = nn.Sequential(
            nn.Conv2d(channels, channels, 5, 1, 2, groups=channels), nn.GELU(),
            nn.Conv2d(channels, channels, 1),
        )
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        patches, _ = self.teacher(x)
        return self.patch_proj(torch.stack(patches, 0).mean(0))

    def apply(self, e2, q1, dino_patch):
        ep = F.avg_pool2d(e2, 8)
        qp = F.avg_pool2d(q1, 8)
        logits = self.route(torch.cat([dino_patch, ep, qp], dim=1))
        a = torch.softmax(logits, dim=1)
        a1 = F.interpolate(a[:, 1:2], size=e2.shape[-2:], mode="bilinear", align_corners=False) * self.gate
        a2 = F.interpolate(a[:, 2:3], size=e2.shape[-2:], mode="bilinear", align_corners=False) * self.gate
        return e2 + a1 * (self.local(e2) - e2) + a2 * (self.context(e2) - e2)


class F5DeformSkip(nn.Module):
    """F5: DINO predicts skip sampling offsets; values are sampled from E2."""

    def __init__(self, channels=64, weights_path=DEFAULT_DINO_WEIGHTS, max_offset=4.0):
        super().__init__()
        self.teacher = FrozenDINOFeatures(weights_path=DEFAULT_DINO_WEIGHTS if weights_path is None else weights_path)
        self.patch_proj = nn.Conv2d(768, 32, 1)
        self.offset = nn.Sequential(
            nn.Conv2d(32 + channels + channels, 64, 3, 1, 1), nn.GELU(),
            nn.Conv2d(64, 2, 1),
        )
        nn.init.zeros_(self.offset[-1].weight); nn.init.zeros_(self.offset[-1].bias)
        self.gate = nn.Parameter(torch.ones(1))
        self.max_offset = float(max_offset)
        size = 128
        ys = (torch.arange(size, dtype=torch.float32) + 0.5) / size * 2 - 1
        xs = (torch.arange(size, dtype=torch.float32) + 0.5) / size * 2 - 1
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        self.register_buffer("base_grid", torch.stack([gx, gy], -1)[None], persistent=False)

    def forward(self, x):
        patches, _ = self.teacher(x)
        return self.patch_proj(torch.stack(patches, 0).mean(0))

    def apply(self, e2, q1, dino_patch):
        ep = F.avg_pool2d(e2, 8)
        qp = F.avg_pool2d(q1, 8)
        off = self.offset(torch.cat([dino_patch, ep, qp], dim=1))
        off = F.interpolate(off, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        off = self.max_offset * torch.tanh(off)
        b, _, h, w = e2.shape
        off = off.permute(0, 2, 3, 1) * torch.tensor([2.0 / w, 2.0 / h], device=e2.device)
        grid = self.base_grid.to(e2.device) + off
        grid = grid.clamp(-1.0, 1.0)
        warped = F.grid_sample(e2, grid, mode="bilinear", padding_mode="border", align_corners=False)
        return e2 + self.gate * (warped - e2)
