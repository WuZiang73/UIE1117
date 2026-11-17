import torch
import torch.nn as nn
import torch.nn.functional as F

from global_semantic_modules import LayerGate, SemanticBasis
from semantic_backbone import DEFAULT_DINO_WEIGHTS, FrozenDINOFeatures


class SemanticSkipFusion(nn.Module):
    """S1: DINO-conditioned near-identity gate on both U-Net skips."""

    def __init__(self, channels, weights_path=DEFAULT_DINO_WEIGHTS, width=32):
        super().__init__()
        self.teacher = FrozenDINOFeatures(weights_path=DEFAULT_DINO_WEIGHTS if weights_path is None else weights_path)
        self.basis = SemanticBasis(width=width, num_layers=3)
        self.gates = nn.ModuleDict({k: LayerGate() for k in channels})
        self.p_enc = nn.ModuleDict({k: nn.Conv2d(c, 16, 1) for k, c in channels.items()})
        self.p_dec = nn.ModuleDict({k: nn.Conv2d(c, 16, 1) for k, c in channels.items()})
        self.p_sem = nn.ModuleDict({k: nn.Conv2d(width, 16, 1) for k in channels})
        self.out = nn.ModuleDict({k: nn.Conv2d(16, c, 1) for k, c in channels.items()})
        for module in self.out.values():
            nn.init.zeros_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x):
        patches, cls = self.teacher(x)
        spatial, _, z = self.basis(patches, cls, x)
        T = {}
        for name in self.gates:
            g = self.gates[name](z)
            T[name] = sum(spatial[i] * g[:, i].view(-1, 1, 1, 1) for i in range(len(spatial)))
        return {"T": T}

    def apply(self, name, enc, dec, T):
        if T.shape[-2:] != enc.shape[-2:]:
            T = F.interpolate(T, size=enc.shape[-2:], mode="bilinear", align_corners=False)
        h = self.p_enc[name](enc) + self.p_dec[name](dec) + self.p_sem[name](T)
        h = F.layer_norm(h.permute(0, 2, 3, 1), (16,)).permute(0, 3, 1, 2)
        v = self.out[name](F.gelu(h))
        return enc * (1.0 + 0.1 * torch.tanh(v))


class DinoDynamicConv(nn.Module):
    """S2: DINO global descriptor predicts a low-rank dynamic 1x1 convolution."""

    def __init__(self, channels=64, weights_path=DEFAULT_DINO_WEIGHTS, rank=4):
        super().__init__()
        self.teacher = FrozenDINOFeatures(weights_path=DEFAULT_DINO_WEIGHTS if weights_path is None else weights_path)
        self.a = nn.Parameter(torch.empty(channels, rank))
        self.b = nn.Parameter(torch.empty(rank, channels))
        nn.init.normal_(self.a, std=0.02)
        nn.init.normal_(self.b, std=0.02)
        self.head = nn.Sequential(nn.Linear(768, 32), nn.GELU(), nn.Linear(32, rank))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x):
        _, cls = self.teacher(x)
        return 0.1 * torch.tanh(self.head(cls[-1]))

    def apply(self, f, coeff):
        w = torch.einsum("cr,br,rd->bcd", self.a, coeff, self.b)
        return torch.einsum("bcd,bdhw->bchw", w, f)


class DinoPromptBottleneck(nn.Module):
    """S3: learnable prompt memory reads DINO patches and AdaLN-modulates B0."""

    def __init__(self, channels=128, weights_path=DEFAULT_DINO_WEIGHTS, num_prompts=4, dim=128):
        super().__init__()
        self.teacher = FrozenDINOFeatures(weights_path=DEFAULT_DINO_WEIGHTS if weights_path is None else weights_path)
        self.prompts = nn.Parameter(torch.randn(num_prompts, dim) * 0.02)
        self.kv = nn.Linear(768, 2 * dim)
        self.attn = nn.MultiheadAttention(dim, 4, batch_first=True, dropout=0.0)
        self.head = nn.Linear(dim, 2 * channels)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        patches, _ = self.teacher(x)
        tokens = patches[-1].flatten(2).transpose(1, 2)
        b = tokens.shape[0]
        kv = self.kv(tokens)
        k, v = kv.chunk(2, dim=-1)
        q = self.prompts[None].expand(b, -1, -1)
        a, _ = self.attn(q, k, v, need_weights=False)
        a = a.mean(dim=1)
        d = self.head(a)
        gamma, beta = d.chunk(2, dim=-1)
        return 0.1 * torch.tanh(gamma), 0.1 * torch.tanh(beta)

    def apply(self, f, gamma, beta):
        return f * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]


class DinoClusterGate(nn.Module):
    """S4: DINO global descriptor reweights the three dark-cluster masses."""

    def __init__(self, num_clusters=3, weights_path=DEFAULT_DINO_WEIGHTS, hidden=32):
        super().__init__()
        self.teacher = FrozenDINOFeatures(weights_path=DEFAULT_DINO_WEIGHTS if weights_path is None else weights_path)
        self.head = nn.Sequential(
            nn.Linear(768, hidden), nn.GELU(), nn.Linear(hidden, num_clusters)
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x):
        _, cls = self.teacher(x)
        g = 1.0 + 0.1 * torch.tanh(self.head(cls[-1]))
        return g[:, :, None, None]


class DinoDecoder2Cross(nn.Module):
    """S5: full-resolution decoder2 tokens query DINO patch tokens."""

    def __init__(self, channels=32, weights_path=DEFAULT_DINO_WEIGHTS, dim=64, grid=32):
        super().__init__()
        self.teacher = FrozenDINOFeatures(weights_path=DEFAULT_DINO_WEIGHTS if weights_path is None else weights_path)
        self.grid = grid
        self.q = nn.Conv2d(channels, dim, 1, bias=False)
        self.kv = nn.Conv2d(768, 2 * dim, 1, bias=False)
        self.attn = nn.MultiheadAttention(dim, 4, batch_first=True, dropout=0.0)
        self.out = nn.Conv2d(dim, channels, 1, bias=False)
        nn.init.zeros_(self.out.weight)

    def forward(self, x):
        patches, _ = self.teacher(x)
        return patches[-1]

    def apply(self, f, tokens):
        b, _, h, w = f.shape
        q = self.q(F.adaptive_avg_pool2d(f, self.grid))
        kv = self.kv(tokens)
        k, v = kv.chunk(2, dim=1)
        q = q.flatten(2).transpose(1, 2)
        k = k.flatten(2).transpose(1, 2)
        v = v.flatten(2).transpose(1, 2)
        a, _ = self.attn(q, k, v, need_weights=False)
        a = a.transpose(1, 2).reshape(b, -1, self.grid, self.grid)
        a = F.interpolate(a, size=(h, w), mode="bilinear", align_corners=False)
        return self.out(a)
