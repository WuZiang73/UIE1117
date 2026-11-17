import torch
import torch.nn as nn
import torch.nn.functional as F

from global_semantic_modules import LayerGate, SemanticBasis
from semantic_backbone import DEFAULT_DINO_WEIGHTS, FrozenDINOFeatures


class DinoResidualBranch(nn.Module):
    """T1: parallel DINO-to-RGB residual branch."""

    def __init__(self, out_channels=3, weights_path=DEFAULT_DINO_WEIGHTS, width=32):
        super().__init__()
        self.teacher = FrozenDINOFeatures(weights_path=DEFAULT_DINO_WEIGHTS if weights_path is None else weights_path)
        self.basis = SemanticBasis(width=width, num_layers=3)
        self.gate = LayerGate()
        self.decoder = nn.Sequential(
            nn.Conv2d(width, 64, 3, 1, 1), nn.GELU(), nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 32, 3, 1, 1), nn.GELU(), nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(32, 32, 3, 1, 1), nn.GELU(), nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(32, 16, 3, 1, 1), nn.GELU(), nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(16, out_channels, 3, 1, 1),
        )
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)

    def forward(self, x):
        patches, cls = self.teacher(x)
        spatial, _, z = self.basis(patches, cls, x)
        g = self.gate(z)
        t = sum(spatial[i] * g[:, i].view(-1, 1, 1, 1) for i in range(len(spatial)))
        return self.decoder(t)


class DinoOutputMoE(nn.Module):
    """T2: DINO global router over three output residual heads."""

    def __init__(self, feat_channels=32, out_channels=3, weights_path=DEFAULT_DINO_WEIGHTS):
        super().__init__()
        self.teacher = FrozenDINOFeatures(weights_path=DEFAULT_DINO_WEIGHTS if weights_path is None else weights_path)
        self.router = nn.Sequential(nn.Linear(768, 32), nn.GELU(), nn.Linear(32, 3))
        nn.init.zeros_(self.router[-1].weight)
        nn.init.zeros_(self.router[-1].bias)
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(feat_channels, feat_channels, 3, 1, 1), nn.GELU(),
                nn.Conv2d(feat_channels, out_channels, 3, 1, 1),
            ) for _ in range(3)
        ])
        for head in self.heads:
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    def forward(self, x):
        _, cls = self.teacher(x)
        return torch.softmax(self.router(cls[-1]), dim=-1)

    def apply(self, feat, weights):
        out = 0.0
        for k, head in enumerate(self.heads):
            out = out + weights[:, k].view(-1, 1, 1, 1) * head(feat)
        return out


class DinoEncoderCross(nn.Module):
    """T3: encoder2 tokens query DINO patch tokens."""

    def __init__(self, channels=64, weights_path=DEFAULT_DINO_WEIGHTS, dim=64, grid=16):
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
        q = self.q(F.adaptive_avg_pool2d(f, self.grid)).flatten(2).transpose(1, 2)
        kv = self.kv(tokens)
        k, v = kv.chunk(2, dim=1)
        k = k.flatten(2).transpose(1, 2)
        v = v.flatten(2).transpose(1, 2)
        a, _ = self.attn(q, k, v, need_weights=False)
        a = a.transpose(1, 2).reshape(b, -1, self.grid, self.grid)
        a = F.interpolate(a, size=(h, w), mode="bilinear", align_corners=False)
        return self.out(a)


class DinoGlobalAdaIN(nn.Module):
    """T4: global DINO descriptor produces near-identity AdaIN at D1/D2."""

    def __init__(self, d1_channels=64, d2_channels=32, weights_path=DEFAULT_DINO_WEIGHTS):
        super().__init__()
        self.teacher = FrozenDINOFeatures(weights_path=DEFAULT_DINO_WEIGHTS if weights_path is None else weights_path)
        self.norm = nn.LayerNorm(768)
        self.head1 = nn.Linear(768, 2 * d1_channels)
        self.head2 = nn.Linear(768, 2 * d2_channels)
        nn.init.zeros_(self.head1.weight); nn.init.zeros_(self.head1.bias)
        nn.init.zeros_(self.head2.weight); nn.init.zeros_(self.head2.bias)

    def forward(self, x):
        _, cls = self.teacher(x)
        g = self.norm(cls[-1])
        d1 = 0.02 * torch.tanh(self.head1(g))
        d2 = 0.02 * torch.tanh(self.head2(g))
        return d1, d2

    @staticmethod
    def apply(feat, d):
        c = feat.shape[1]
        gamma, beta = d[:, :c], d[:, c:]
        return feat * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]


class DinoWaveletMod(nn.Module):
    """T5: DINO global descriptor modulates D1 Haar sub-bands."""

    def __init__(self, channels=64, weights_path=DEFAULT_DINO_WEIGHTS):
        super().__init__()
        self.teacher = FrozenDINOFeatures(weights_path=DEFAULT_DINO_WEIGHTS if weights_path is None else weights_path)
        self.head = nn.Sequential(nn.Linear(768, 64), nn.GELU(), nn.Linear(64, 4 * channels))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x):
        _, cls = self.teacher(x)
        return 0.1 * torch.tanh(self.head(cls[-1]))

    def apply(self, f, coeff):
        c = f.shape[1]
        ll, lh, hl, hh = _haar_dwt(f)
        bands = [ll, lh, hl, hh]
        out = []
        for i, b in enumerate(bands):
            g = coeff[:, i * c:(i + 1) * c].view(-1, c, 1, 1)
            out.append(b * g)
        return f + _haar_idwt(*out)


def _haar_kernels(device, dtype, channels):
    ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]], device=device, dtype=dtype)
    lh = torch.tensor([[-0.5, -0.5], [0.5, 0.5]], device=device, dtype=dtype)
    hl = torch.tensor([[-0.5, 0.5], [-0.5, 0.5]], device=device, dtype=dtype)
    hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]], device=device, dtype=dtype)
    return torch.stack([ll, lh, hl, hh], 0).unsqueeze(1).repeat(channels, 1, 1, 1)


def _haar_dwt(x):
    c = x.shape[1]
    k = _haar_kernels(x.device, x.dtype, c)
    return F.conv2d(x, k, stride=2, groups=c).chunk(4, 1)


def _haar_idwt(ll, lh, hl, hh):
    c = ll.shape[1]
    k = _haar_kernels(ll.device, ll.dtype, c)
    return F.conv_transpose2d(torch.cat([ll, lh, hl, hh], 1), k, stride=2, groups=c)
