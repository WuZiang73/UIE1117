import torch
import torch.nn as nn
import torch.nn.functional as F

from semantic_backbone import DEFAULT_DINO_WEIGHTS, FrozenDINOFeatures


class SemanticBasis(nn.Module):
    """Shared token projection for the three frozen DINO layers.

    A_l and LN_l are shared between the patch tokens and the CLS token of the
    same layer.  D_l is the spatial 32-channel semantic map; d_l is the
    32-channel CLS embedding.
    """

    def __init__(self, in_dim=768, width=32, num_layers=3):
        super().__init__()
        self.width = width
        self.proj = nn.ModuleList([nn.Linear(in_dim, width, bias=True) for _ in range(num_layers)])
        self.norm = nn.ModuleList([nn.LayerNorm(width) for _ in range(num_layers)])

    def forward(self, patches, cls_tokens, x):
        if len(patches) != len(self.proj) or len(cls_tokens) != len(self.proj):
            raise ValueError("DINO layer count mismatch")
        spatial = []
        cls_out = []
        for proj, norm, patch, cls in zip(self.proj, self.norm, patches, cls_tokens):
            b, c, h, w = patch.shape
            p = patch.flatten(2).transpose(1, 2)          # B,N,768
            d = F.gelu(norm(proj(p)))                    # B,N,32
            spatial.append(d.transpose(1, 2).reshape(b, self.width, h, w))
            cls_out.append(F.gelu(norm(proj(cls))))      # B,32
        rgb_mean = x.mean(dim=(2, 3))
        rgb_std = x.std(dim=(2, 3), unbiased=False)
        z = torch.cat(cls_out + [rgb_mean, rgb_std], dim=1)  # B,102
        return spatial, cls_out, z


class LayerGate(nn.Module):
    """Port-specific softmax over the three DINO layers."""

    def __init__(self, in_dim=102, hidden=32, num_layers=3):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, num_layers)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, z):
        return torch.softmax(self.fc2(self.act(self.fc1(z))), dim=-1)


class FiLMModulator(nn.Module):
    """Identity-initialized FiLM around InstanceNorm.

    f' = f + alpha * (gamma * InstanceNorm(f) + beta - f)
    alpha=0, gamma=1, beta=0 at init.  The blend alpha is the minimal
    correction needed to make the doc's gamma=1/beta=0 FiLM exactly equal to
    the C0 feature at initialization (pure FiLM alone is not identity).
    """

    def __init__(self, channels, width=32):
        super().__init__()
        self.fc1 = nn.Linear(width, width)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(width, 2 * channels)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        with torch.no_grad():
            self.fc2.bias[:channels].fill_(1.0)
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, f, t):
        if t.shape[-2:] != f.shape[-2:]:
            t = F.interpolate(t, size=f.shape[-2:], mode="bilinear", align_corners=False)
        h = t.permute(0, 2, 3, 1)
        h = self.act(self.fc1(h))
        h = self.fc2(h).permute(0, 3, 1, 2)
        gamma, beta = h.chunk(2, dim=1)
        film = gamma * F.instance_norm(f) + beta
        return f + self.alpha * (film - f)


class SemanticCrossAttention(nn.Module):
    """R2: bottleneck tokens query frozen DINO patch tokens."""

    def __init__(self, channels=128, dino_dim=768, dim=128, heads=4, grid=16):
        super().__init__()
        self.grid = grid
        self.q = nn.Conv2d(channels, dim, 1, bias=False)
        self.kv = nn.Conv2d(dino_dim, 2 * dim, 1, bias=False)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True, dropout=0.0)
        self.out = nn.Conv2d(dim, channels, 1, bias=False)
        nn.init.zeros_(self.out.weight)

    def forward(self, f, raw):
        b, c, h, w = f.shape
        q = self.q(F.adaptive_avg_pool2d(f, self.grid))
        kv = self.kv(raw)
        k, v = kv.chunk(2, dim=1)
        q = q.flatten(2).transpose(1, 2)
        k = k.flatten(2).transpose(1, 2)
        v = v.flatten(2).transpose(1, 2)
        a, _ = self.attn(q, k, v, need_weights=False)
        a = a.transpose(1, 2).reshape(b, -1, self.grid, self.grid)
        a = F.interpolate(a, size=(h, w), mode="bilinear", align_corners=False)
        return f + self.out(a)


class ColorMatrixHead(nn.Module):
    """R3: global 3x3 colour transform conditioned on DINO + decoder2."""

    def __init__(self, feat_channels=32, sem_channels=32, hidden=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feat_channels + sem_channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, 12),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        self.register_buffer("eye", torch.eye(3).view(1, 3, 3), persistent=False)

    def forward(self, pred, feature, sem):
        g = torch.cat([feature.mean(dim=(2, 3)), sem.mean(dim=(2, 3))], dim=1)
        d = self.mlp(g).view(-1, 3, 4)
        m = self.eye + d[:, :, :3]
        b = d[:, :, 3]
        return torch.einsum("bij,bjhw->bihw", m, pred) + b[:, :, None, None]


class GlobalSemanticPyramid(nn.Module):
    """N0 (CLS) / G1 (patch pyramid) global semantic branch.

    The frozen DINO teacher is run once.  Each restoration port has its own
    layer gate and its own zero-initialized 1x1 output projection.  The neutral
    state is exactly the C0 restoration model.
    """

    def __init__(self, channels, mode="G1", weights_path=DEFAULT_DINO_WEIGHTS,
                 layers=(2, 6, 10), width=32, shared_gate=False):
        super().__init__()
        if mode not in ("N0", "G1", "R1", "R2", "R3"):
            raise ValueError(f"unsupported semantic mode {mode}")
        if weights_path is None:
            weights_path = DEFAULT_DINO_WEIGHTS
        self.mode = mode
        self.channels = dict(channels)
        self.width = width
        self.shared_gate = bool(shared_gate)
        self.teacher = FrozenDINOFeatures(weights_path=weights_path, layers=layers)
        self.basis = SemanticBasis(width=width, num_layers=len(layers))
        gate_dim = 3 * width + 6
        if mode == "R2":
            self.gate = LayerGate(in_dim=gate_dim)
        elif self.shared_gate:
            self.gate = LayerGate(in_dim=gate_dim)
        else:
            self.gates = nn.ModuleDict({name: LayerGate(in_dim=gate_dim) for name in channels})
        if mode == "R1":
            self.film = nn.ModuleDict({
                name: FiLMModulator(c, width=width) for name, c in channels.items()
            })
        elif mode in ("N0", "G1"):
            self.out = nn.ModuleDict({
                name: nn.Conv2d(width, c, 1, bias=False) for name, c in channels.items()
            })
            for module in self.out.values():
                nn.init.zeros_(module.weight)

    def forward(self, x):
        patches, cls_tokens = self.teacher(x)
        spatial, cls_out, z = self.basis(patches, cls_tokens, x)
        if self.mode == "R2":
            g = self.gate(z)
            raw = sum(patches[i] * g[:, i].view(-1, 1, 1, 1) for i in range(len(patches)))
            return {"raw": raw, "gates": g}
        if self.shared_gate:
            g = self.gate(z)
            gates = {name: g for name in self.channels}
        else:
            gates = {name: self.gates[name](z) for name in self.channels}

        outputs = {}
        for name in self.channels:
            g = gates[name]
            if self.mode == "N0":
                v = sum(g[:, i:i + 1] * cls_out[i] for i in range(len(cls_out)))
                outputs[name] = v[:, :, None, None]
            else:
                maps = [spatial[i] * g[:, i].view(-1, 1, 1, 1) for i in range(len(spatial))]
                outputs[name] = sum(maps)
        return {"T": outputs, "gates": gates}

    def apply(self, name, feature, T):
        if self.mode == "R2":
            raise RuntimeError("R2 is applied through SemanticCrossAttention, not apply()")
        if name not in self.channels:
            raise KeyError(f"unknown semantic port {name}")
        if feature.shape[1] != self.channels[name]:
            raise ValueError(f"channel mismatch at {name}: {feature.shape[1]} vs {self.channels[name]}")
        if self.mode == "R1":
            return self.film[name](feature, T)
        if T.shape[-2:] != feature.shape[-2:]:
            T = F.interpolate(T, size=feature.shape[-2:], mode="bilinear", align_corners=False)
        return feature + 0.1 * self.out[name](T)

    def trainable_parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
