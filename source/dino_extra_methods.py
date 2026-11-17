import torch
import torch.nn as nn
import torch.nn.functional as F

from semantic_backbone import DEFAULT_DINO_WEIGHTS, FrozenDINOFeatures


class DinoRoutingBias(nn.Module):
    """D1: add a zero-initialized global DINO bias to the dark selector logits."""

    def __init__(self, weights_path=DEFAULT_DINO_WEIGHTS, hidden=32, out_dim=4):
        super().__init__()
        self.teacher = FrozenDINOFeatures(weights_path=DEFAULT_DINO_WEIGHTS if weights_path is None else weights_path)
        self.head = nn.Sequential(
            nn.Linear(768, hidden), nn.GELU(), nn.Linear(hidden, out_dim)
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x):
        _, cls = self.teacher(x)
        return self.head(cls[-1])


class DinoDarkCross(nn.Module):
    """D2: dark cluster tokens query frozen DINO patch tokens."""

    def __init__(self, token_dim=22, weights_path=DEFAULT_DINO_WEIGHTS, dim=64, heads=4):
        super().__init__()
        self.teacher = FrozenDINOFeatures(weights_path=DEFAULT_DINO_WEIGHTS if weights_path is None else weights_path)
        self.q = nn.Linear(token_dim, dim)
        self.kv = nn.Linear(768, 2 * dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True, dropout=0.0)
        self.out = nn.Linear(dim, token_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        patches, _ = self.teacher(x)
        return patches[-1]

    def cross(self, t, tokens):
        q = self.q(t)
        kv = self.kv(tokens.flatten(2).transpose(1, 2))
        k, v = kv.chunk(2, dim=-1)
        a, _ = self.attn(q, k, v, need_weights=False)
        return t + self.out(a)


class DinoChannelGate(nn.Module):
    """D3: global DINO descriptor predicts a near-identity channel scale at D1."""

    def __init__(self, channels=64, weights_path=DEFAULT_DINO_WEIGHTS, hidden=64):
        super().__init__()
        self.teacher = FrozenDINOFeatures(weights_path=DEFAULT_DINO_WEIGHTS if weights_path is None else weights_path)
        self.head = nn.Sequential(
            nn.Linear(768, hidden), nn.GELU(), nn.Linear(hidden, channels)
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x):
        _, cls = self.teacher(x)
        scale = 1.0 + 0.1 * torch.tanh(self.head(cls[-1]))
        return scale[:, :, None, None]


class DinoColorGain(nn.Module):
    """D4: bounded per-channel DINO-conditioned colour gain/bias at output."""

    def __init__(self, weights_path=DEFAULT_DINO_WEIGHTS, hidden=32):
        super().__init__()
        self.teacher = FrozenDINOFeatures(weights_path=DEFAULT_DINO_WEIGHTS if weights_path is None else weights_path)
        self.head = nn.Sequential(
            nn.Linear(768, hidden), nn.GELU(), nn.Linear(hidden, 6)
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x):
        _, cls = self.teacher(x)
        return 0.1 * torch.tanh(self.head(cls[-1]))


def dino_perceptual_loss(teacher, pred, target):
    """L1: patch-level cosine consistency with the frozen DINO teacher."""
    pp, _ = teacher.forward_features(pred.clamp(0.0, 1.0), with_grad=True)
    tp, _ = teacher.forward_features(target, with_grad=False)
    loss = 0.0
    for p, t in zip(pp, tp):
        p = F.normalize(p.flatten(2), dim=1)
        t = F.normalize(t.flatten(2), dim=1)
        loss = loss + (1.0 - (p * t).sum(dim=1)).mean()
    return loss / len(pp)
