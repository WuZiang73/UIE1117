import torch
import torch.nn as nn
import torch.nn.functional as F

from global_semantic_modules import LayerGate, SemanticBasis
from semantic_backbone import DEFAULT_DINO_WEIGHTS, FrozenDINOFeatures


class DinoC01(nn.Module):
    """Frozen DINOv2 guidance for the C01 restoration trunk."""

    MODES = ("C01D1G", "C01D1A", "C01D1R")

    def __init__(
        self,
        mode,
        channels=64,
        width=32,
        heads=4,
        weights_path=DEFAULT_DINO_WEIGHTS,
        layers=(2, 6, 10),
    ):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"unsupported C01 mode {mode}")
        self.mode = mode
        self.channels = int(channels)
        self.width = int(width)
        self.heads = int(heads)
        if self.width % self.heads != 0:
            raise ValueError("width must be divisible by heads")
        self.teacher = FrozenDINOFeatures(
            weights_path=DEFAULT_DINO_WEIGHTS if weights_path is None else weights_path,
            layers=layers,
        )
        self.basis = SemanticBasis(width=self.width, num_layers=len(layers))
        self.layer_gate = LayerGate()

        if mode == "C01D1G":
            self.gate_head = nn.Sequential(
                nn.Conv2d(self.width, self.width, 3, 1, 1),
                nn.GELU(),
                nn.Conv2d(self.width, 1, 1),
            )
            nn.init.zeros_(self.gate_head[-1].weight)
            nn.init.zeros_(self.gate_head[-1].bias)
        elif mode == "C01D1A":
            self.q = nn.Conv2d(self.channels, self.width, 1, bias=False)
            self.v = nn.Conv2d(self.channels, self.width, 1, bias=False)
            self.attn = nn.MultiheadAttention(
                self.width, self.heads, batch_first=True, dropout=0.0
            )
            self.out = nn.Conv2d(self.width, self.channels, 1, bias=False)
            nn.init.zeros_(self.out.weight)
            self.gate_scale = nn.Parameter(torch.ones(1))
        else:
            self.out = nn.Conv2d(self.width, self.channels, 1, bias=False)
            nn.init.zeros_(self.out.weight)
            self.gate_scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        patches, cls_tokens = self.teacher(x)
        spatial, _, z = self.basis(patches, cls_tokens, x)
        gate = self.layer_gate(z)
        fused = sum(
            spatial[i] * gate[:, i].view(-1, 1, 1, 1) for i in range(len(spatial))
        )
        return {"spatial": fused, "z": z, "gates": gate}

    def apply(self, pre_dark, dark, aux):
        if self.mode == "C01D1G":
            logits = self.gate_head(aux["spatial"])
            logits = F.interpolate(
                logits, size=dark.shape[-2:], mode="bilinear", align_corners=False
            )
            return dark + 0.1 * torch.tanh(logits) * pre_dark

        spatial = aux["spatial"]
        if self.mode == "C01D1A":
            b, c, h, w = dark.shape
            q = self.q(F.adaptive_avg_pool2d(dark, spatial.shape[-2:]))
            v = self.v(F.adaptive_avg_pool2d(dark, spatial.shape[-2:]))
            q = q.flatten(2).transpose(1, 2)
            k = spatial.flatten(2).transpose(1, 2)
            v = v.flatten(2).transpose(1, 2)
            out, _ = self.attn(q, k, v, need_weights=False)
            out = out.transpose(1, 2).reshape(b, self.width, *spatial.shape[-2:])
        else:
            out = spatial

        out = F.interpolate(
            out, size=dark.shape[-2:], mode="bilinear", align_corners=False
        )
        out = self.out(out)
        return dark + 0.25 * self.gate_scale * out

    def trainable_parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @torch.no_grad()
    def teacher_parameter_count(self):
        return sum(p.numel() for p in self.teacher.parameters())
