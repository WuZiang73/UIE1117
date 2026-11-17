import torch
import torch.nn as nn
import torch.nn.functional as F


class DinoOutputArbiter(nn.Module):
    """DINO-conditioned suppression of the DG7 residual."""

    MODES = ("image", "pixel")

    def __init__(self, mode="image", in_dim=102, width=32):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"unsupported arbiter mode {mode}")
        self.mode = mode
        if mode == "image":
            self.head = nn.Sequential(
                nn.Linear(in_dim, width),
                nn.GELU(),
                nn.Linear(width, 1),
            )
        else:
            self.head = nn.Sequential(
                nn.Conv2d(32, width, 3, 1, 1),
                nn.GELU(),
                nn.Conv2d(width, 1, 1),
            )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, pre_dg7, post_dg7, aux):
        if self.mode == "image":
            logits = self.head(aux["z"])
            alpha = (1.0 + torch.tanh(logits)).clamp(0.0, 1.0)
            alpha = alpha[:, :, None, None]
        else:
            logits = self.head(aux["spatial"])
            logits = F.interpolate(
                logits, size=post_dg7.shape[-2:], mode="bilinear", align_corners=False
            )
            alpha = (1.0 + torch.tanh(logits)).clamp(0.0, 1.0)
        return pre_dg7 + alpha * (post_dg7 - pre_dg7)


class DinoSemanticArbiter(nn.Module):
    """DINO feature extractor plus output arbitration, without C01 modulation."""

    def __init__(
        self,
        mode="image",
        weights_path=None,
        layers=(2, 6, 10),
        width=32,
    ):
        super().__init__()
        from global_semantic_modules import LayerGate, SemanticBasis
        from semantic_backbone import DEFAULT_DINO_WEIGHTS, FrozenDINOFeatures

        self.teacher = FrozenDINOFeatures(
            weights_path=DEFAULT_DINO_WEIGHTS if weights_path is None else weights_path,
            layers=layers,
        )
        self.basis = SemanticBasis(width=width, num_layers=len(layers))
        self.layer_gate = LayerGate()
        self.arbiter = DinoOutputArbiter(mode=mode)

    def forward(self, x):
        patches, cls_tokens = self.teacher(x)
        spatial, _, z = self.basis(patches, cls_tokens, x)
        gate = self.layer_gate(z)
        fused = sum(
            spatial[i] * gate[:, i].view(-1, 1, 1, 1) for i in range(len(spatial))
        )
        return {"spatial": fused, "z": z, "gates": gate}

    def apply(self, pre_dg7, post_dg7, aux):
        return self.arbiter(pre_dg7, post_dg7, aux)
