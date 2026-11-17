import torch
import torch.nn as nn
import torch.nn.functional as F

from semantic_backbone import DEFAULT_DINO_WEIGHTS, FrozenDINOFeatures


class DinoWaveletVariant(nn.Module):
    """Five small variants around T5's D1 Haar sub-band modulation.

    U1: LayerNorm + 0.02 bound, all bands
    U2: high bands only, 0.05 bound
    U3: all bands + per-band learnable scale
    U4: high bands, patch-GAP descriptor instead of CLS
    U5: high bands + spatial DINO gate
    """

    def __init__(self, channels=64, variant="U1", weights_path=DEFAULT_DINO_WEIGHTS):
        super().__init__()
        if variant not in ("U1", "U2", "U3", "U4", "U5"):
            raise ValueError(variant)
        self.variant = variant
        self.channels = channels
        self.teacher = FrozenDINOFeatures(
            weights_path=DEFAULT_DINO_WEIGHTS if weights_path is None else weights_path
        )
        self.norm = nn.LayerNorm(768)
        out_dim = 4 * channels if variant in ("U1", "U3") else 3 * channels
        self.head = nn.Sequential(
            nn.Linear(768, 64), nn.GELU(), nn.Linear(64, out_dim)
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        if variant == "U3":
            self.band_scale = nn.Parameter(torch.ones(4))
        if variant == "U5":
            self.gate = nn.Conv2d(768, 1, 1)
            nn.init.zeros_(self.gate.weight)
            nn.init.zeros_(self.gate.bias)

    def _descriptor(self, patches, cls):
        if self.variant in ("U4", "U5"):
            return patches[-1].mean(dim=(2, 3))
        return cls[-1]

    def forward(self, x):
        patches, cls = self.teacher(x)
        d = self.norm(self._descriptor(patches, cls))
        coeff = self.head(d)
        if self.variant == "U1":
            coeff = 0.02 * torch.tanh(coeff)
        elif self.variant == "U2" or self.variant == "U4" or self.variant == "U5":
            coeff = 0.05 * torch.tanh(coeff)
        else:
            coeff = 0.1 * torch.tanh(coeff).view(-1, 4, self.channels) * self.band_scale.view(1, 4, 1)
            coeff = coeff.reshape(-1, 4 * self.channels)
        if self.variant == "U5":
            return coeff, 1.0 + 0.1 * torch.tanh(self.gate(patches[-1]))
        return coeff

    def apply(self, f, out):
        c = f.shape[1]
        ll, lh, hl, hh = _haar_dwt(f)
        if self.variant in ("U1", "U3"):
            coeff = out.view(-1, 4, c)
            bands = [ll, lh, hl, hh]
            delta = _haar_idwt(*[
                bands[i] * coeff[:, i].view(-1, c, 1, 1) for i in range(4)
            ])
        else:
            coeff = out[0] if self.variant == "U5" else out
            coeff = coeff.view(-1, 3, c)
            delta = _haar_idwt(
                torch.zeros_like(ll),
                lh * coeff[:, 0].view(-1, c, 1, 1),
                hl * coeff[:, 1].view(-1, c, 1, 1),
                hh * coeff[:, 2].view(-1, c, 1, 1),
            )
            if self.variant == "U5":
                g = F.interpolate(out[1], size=f.shape[-2:], mode="bilinear", align_corners=False)
                delta = delta * g
        return f + delta


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
