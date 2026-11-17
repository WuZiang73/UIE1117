import hashlib
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from third_party.dinov2_vendor.models.vision_transformer import vit_base


DEFAULT_DINO_WEIGHTS = "/data1/wza/third_party/dino_weights/vitb14.pth"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


class FrozenDINOFeatures(nn.Module):
    """Frozen DINOv2 ViT-B/14 teacher.

    Input is the same augmented RGB tensor as the restoration model, in [0,1].
    It is resized to 224x224, ImageNet-normalized, and passed once through the
    teacher.  Three zero-based blocks [2,6,10] are returned as patch + CLS.
    """

    def __init__(self, weights_path=DEFAULT_DINO_WEIGHTS, layers=(2, 6, 10),
                 resolution=224, patch_size=14, embed_dim=768):
        super().__init__()
        self.weights_path = os.path.abspath(weights_path)
        self.layers = tuple(layers)
        self.resolution = int(resolution)
        self.patch_size = int(patch_size)
        self.embed_dim = int(embed_dim)
        self.grid = self.resolution // self.patch_size
        if self.resolution % self.patch_size != 0:
            raise ValueError("DINO resolution must be divisible by patch size")

        model = vit_base(
            patch_size=self.patch_size,
            img_size=518,
            num_register_tokens=0,
            block_chunks=0,
            init_values=1.0,
            interpolate_antialias=False,
            interpolate_offset=0.1,
        )
        state = torch.load(self.weights_path, map_location="cpu")
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"DINO weight mismatch missing={missing[:8]} unexpected={unexpected[:8]}"
            )
        del state
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        self.model = model
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)
        self.weights_sha256 = sha256_file(self.weights_path)

    def train(self, mode=True):
        # The teacher must remain in eval mode even when the parent is trained.
        super().train(False)
        return self

    def forward_features(self, x, with_grad=False):
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(f"expected Bx3xHxW, got {tuple(x.shape)}")
        x = F.interpolate(
            x.float(), size=(self.resolution, self.resolution),
            mode="bicubic", align_corners=False,
        ).clamp(0.0, 1.0)
        x = (x - self.mean) / self.std
        if with_grad:
            outputs = self.model.get_intermediate_layers(
                x, n=list(self.layers), reshape=True,
                return_class_token=True, norm=True,
            )
        else:
            with torch.no_grad():
                outputs = self.model.get_intermediate_layers(
                    x, n=list(self.layers), reshape=True,
                    return_class_token=True, norm=True,
                )
        patches = []
        cls_tokens = []
        for patch, cls in outputs:
            if patch.shape[-2:] != (self.grid, self.grid):
                raise RuntimeError(f"unexpected DINO patch grid {tuple(patch.shape)}")
            patches.append(patch.contiguous())
            cls_tokens.append(cls.contiguous())
        return patches, cls_tokens

    def forward(self, x):
        return self.forward_features(x, with_grad=False)
