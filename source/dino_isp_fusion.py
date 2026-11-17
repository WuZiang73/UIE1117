"""Single ISP restoration backbone plus a frozen DINOv2 fusion branch.

The graph exchanges native restoration values, using DINO only as a metric.
The precision variant routes with the current encoder's propagated noise proxy.
No pretrained restoration checkpoint, output mixture, or extra objective.
"""
import torch
from torch import nn
from torch.nn import functional as F

from optical_regional_isp import RegionalISPUNet
from semantic_backbone import FrozenDINOFeatures


class GraphExchange(nn.Module):
    def __init__(self, channels, strength=0.5):
        super().__init__()
        self.gain = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.strength = strength

    def forward(self, feature, affinity, receiver_gate):
        base = F.adaptive_avg_pool2d(feature, (16, 16))
        tokens = base.flatten(2).transpose(1, 2)
        message = (affinity @ tokens - tokens).transpose(1, 2).reshape_as(base)
        message = message * receiver_gate
        message = F.interpolate(message, size=feature.shape[-2:], mode='bilinear', align_corners=False)
        return feature + self.strength * self.gain.tanh() * message


class DirectFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.project = nn.Sequential(nn.Conv2d(768, 32, 1), nn.GELU(), nn.Conv2d(32, channels, 1))
        nn.init.zeros_(self.project[-1].weight)
        nn.init.zeros_(self.project[-1].bias)

    def forward(self, feature, patch):
        delta = self.project(patch)
        return feature + 0.1 * F.interpolate(delta, size=feature.shape[-2:], mode='bilinear', align_corners=False)


class DinoISPUNet(RegionalISPUNet):
    MODES = ('precision_graph', 'semantic_graph', 'rgb_graph', 'direct')

    def __init__(self, *args, fusion_mode='precision_graph', **kwargs):
        assert fusion_mode in self.MODES
        weights = kwargs.get('semantic_weights')
        super().__init__(*args, regional_isp_mode='diagonal', **kwargs)
        self.fusion_mode = fusion_mode
        self.fusion_enabled = True
        # Modules are created after the native backbone; its tensors are loaded
        # from exactly the same shared random initialization for every method.
        with torch.random.fork_rng(devices=[]):
            if fusion_mode != 'rgb_graph':
                self.dino_teacher = FrozenDINOFeatures(weights_path=weights, layers=(10,))
            else:
                self.dino_teacher = None
            if fusion_mode == 'direct':
                self.fusion_bot = DirectFusion(128)
                self.fusion_dec = DirectFusion(64)
            else:
                self.fusion_bot = GraphExchange(128)
                self.fusion_dec = GraphExchange(64)
        yy, xx = torch.meshgrid(torch.linspace(0, 1, 16), torch.linspace(0, 1, 16), indexing='ij')
        coords = torch.stack((yy, xx), -1).reshape(256, 2)
        self.register_buffer('graph_distance', (coords[:, None] - coords[None]).square().sum(-1), persistent=False)
        self._condition = None
        self._noise = None
        self.observer = self._observe_noise
        # Hooks preserve the original ISP forward implementation byte-for-byte.
        self.bottleneck.register_forward_hook(self._fuse_bot)
        self.decoder1.register_forward_hook(self._fuse_dec)

    def _observe_noise(self, f, normalized, covariance, gain, ridge, variance):
        # Routing treats this as a reliability measurement, not a trainable
        # shortcut through kernel magnitudes. Native ISP gradients are intact.
        self._noise = covariance.detach().diagonal(dim1=-2, dim2=-1).mean(-1)[:, None]

    def _graph(self, x):
        if self.fusion_mode == 'rgb_graph':
            patch = F.adaptive_avg_pool2d(x, (16, 16))
        else:
            patch = self.dino_teacher(x)[0][0]
        if self.fusion_mode == 'direct':
            return patch
        tokens = F.normalize(patch.flatten(2).transpose(1, 2), dim=-1, eps=1e-6)
        logits = (tokens @ tokens.transpose(1, 2)) / 0.15 - 2.0 * self.graph_distance[None]
        return logits

    def _apply_fusion(self, feature, module):
        if not self.fusion_enabled:
            return feature
        if self.fusion_mode == 'direct':
            return module(feature, self._condition)
        logits = self._condition
        if self.fusion_mode == 'precision_graph':
            noise = F.interpolate(self._noise, size=(16, 16), mode='bilinear', align_corners=False).clamp_min(1e-8)
            noise_tokens = noise.flatten(2).transpose(1, 2)
            logits = logits - noise_tokens.transpose(1, 2).log()
            affinity = logits.softmax(-1)
            message_noise = affinity @ noise_tokens
            receiver_gate = (noise_tokens / (noise_tokens + message_noise)).transpose(1, 2).reshape_as(noise)
        else:
            affinity = logits.softmax(-1)
            # A constant 1/2 matches the precision branch at uniform noise.
            receiver_gate = feature.new_tensor(0.5)
        return module(feature, affinity, receiver_gate)

    def _fuse_bot(self, module, inputs, output):
        return self._apply_fusion(output, self.fusion_bot)

    def _fuse_dec(self, module, inputs, output):
        return self._apply_fusion(output, self.fusion_dec)

    def forward(self, x, return_pair=False):
        self._condition = self._graph(x) if self.fusion_enabled else None
        try:
            return super().forward(x, return_pair=return_pair)
        finally:
            self._condition = None
            self._noise = None
