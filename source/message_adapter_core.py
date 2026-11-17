"""Low-rank interpretation of native message differences; no DINO values injected."""
import torch
from torch import nn
from torch.nn import functional as F
from precision_graph_core import DinoISPUNet as OriginalPrecisionGraph


class MessageAdapter(nn.Module):
    def __init__(self, channels, rank=32):
        super().__init__()
        self.gain = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.strength = 0.5
        self.down = nn.Conv2d(channels, rank, 1, bias=False)
        self.up = nn.Conv2d(rank, channels, 1, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, feature, affinity, receiver_gate):
        base = F.adaptive_avg_pool2d(feature, (16, 16))
        tokens = base.flatten(2).transpose(1, 2)
        message = (affinity @ tokens-tokens).transpose(1, 2).reshape_as(base)*receiver_gate
        native = feature+self.strength*self.gain.tanh()*F.interpolate(
            message, size=feature.shape[-2:], mode='bilinear', align_corners=False)
        converted = self.up(self.down(message))
        return native+0.1*F.interpolate(converted, size=feature.shape[-2:], mode='bilinear', align_corners=False)


class DinoISPUNet(OriginalPrecisionGraph):
    MODES = ('native_adapter', 'plain_adapter', 'rgb_adapter', 'precision_graph', 'rgb_graph')

    def __init__(self, *args, fusion_mode='native_adapter', **kwargs):
        assert fusion_mode in ('native_adapter', 'plain_adapter', 'rgb_adapter')
        self.adapter_mode = fusion_mode
        base = 'rgb_graph' if fusion_mode == 'rgb_adapter' else 'precision_graph'
        super().__init__(*args, fusion_mode=base, **kwargs)
        with torch.random.fork_rng(devices=[]):
            self.fusion_bot = MessageAdapter(128)
            self.fusion_dec = MessageAdapter(64)
        # Physical routing is shared by DINO and RGB cases. The plain case
        # removes only fusion routing's physical condition; original ISP stays.
        self.fusion_mode = 'semantic_graph' if fusion_mode == 'plain_adapter' else 'precision_graph'

    def _graph(self, x):
        if self.dino_teacher is not None:
            return super()._graph(x)
        patch = F.adaptive_avg_pool2d(x, (16, 16))
        tokens = F.normalize(patch.flatten(2).transpose(1, 2), dim=-1, eps=1e-6)
        return (tokens @ tokens.transpose(1, 2))/0.15-2*self.graph_distance[None]
