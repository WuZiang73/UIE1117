"""Two separable factors: frozen DINO relation and native-message channel conversion."""
import torch
from precision_graph_core import DinoISPUNet as OriginalGraph
from message_adapter_core import MessageAdapter


class DinoISPUNet(OriginalGraph):
    # Parent construction validates its own internal mode through self.MODES.
    MODES = ('spatial_gain', 'dino_gain', 'spatial_adapter', 'dino_adapter',
             'semantic_graph', 'rgb_graph')

    def __init__(self, *args, fusion_mode='dino_adapter', **kwargs):
        assert fusion_mode in ('spatial_gain', 'dino_gain', 'spatial_adapter', 'dino_adapter')
        self.factor_mode = fusion_mode
        self.use_dino_relation = fusion_mode.startswith('dino_')
        self.use_channel_conversion = fusion_mode.endswith('_adapter')
        super().__init__(*args, fusion_mode='semantic_graph' if self.use_dino_relation else 'rgb_graph', **kwargs)
        if self.use_channel_conversion:
            with torch.random.fork_rng(devices=[]):
                self.fusion_bot = MessageAdapter(128)
                self.fusion_dec = MessageAdapter(64)
        # Shared receiver=0.5, same gain and spatial distance in all four cells.
        # Physical ISP normalization remains active in the native backbone.
        self.fusion_mode = 'semantic_graph'

    def _graph(self, x):
        if self.use_dino_relation:
            return super()._graph(x)
        # Remove only DINO cosine: no RGB substitute or additional learned metric.
        return (-2.0*self.graph_distance[None]).expand(x.shape[0], -1, -1)
