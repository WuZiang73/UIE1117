"""DINO relations checked against native affine patch-mean noise units."""
import torch
from torch.nn import functional as F
from precision_graph_core import DinoISPUNet as OriginalPrecisionGraph
from message_adapter_core import MessageAdapter


def mean_kernels(w0, dw, pw, tile_h, tile_w):
    """Jacobian kernels of boundary categories' bias-free native patch means."""
    seed = pw[:, :, 0, 0, None, None].expand(-1, -1, tile_h, tile_w)/(tile_h*tile_w)
    reverse_dw = F.conv_transpose2d(seed, dw, groups=dw.shape[0])
    kernels = []
    for category in range(9):
        vertical, horizontal = divmod(category, 3)
        current = reverse_dw.clone()
        if vertical == 1:
            current[..., 0, :] = 0
        elif vertical == 2:
            current[..., -1, :] = 0
        if horizontal == 1:
            current[..., :, 0] = 0
        elif horizontal == 2:
            current[..., :, -1] = 0
        kernels.append(F.conv_transpose2d(current, w0))
    return torch.stack(kernels)


def patch_measurements(x, variance, w0, dw, pw, grid=16):
    """Exact marginals for an independent-pixel RGB noise proxy."""
    batch, _, height, width = x.shape
    assert grid > 1 and height%grid == width%grid == 0
    th, tw = height//grid, width//grid
    kernels = mean_kernels(w0, dw, pw, th, tw)
    _, channels, colors, kh, kw = kernels.shape
    weights = kernels.reshape(9*channels, colors, kh, kw)
    mu = F.conv2d(x, weights, stride=(th,tw), padding=2).reshape(batch,9,channels,grid,grid)
    noise = F.conv2d(variance, weights.square(), stride=(th,tw), padding=2).reshape_as(mu)
    yy, xx = torch.meshgrid(torch.arange(grid,device=x.device),torch.arange(grid,device=x.device),indexing='ij')
    vertical = torch.where(yy==0,1,torch.where(yy==grid-1,2,0))
    horizontal = torch.where(xx==0,1,torch.where(xx==grid-1,2,0))
    select = F.one_hot(3*vertical+horizontal,9).permute(2,0,1).to(x.dtype)[None,:,None]
    return (mu*select).sum(1), (noise*select).sum(1).clamp_min(0)


def appearance_penalty(mu, variance, uniform=False):
    values = mu.flatten(2).transpose(1,2)
    noise = variance.flatten(2).transpose(1,2)
    if uniform:
        noise = noise.mean(1,keepdim=True).expand_as(noise)
    denom = noise[:,:,None,:]+noise[:,None,:,:]
    denom = (denom+1e-6*denom.mean(-1,keepdim=True)).clamp_min(1e-12)
    distance = ((values[:,:,None,:]-values[:,None,:,:]).square()/denom).mean(-1)
    return 0.5*torch.log1p(distance)


class DinoISPUNet(OriginalPrecisionGraph):
    MODES = ('patch_adapter', 'uniform_adapter', 'plain_adapter', 'rgb_patch_adapter', 'precision_graph', 'rgb_graph')

    def __init__(self, *args, fusion_mode='patch_adapter', **kwargs):
        assert fusion_mode in ('patch_adapter','uniform_adapter','plain_adapter','rgb_patch_adapter')
        self.compat_mode = fusion_mode
        super().__init__(*args,fusion_mode='rgb_graph' if fusion_mode=='rgb_patch_adapter' else 'precision_graph',**kwargs)
        with torch.random.fork_rng(devices=[]):
            self.fusion_bot = MessageAdapter(128)
            self.fusion_dec = MessageAdapter(64)
        self.fusion_mode = 'semantic_graph'
        self._compat_input = None
        self._appearance = None

    def _graph(self, x):
        if self.dino_teacher is not None:
            return super()._graph(x)
        patch = F.adaptive_avg_pool2d(x,(16,16))
        tokens = F.normalize(patch.flatten(2).transpose(1,2),dim=-1,eps=1e-6)
        return (tokens@tokens.transpose(1,2))/0.15-2*self.graph_distance[None]

    def _observe_noise(self, f, normalized, covariance, gain, ridge, variance):
        super()._observe_noise(f,normalized,covariance,gain,ridge,variance)
        if self.compat_mode == 'plain_adapter':
            return
        with torch.no_grad():
            x = self._compat_input
            grid = 16 if x.shape[-2]%16 == x.shape[-1]%16 == 0 else 8
            mu, noise = patch_measurements(x,variance,self.first.weight,
                self.encoder1.conv_1.conv1.weight,self.encoder1.conv_1.conv2.weight[:f.shape[1]],grid)
            if grid != 16:
                # Only rectangular smoke inputs: the real256x256 recipe uses
                # exact16x16 noninterpolated patch measurements.
                mu = F.interpolate(mu,size=(16,16),mode='bilinear',align_corners=False)
                noise = F.interpolate(noise,size=(16,16),mode='bilinear',align_corners=False)
            self._appearance = appearance_penalty(mu,noise,self.compat_mode=='uniform_adapter')

    def _apply_fusion(self, feature, module):
        if not self.fusion_enabled:
            return feature
        if self.compat_mode == 'plain_adapter':
            return super()._apply_fusion(feature,module)
        old = self._condition
        self._condition = old-self._appearance
        try:
            return super()._apply_fusion(feature,module)
        finally:
            self._condition = old

    def forward(self, x, return_pair=False):
        self._compat_input = x
        try:
            return super().forward(x,return_pair=return_pair)
        finally:
            self._compat_input = None
            self._appearance = None
