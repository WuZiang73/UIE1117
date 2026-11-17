"""Boundary-exact regional noise covariance in the original display encoder.

Analytic covariance, matrix normalization and spatial interpolation are known
tools. The hypothesis is a camera-proxy noise FIELD through the current native
affine prefix, with matched spatial, diagonal and domain controls in C0.
"""
import torch
from torch import nn
from torch.nn import functional as F
from functools import lru_cache
from baseline_model import BaselineUNet
from optical_controlled import linearize
from optical_isp_covariance import inverse_response_derivative
from optical_noise_propagation import fuse_kernels
from optical_photon_whitening import regularize, inverse_root

MODES = ('regional', 'global', 'diagonal', 'display_matched', 'uniform_rgb', 'mirrored')
GRID = 8


def boundary_kernels(w0, dw, pw):
    kernels = []
    for category in range(9):
        vertical, horizontal = divmod(category, 3)
        mask = dw.new_ones(3, 3)
        if vertical == 1:
            mask[0, :] = 0
        elif vertical == 2:
            mask[2, :] = 0
        if horizontal == 1:
            mask[:, 0] = 0
        elif horizontal == 2:
            mask[:, 2] = 0
        kernels.append(fuse_kernels(w0, dw*mask[None, None], pw))
    return torch.stack(kernels).flatten(-2)


@lru_cache(maxsize=8)
def boundary_geometry(height, width, grid, device, dtype):
    tile_h, tile_w = height//grid, width//grid
    yy, xx = torch.meshgrid(torch.arange(height, device=device),
        torch.arange(width, device=device), indexing='ij')
    vertical = torch.where(yy==0, 1, torch.where(yy==height-1, 2, 0))
    horizontal = torch.where(xx==0, 1, torch.where(xx==width-1, 2, 0))
    category = 3*vertical+horizontal
    edge = category != 0
    y, x = yy[edge], xx[edge]
    offsets = torch.tensor([(dy, dx) for dy in range(-2, 3) for dx in range(-2, 3)], device=device)
    index = (y[:, None]+offsets[None, :, 0]+2)*(width+4)+x[:, None]+offsets[None, :, 1]+2
    bins = (category[edge]-1)*(grid*grid)+(y//tile_h)*grid+x//tile_w
    # Fixed routing followed by GEMM avoids atomic scatter accumulation.
    routing = F.one_hot(bins, num_classes=8*grid*grid).to(dtype)
    return index, routing


def variance_statistics(variance, grid=GRID):
    batch, colors, height, width = variance.shape
    assert height%grid == width%grid == 0 and min(height, width) >= 4
    tile_h, tile_w = height//grid, width//grid
    padded = F.pad(variance, (2, 2, 2, 2))
    pooled = F.avg_pool2d(padded, (tile_h, tile_w), stride=1)
    offsets = [(dy, dx) for dy in range(-2, 3) for dx in range(-2, 3)]
    full = torch.stack([pooled[:, :, 2+dy:2+dy+height:tile_h,
        2+dx:2+dx+width:tile_w] for dy, dx in offsets], 2)
    index, routing = boundary_geometry(height, width, grid, variance.device, variance.dtype)
    values = padded.flatten(2).index_select(2, index.flatten()).reshape(batch, colors, -1, 25).transpose(2, 3)
    accumulated = (values.reshape(-1, index.shape[0])@routing).reshape(batch, colors, 25, 8*grid*grid)
    edges = accumulated.reshape(batch, colors, 25, 8, grid, grid).permute(0, 3, 1, 2, 4, 5)/(tile_h*tile_w)
    interior = (full-edges.sum(1)).clamp_min(0)
    return torch.cat((interior[:, None], edges), 1)


def covariance_from_statistics(statistics, kernels):
    return torch.einsum('docv,dpcv,bdcvhw->bhwop', kernels, kernels, statistics)


def apply_matrix_field(features, matrices):
    """Exact align_corners=False bilinear matrix field without dense HWC^2."""
    batch, channels, height, width = features.shape
    grid_h, grid_w = matrices.shape[1:3]
    assert height%grid_h == width%grid_w == 0
    tile_h, tile_w = height//grid_h, width//grid_w
    tiled = features.reshape(batch, channels, grid_h, tile_h, grid_w, tile_w).permute(0, 2, 4, 1, 3, 5).reshape(batch, grid_h, grid_w, channels, tile_h*tile_w)
    ty = (torch.arange(tile_h, device=features.device, dtype=features.dtype)+.5)/tile_h-.5
    tx = (torch.arange(tile_w, device=features.device, dtype=features.dtype)+.5)/tile_w-.5
    wy = {-1: (-ty).clamp_min(0), 0: 1-ty.abs(), 1: ty.clamp_min(0)}
    wx = {-1: (-tx).clamp_min(0), 0: 1-tx.abs(), 1: tx.clamp_min(0)}
    result = torch.zeros_like(tiled)
    for dy in (-1, 0, 1):
        iy = (torch.arange(grid_h, device=features.device)+dy).clamp(0, grid_h-1)
        for dx in (-1, 0, 1):
            ix = (torch.arange(grid_w, device=features.device)+dx).clamp(0, grid_w-1)
            shifted = matrices.index_select(1, iy).index_select(2, ix)
            weights = (wy[dy][:, None]*wx[dx][None]).flatten()
            result = result+(shifted@tiled)*weights[None, None, None, None]
    return result.reshape(batch, grid_h, grid_w, channels, tile_h, tile_w).permute(0, 3, 1, 4, 2, 5).reshape_as(features)


class RegionalISPUNet(BaselineUNet):
    def __init__(self, *args, regional_isp_mode='regional', **kwargs):
        kwargs['semantic'] = None
        super().__init__(*args, **kwargs)
        assert self.variant == 'M2v2R' and regional_isp_mode in MODES
        self.regional_isp_mode = regional_isp_mode
        self.intervention = 'normal'
        self.observer = None

    def forward(self, x, return_pair=False):
        if return_pair:
            raise ValueError('No output correction pair')
        mode = self.intervention if self.intervention in MODES else self.regional_isp_mode
        context = self.dark_context_v2(x)
        initial = self.first(x)
        block = self.encoder1
        f, bypass = block.conv_1(initial).chunk(2, 1)
        if self.intervention == 'disabled':
            normalized = block.norm(f)
        else:
            unit = x.clamp(0, 1)
            variance = linearize(unit)/inverse_response_derivative(unit).square()
            if mode == 'display_matched':
                scale = variance.mean((1, 2, 3), keepdim=True)/unit.mean((1, 2, 3), keepdim=True).clamp_min(1e-12)
                variance = unit*scale
            elif mode == 'uniform_rgb':
                variance = variance.mean(1, keepdim=True).expand_as(variance)
            if self.intervention == 'shuffled_variance':
                variance = variance.roll(1, 0)
            statistics = variance_statistics(variance)
            kernels = boundary_kernels(self.first.weight, block.conv_1.conv1.weight,
                block.conv_1.conv2.weight[:f.shape[1], :, 0, 0])
            covariance = covariance_from_statistics(statistics, kernels)
            if mode == 'global':
                covariance = covariance.mean((1, 2), keepdim=True).expand_as(covariance)
            batch, gh, gw, channels, _ = covariance.shape
            matrix, ridge = regularize(covariance.reshape(-1, channels, channels))
            if mode == 'diagonal':
                gain = torch.diag_embed(matrix.diagonal(dim1=-2, dim2=-1).rsqrt())
            else:
                gain = inverse_root(matrix)
            gain = gain.reshape(batch, gh, gw, channels, channels)
            if mode == 'mirrored':
                gain = gain.flip(2)
            transformed = apply_matrix_field(f, gain)
            normalized = transformed*block.norm.weight[None, :, None, None]+block.norm.bias[None, :, None, None]
            if self.observer is not None:
                self.observer(f, normalized, covariance, gain, ridge, variance)
        out = block.relu_1(torch.cat((normalized, bypass), 1))
        out = block.relu_2(block.conv_2(out))
        first = block.drop_path(out)+block.identity(initial)
        second = self.encoder2(self.down1(first))
        bottleneck = self.bottleneck(self.down2(second))
        out = self.decoder1(self.up1(bottleneck)+second)
        out = self.dark_dec1_v2(out, context)
        out = self.decoder2(self.up2(out)+first)
        return x+self.out(out)
