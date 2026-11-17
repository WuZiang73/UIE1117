"""Current-weight Poisson/ISP variance propagation in the first C0 HIN block.

The early convolutions are genuinely affine. Their fused contrast kernel
therefore gives exact conditional variance for a supplied independent input
variance map, away from the boundary. The UIEB variance estimate itself is an
uncalibrated shot/ISP proxy; no Gaussian/Poisson noise is added to training.
"""
import torch
from torch import nn
from torch.nn import functional as F
from baseline_model import BaselineUNet
from optical_controlled import linearize

MODES = ('exact', 'diagonal', 'homogeneous')
WINDOW = 7
VALID_RIM = 8
ENERGY_FLOOR = 1e-8
ROOT_EPSILON = 1e-4
MAD_NORMALIZER = .6744897501960817


def reflect_mean(x):
    return F.avg_pool2d(F.pad(x, (3, 3, 3, 3), mode='reflect'), WINDOW, 1)


def observed_shot_variance(x):
    unit = x.clamp(0, 1)
    intensity = linearize(unit)
    root = (intensity+ROOT_EPSILON).sqrt()
    haar = x.new_tensor([[1., -1.], [-1., 1.]])[None, None]/2
    coefficients = F.conv2d(root, haar.expand(3, 1, 2, 2), stride=2, groups=3).flatten(2)
    centered = coefficients-coefficients.median(2, keepdim=True).values
    root_variance = (centered.abs().median(2).values/MAD_NORMALIZER).square()
    photon_scale = 4*root_variance[:, :, None, None]
    derivative = torch.where(unit <= .04045, torch.full_like(unit, 1/12.92),
        (2.4/1.055)*((unit+.055)/1.055).pow(1.4))
    # Delta-method variance in observed RGB; read noise and cross-color/ISP
    # correlations are unknown, so this is not a calibrated noise model.
    return photon_scale*intensity/derivative.square(), photon_scale


def fuse_kernels(w0, dw, pw):
    # Composition of two PyTorch correlations is full convolution of their
    # stored kernels; flip the second kernel for conv2d to perform that sum.
    composed = F.conv2d(w0.permute(1, 0, 2, 3), dw.flip((-2, -1)),
                        padding=2, groups=w0.shape[0]).permute(1, 0, 2, 3)
    return torch.einsum('oi,icuv->ocuv', pw, composed)


def fused_early_kernel(first, block):
    return fuse_kernels(first.weight, block.conv_1.conv1.weight,
                        block.conv_1.conv2.weight[..., 0, 0])


def contrast_kernel(kernel):
    out_channels, in_channels = kernel.shape[:2]
    box = kernel.new_full((1, 1, WINDOW, WINDOW), 1/(WINDOW*WINDOW))
    average = F.conv2d(kernel.reshape(-1, 1, 5, 5), box, padding=6)
    return F.pad(kernel, (3, 3, 3, 3))-average.reshape(out_channels, in_channels, 11, 11)


def propagated_contrast_variance(variance, first, block, mode):
    channels = block.norm.num_features
    if mode in ('exact', 'homogeneous'):
        if mode == 'homogeneous':
            variance = variance.mean((-2, -1), keepdim=True).expand_as(variance)
        kernel = contrast_kernel(fused_early_kernel(first, block)[:channels])
        valid = F.conv2d(variance, kernel.square())
        return F.pad(valid, (5, 5, 5, 5))
    if mode != 'diagonal':
        raise ValueError(mode)
    # Matched approximate control: retain only diagonal covariances after
    # each affine convolution, then after local centering. Overlapping paths
    # and feature-channel correlations are deliberately omitted.
    v0 = F.conv2d(variance, first.weight.square(), padding=1)
    v1 = F.conv2d(v0, block.conv_1.conv1.weight.square(), padding=1, groups=v0.shape[1])
    vf = F.conv2d(v1, block.conv_1.conv2.weight[:channels].square())
    highpass = vf.new_full((1, 1, WINDOW, WINDOW), -1/(WINDOW*WINDOW))
    highpass[..., 3, 3] += 1
    valid = F.conv2d(vf, highpass.square().expand(channels, 1, WINDOW, WINDOW), groups=channels)
    return F.pad(valid, (3, 3, 3, 3))


def local_wiener(features, contrast_variance):
    mean = reflect_mean(features)
    residual = features-mean
    energy = reflect_mean(residual.square())
    noise_energy = reflect_mean(contrast_variance)
    shrinkage = (1-noise_energy/(energy+ENERGY_FLOOR)).clamp(0, 1)
    # Both contrast and energy supports must lie inside the affine kernel's
    # valid region. Preserve the outer eight pixels instead of asserting an
    # invalid independent-noise formula across reflection/zero-pad boundaries.
    correction = (1-shrinkage)*residual
    core = correction[..., VALID_RIM:-VALID_RIM, VALID_RIM:-VALID_RIM]
    correction = F.pad(core, (VALID_RIM,)*4)
    return features-correction, shrinkage


class NoisePropagationUNet(BaselineUNet):
    def __init__(self, *args, noise_propagation_mode='exact', **kwargs):
        kwargs['semantic'] = None
        super().__init__(*args, **kwargs)
        assert self.variant == 'M2v2R' and noise_propagation_mode in MODES
        self.noise_propagation_mode = noise_propagation_mode
        self.intervention = 'normal'
        self.observer = None

    def forward(self, x, return_pair=False):
        if return_pair:
            raise ValueError('No output correction pair')
        assert min(x.shape[-2:]) > 2*VALID_RIM
        context = self.dark_context_v2(x)
        initial = self.first(x)
        block = self.encoder1
        affine = block.conv_1(initial)
        normal, bypass = affine.chunk(2, 1)
        mode = self.intervention if self.intervention in MODES else self.noise_propagation_mode
        if self.intervention == 'disabled_filter':
            denoised = normal
        else:
            variance, photon_scale = observed_shot_variance(x)
            if self.intervention == 'shuffled_variance':
                variance = variance.roll(1, 0)
            noise = propagated_contrast_variance(variance, self.first, block, mode)
            denoised, shrinkage = local_wiener(normal, noise)
            if self.observer is not None:
                self.observer(normal, denoised, shrinkage, noise, photon_scale)
        out = torch.cat((block.norm(denoised), bypass), 1)
        out = block.relu_1(out)
        out = block.relu_2(block.conv_2(out))
        first = out+block.identity(initial)
        second = self.encoder2(self.down1(first))
        bottleneck = self.bottleneck(self.down2(second))
        out = self.decoder1(self.up1(bottleneck)+second)
        out = self.dark_dec1_v2(out, context)
        out = self.decoder2(self.up2(out)+first)
        return x+self.out(out)
