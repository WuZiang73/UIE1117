"""Optical architecture experiments. No extra loss, teacher, or pretraining.

The RGB state is an effective imaging proxy; learned parameters are not claimed
to be calibrated water properties. Existing models/checkpoints are untouched.
"""
import torch
from torch import nn
from torch.nn import functional as F

from baseline_model import BaselineUNet
from psf_recovery.kernels import kernel_bank


def linearize(x):
    x = x.clamp(0, 1)
    return torch.where(x <= 0.04045, x / 12.92,
                       ((x + 0.055) / 1.055).pow(2.4))


def reflect_index(n, radius, device):
    a = torch.arange(-radius, n + radius, device=device)
    a = a.remainder(2 * (n - 1))
    return torch.minimum(a, 2 * (n - 1) - a)


def reflect_conv(x, weight):
    r = weight.shape[-1] // 2
    return F.conv2d(F.pad(x, (r, r, r, r), mode='reflect'),
                    weight, groups=x.shape[1])


def reflect_conv_adjoint(v, weight):
    """Exact transpose, including the reflection extension (not same conv)."""
    r = weight.shape[-1] // 2
    n, c, h, w = v.shape
    ext = F.conv_transpose2d(v, weight, groups=c)
    cols = ext.new_zeros(n, c, h + 2*r, w)
    cols = cols.index_add(-1, reflect_index(w, r, v.device), ext)
    out = ext.new_zeros(n, c, h, w)
    return out.index_add(-2, reflect_index(h, r, v.device), cols)


class OpticalParameters(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(3, 24, 3, 4, 1), nn.GELU(),
                                 nn.Conv2d(24, 24, 3, 4, 1), nn.GELU(),
                                 nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                 nn.Linear(24, 18))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        with torch.no_grad():
            self.net[-1].bias[:3] = 2
            self.net[-1].bias[3:15].view(3, 4)[:, 0] = 3
            self.net[-1].bias[15:] = -4

    def forward(self, x):
        a = self.net(x)
        t = (0.05 + 0.95*a[:, :3].sigmoid())[:, :, None, None]
        pi = a[:, 3:15].view(-1, 3, 4).softmax(-1)
        b = a[:, 15:].sigmoid()[:, :, None, None]
        return t, pi, b


class OpticalBlock(nn.Module):
    def __init__(self, channels=64, mode='phase'):
        super().__init__()
        self.mode = mode
        self.cal = OpticalParameters()
        self.rin = nn.Sequential(nn.Conv2d(channels*2, 32, 1), nn.GELU(),
                                 nn.Conv2d(32, 12, 1))
        self.rout = nn.Conv2d(12, channels, 1)
        nn.init.zeros_(self.rout.weight)
        nn.init.zeros_(self.rout.bias)
        self.register_buffer('kernels', kernel_bank(radius=8))
        q = torch.tensor([[1., 2., 1.], [2., 4., 2.], [1., 2., 1.]])/16
        self.register_buffer('qkernel', q[None, None].repeat(3, 1, 1, 1))
        if mode == 'band':
            # Same feature widths: identity starts aligned decoder feature map.
            self.align = nn.Conv2d(channels, channels, 1, bias=False)
            nn.init.dirac_(self.align.weight)
            self.scale = nn.Parameter(torch.zeros(()))
        if mode == 'adapter':
            self.adapter = nn.Conv2d(channels*2, channels, 1)
            nn.init.zeros_(self.adapter.weight)
            nn.init.zeros_(self.adapter.bias)
        self.last_diag = {}

    def blur(self, u, pi):
        # Three color groups, each containing M kernels in the same order.
        c = u.shape[1]
        weight = self.kernels[:, None].repeat(1, c, 1, 1)
        weight = weight.permute(1, 0, 2, 3).reshape(c*4, 1, 17, 17)
        v = F.conv2d(F.pad(u, (8, 8, 8, 8), mode='reflect'), weight, groups=c)
        return (v.view(u.shape[0], c, 4, *u.shape[-2:]) *
                pi[..., None, None]).sum(2)

    def blur_adjoint(self, v, pi):
        # Sum of exact reflection-convolution transposes.
        return sum(reflect_conv_adjoint(v*pi[:, :, m, None, None],
                                       k[None, None].repeat(3, 1, 1, 1))
                   for m, k in enumerate(self.kernels))

    def highpass(self, u):
        return u - reflect_conv(u, self.qkernel)

    def highpass_adjoint(self, u):
        return u - reflect_conv_adjoint(u, self.qkernel)

    def forward(self, x, e2, q1):
        if self.mode == 'adapter':
            return self.adapter(torch.cat((e2, q1), 1)), {}
        t, pi, b = self.cal(x)
        if self.mode == 'band':
            # Proxy conditioning of arbitrary features, not an exact feature PSF.
            # Frequencies are in original RGB pixels, only used as band summaries.
            freq = x.new_tensor([0.08, 0.20])
            sigma = x.new_tensor([0., .6, 1.2, 2.4])
            response = torch.exp(-2*torch.pi**2 *
                                 sigma[:, None].square()*freq[None].square())
            h = (pi[..., None]*response).sum(2)*t.flatten(1)[..., None]
            availability = (h.square()/(h.square()+0.01)).mean(1)
            diff = self.align(q1)-e2
            low = F.avg_pool2d(F.pad(diff, (2, 2, 2, 2), mode='reflect'), 5, 1)
            very_low = F.avg_pool2d(F.pad(diff, (4, 4, 4, 4), mode='reflect'), 9, 1)
            bands = (low-very_low, diff-low)
            delta = sum((1-availability[:, j, None, None, None])*band
                        for j, band in enumerate(bands))
            delta = .25*self.scale.tanh()*delta
        else:
            obs = linearize(x)
            z0 = F.pixel_unshuffle(obs, 2) + .1*torch.tanh(
                self.rin(torch.cat((e2, q1), 1)))
            u0 = F.pixel_shuffle(z0, 2)
            if self.mode == 'attenuation':
                # Isolate color attenuation/backscatter from spatial scattering.
                residual = obs - (t*u0+b)
                grad = t*residual
            else:
                if self.mode == 'delta':
                    pi = torch.zeros_like(pi)
                    pi[:, :, 0] = 1
                residual = obs-(self.blur(t*u0, pi)+b)
                grad = t*self.blur_adjoint(
                    self.highpass_adjoint(self.highpass(residual)), pi)
            delta = self.rout(F.pixel_unshuffle(.5*grad, 2))
        self.last_diag = {'t': float(t.detach().mean()),
                          'non_delta_mass': float((1-pi[:, :, 0]).detach().mean()),
                          'delta_abs': float(delta.detach().abs().mean())}
        return delta, self.last_diag


class ControlledUNet(BaselineUNet):
    def __init__(self, *args, optical_mode='phase', **kwargs):
        # Reuse the original forward and skip injection. Shared modules remain
        # exactly the C0 modules, loaded from the common RANDOM initialization.
        kwargs['semantic'] = None
        super().__init__(*args, **kwargs)
        if optical_mode != 'c0':
            self.internal = OpticalBlock(64, optical_mode)
