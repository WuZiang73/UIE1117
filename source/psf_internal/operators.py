import torch
import torch.nn.functional as F

from psf_recovery.kernels import kernel_bank
from psf_recovery.operators import _conv3


def to_linear(x):
    return x.clamp(0, 1).pow(2.2)


def from_linear(x):
    return x.clamp(0, 1).pow(1.0 / 2.2)


def rgb_to_state(x):
    return F.pixel_unshuffle(x, 2)


def state_to_rgb(z):
    return F.pixel_shuffle(z, 2)


def blur_rgb(u, kernels):
    return _conv3(u, kernels)


def make_kernels(radius=8, device=None):
    k = kernel_bank((0.0, 0.6, 1.2, 2.4), radius=radius)
    return k.to(device) if device is not None else k


def render_state(z, kernels, t, pi, b):
    u = state_to_rgb(z)
    blurred = blur_rgb(t * u, kernels)
    y = (blurred * pi[:, :, :, None, None]).sum(2) + b
    return rgb_to_state(y)


def state_adjoint(z, kernels, t, pi, b, v):
    with torch.enable_grad():
        zz = z if z.requires_grad else z.detach().requires_grad_(True)
        y = render_state(zz, kernels, t, pi, b)
        g = torch.autograd.grad(y, zz, v, create_graph=True)[0]
    return g


def highpass_state(z):
    u = state_to_rgb(z)
    k = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=u.dtype, device=u.device)
    k = k / k.sum()
    k = k[None, None].repeat(3, 1, 1, 1)
    blur = F.conv2d(F.pad(u, (1, 1, 1, 1), mode="reflect"), k, groups=3)
    return rgb_to_state(u - blur)


def highpass_adjoint(z, v):
    with torch.enable_grad():
        zz = z if z.requires_grad else z.detach().requires_grad_(True)
        y = highpass_state(zz)
        g = torch.autograd.grad(y, zz, v, create_graph=True)[0]
    return g
