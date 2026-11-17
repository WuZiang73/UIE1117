"""First-HIN normalization from the actual encoder's photon-noise covariance.

Analytic covariance, ZCA and Newton--Schulz are established methods. The
controlled hypothesis is the current native RGB-to-feature kernel plus observed
per-RGB flux, as a full covariance conditioner in C0, without an added head.
"""
import torch
from baseline_model import BaselineUNet
from optical_controlled import linearize
from optical_noise_propagation import fused_early_kernel

MODES = ('full', 'diagonal', 'empirical', 'linear_in', 'srgb_full', 'centered', 'uniform_budget')
RIDGE_RELATIVE = .01
RIDGE_ABSOLUTE = 1e-5
INVERSE_ROOT_STEPS = 15


def photon_covariance(kernel, budget):
    # K=[O,RGB,5,5], q=[N,RGB]. Cross-feature noise paths share RGB samples.
    return torch.einsum('ocuv,pcuv,bc->bop', kernel, kernel, budget)


def regularize(covariance):
    size = covariance.shape[-1]
    identity = torch.eye(size, device=covariance.device, dtype=covariance.dtype)[None]
    trace = covariance.diagonal(dim1=-2, dim2=-1).sum(-1)
    ridge = RIDGE_RELATIVE*trace/size+RIDGE_ABSOLUTE
    return covariance+ridge[:, None, None]*identity, ridge


def inverse_root(matrix):
    size = matrix.shape[-1]
    identity = torch.eye(size, device=matrix.device, dtype=matrix.dtype)[None]
    trace = matrix.diagonal(dim1=-2, dim2=-1).sum(-1)[:, None, None]
    y = matrix/trace
    z = identity.expand_as(matrix)
    for _ in range(INVERSE_ROOT_STEPS):
        update = .5*(3*identity-z@y)
        y = y@update
        z = update@z
        y = .5*(y+y.transpose(-1, -2))
        z = .5*(z+z.transpose(-1, -2))
    return z/trace.sqrt()


class PhotonWhiteningUNet(BaselineUNet):
    def __init__(self, *args, photon_whitening_mode='full', **kwargs):
        kwargs['semantic'] = None
        super().__init__(*args, **kwargs)
        assert self.variant == 'M2v2R' and photon_whitening_mode in MODES
        self.photon_whitening_mode = photon_whitening_mode
        self.intervention = 'normal'
        self.observer = None

    def forward(self, x, return_pair=False):
        if return_pair:
            raise ValueError('No output correction pair')
        mode = self.intervention if self.intervention in MODES else self.photon_whitening_mode
        observed = x if mode == 'srgb_full' else linearize(x)
        if self.intervention == 'disabled_all':
            observed = x
        context = self.dark_context_v2(x)
        initial = self.first(observed)
        block = self.encoder1
        f, bypass = block.conv_1(initial).chunk(2, 1)
        if mode == 'linear_in' or self.intervention in ('disabled_all', 'original_in'):
            normalized = block.norm(f)
        else:
            budget = observed.mean((-2, -1))
            if self.intervention == 'shuffled_budget':
                budget = budget.roll(1, 0)
            if mode == 'uniform_budget':
                budget = budget.mean(1, keepdim=True).expand_as(budget)
            if mode == 'empirical':
                centered = f-f.mean((-2, -1), keepdim=True)
                values = centered.flatten(2)
                covariance = values@values.transpose(1, 2)/values.shape[-1]
            else:
                kernel = fused_early_kernel(self.first, block)[:f.shape[1]]
                covariance = photon_covariance(kernel, budget)
            matrix, ridge = regularize(covariance)
            if mode == 'diagonal':
                gain = torch.diag_embed(matrix.diagonal(dim1=-2, dim2=-1).rsqrt())
            else:
                gain = inverse_root(matrix)
            values = f-f.mean((-2, -1), keepdim=True) if mode == 'centered' else f
            transformed = (gain@values.flatten(2)).reshape_as(f)
            normalized = transformed*block.norm.weight[None, :, None, None]+block.norm.bias[None, :, None, None]
            if self.observer is not None:
                self.observer(f, normalized, covariance, matrix, gain, ridge, budget)
        out = torch.cat((normalized, bypass), 1)
        out = block.relu_1(out)
        out = block.relu_2(block.conv_2(out))
        first = out+block.identity(initial)
        second = self.encoder2(self.down1(first))
        bottleneck = self.bottleneck(self.down2(second))
        out = self.decoder1(self.up1(bottleneck)+second)
        out = self.dark_dec1_v2(out, context)
        out = self.decoder2(self.up2(out)+first)
        return x+self.out(out)
