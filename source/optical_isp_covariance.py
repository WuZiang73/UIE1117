"""Native display-RGB encoder conditioned by a propagated noise proxy.

The delta method and ISP noise propagation are established. This isolates the
covariance budget from changing the backbone input domain, which the previous
linear-input versus display-input family did not fully separate.
"""
import torch
from baseline_model import BaselineUNet
from optical_controlled import linearize
from optical_noise_propagation import fused_early_kernel
from optical_photon_whitening import photon_covariance, regularize, inverse_root

MODES = ('jacobian', 'diagonal_jacobian', 'linear_budget', 'uniform_jacobian', 'read_matched', 'display_matched')


def inverse_response_derivative(x):
    x = x.clamp(0, 1)
    return torch.where(x <= .04045, torch.full_like(x, 1/12.92),
        (2.4/1.055)*((x+.055)/1.055).pow(1.4))


def covariance_budget(x, mode='jacobian'):
    assert mode in MODES or mode == 'parent_display'
    unit = x.clamp(0, 1)
    intensity = linearize(unit)
    if mode == 'linear_budget':
        return intensity.mean((-2, -1))
    if mode == 'parent_display':
        return unit.mean((-2, -1))
    derivative = inverse_response_derivative(unit)
    # For raw variance a*I, display variance is a*I/(dI/dx)^2 to first order.
    # Average the pointwise propagated variance, not a ratio of global means.
    budget = (intensity/derivative.square()).mean((-2, -1))
    if mode == 'uniform_jacobian':
        return budget.mean(1, keepdim=True).expand_as(budget)
    if mode == 'read_matched':
        read = derivative.square().reciprocal().mean((-2, -1))
        return read*(budget.mean(1, keepdim=True)/read.mean(1, keepdim=True).clamp_min(1e-12))
    if mode == 'display_matched':
        display = unit.mean((-2, -1))
        return display*(budget.mean(1, keepdim=True)/display.mean(1, keepdim=True).clamp_min(1e-12))
    return budget


class ISPCovarianceUNet(BaselineUNet):
    def __init__(self, *args, isp_covariance_mode='jacobian', **kwargs):
        kwargs['semantic'] = None
        super().__init__(*args, **kwargs)
        assert self.variant == 'M2v2R' and isp_covariance_mode in MODES
        self.isp_covariance_mode = isp_covariance_mode
        self.intervention = 'normal'
        self.observer = None

    def forward(self, x, return_pair=False):
        if return_pair:
            raise ValueError('No output correction pair')
        mode = self.intervention if self.intervention in (*MODES, 'parent_display') else self.isp_covariance_mode
        context = self.dark_context_v2(x)
        # Original display input throughout all six controls.
        initial = self.first(x)
        block = self.encoder1
        f, bypass = block.conv_1(initial).chunk(2, 1)
        if self.intervention == 'disabled':
            normalized = block.norm(f)
        else:
            budget = covariance_budget(x, mode)
            if self.intervention == 'shuffled_budget':
                budget = budget.roll(1, 0)
            kernel = fused_early_kernel(self.first, block)[:f.shape[1]]
            covariance = photon_covariance(kernel, budget)
            matrix, ridge = regularize(covariance)
            if mode == 'diagonal_jacobian':
                gain = torch.diag_embed(matrix.diagonal(dim1=-2, dim2=-1).rsqrt())
            else:
                gain = inverse_root(matrix)
            transformed = (gain@f.flatten(2)).reshape_as(f)
            normalized = transformed*block.norm.weight[None, :, None, None]+block.norm.bias[None, :, None, None]
            if self.observer is not None:
                self.observer(f, normalized, covariance, matrix, gain, ridge, budget)
        out = block.relu_1(torch.cat((normalized, bypass), 1))
        out = block.relu_2(block.conv_2(out))
        first = out+block.identity(initial)
        second = self.encoder2(self.down1(first))
        bottleneck = self.bottleneck(self.down2(second))
        out = self.decoder1(self.up1(bottleneck)+second)
        out = self.dark_dec1_v2(out, context)
        out = self.decoder2(self.up2(out)+first)
        return x+self.out(out)
