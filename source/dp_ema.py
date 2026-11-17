"""Training-only residual stochastic depth and a non-feedback EMA shadow."""
from copy import deepcopy
import torch
from torch import nn


class SampleDropPath(nn.Module):
    def __init__(self, probability=0.0):
        super().__init__()
        self.probability = float(probability)
        self.last_mask = None

    def forward(self, value):
        if not self.training or self.probability == 0.0:
            if self.training:
                self.last_mask = value.new_ones(value.shape[0])
            return value
        keep = 1.0 - self.probability
        mask = (torch.rand(value.shape[0], 1, 1, 1, device=value.device) < keep).to(value.dtype)
        self.last_mask = mask.detach().flatten()
        return value * (mask / keep)


def configure_drop_path(model, rates):
    names = ['encoder1', 'encoder2', 'bottleneck', 'decoder1', 'decoder2']
    for name, probability in zip(names, rates):
        block = getattr(model, name)
        assert isinstance(block.drop_path, SampleDropPath)
        block.drop_path.probability = float(probability)
    return model


def masks(model):
    return torch.cat([getattr(model, name).drop_path.last_mask for name in
        ['encoder1', 'encoder2', 'bottleneck', 'decoder1', 'decoder2']])


class ModelEMA:
    def __init__(self, model, decay_max=.999):
        self.model = deepcopy(model).eval()
        self.model.requires_grad_(False)
        self.decay_max = float(decay_max)
        self.num_updates = 0

    @torch.no_grad()
    def update(self, model):
        self.num_updates += 1
        decay = min(self.decay_max, (1 + self.num_updates) / (10 + self.num_updates))
        source = dict(model.named_parameters())
        for name, parameter in self.model.named_parameters():
            parameter.mul_(decay).add_(source[name].detach(), alpha=1 - decay)
        # Static optical priors and rho must not drift through repeated floating-point averaging.
        buffers = dict(model.named_buffers())
        for name, buffer in self.model.named_buffers():
            buffer.copy_(buffers[name])
        return decay

    def state_dict(self):
        return dict(model=self.model.state_dict(), num_updates=self.num_updates, decay_max=self.decay_max)

    def load_state_dict(self, state):
        assert float(state['decay_max']) == self.decay_max
        self.model.load_state_dict(state['model'], strict=True)
        self.num_updates = int(state['num_updates'])
        self.model.eval()
