"""EMA initialized from current Raw at a delayed step; fixed decay thereafter."""
from copy import deepcopy
import torch


class DelayedEMA:
    def __init__(self, model, start_step, decay=.999):
        assert int(start_step)==start_step and start_step>=1 and 0<=decay<1
        self.model=deepcopy(model).eval()
        self.model.requires_grad_(False)
        self.start_step=int(start_step)
        self.decay_max=self.decay=float(decay)
        self.num_updates=0  # Optimizer-step clock, including the inactive prefix.
        self.averaging_updates=0
        self.initialized=False

    @torch.no_grad()
    def update(self, model):
        self.num_updates+=1
        if self.num_updates<self.start_step:
            return None
        if not self.initialized:
            assert self.num_updates==self.start_step
            self.model.load_state_dict(model.state_dict(),strict=True)
            self.initialized=True
        else:
            source=dict(model.named_parameters())
            for name,parameter in self.model.named_parameters():
                parameter.mul_(self.decay).add_(source[name].detach(),alpha=1-self.decay)
            buffers=dict(model.named_buffers())
            for name,buffer in self.model.named_buffers():buffer.copy_(buffers[name])
            self.averaging_updates+=1
        assert self.averaging_updates==self.num_updates-self.start_step
        return self.decay

    def state_dict(self):
        return dict(model=self.model.state_dict(),num_updates=self.num_updates,
            averaging_updates=self.averaging_updates,initialized=self.initialized,
            start_step=self.start_step,decay_max=self.decay)

    def load_state_dict(self, value):
        assert int(value['start_step'])==self.start_step and float(value['decay_max'])==self.decay
        self.model.load_state_dict(value['model'],strict=True)
        self.num_updates=int(value['num_updates']);self.averaging_updates=int(value['averaging_updates'])
        self.initialized=bool(value['initialized'])
        assert self.initialized==(self.num_updates>=self.start_step)
        assert self.averaging_updates==max(0,self.num_updates-self.start_step)
