'''New inter-stage branch with direction-preserving RMS amplitude control; ISP untouched.'''
import torch
from torch import nn
from torch.nn import functional as F
from factorized_cross_stage import FactorizedReceiver,FactorizedInteraction,FactorizedDinoUNet
from dino_stage_interaction import teacher_relations
from relation_variants import centered_correlations

class RMSDirectionReceiver(nn.Module):
    def __init__(self,channels,width,sources):
        super().__init__()
        self.width=width;self.sources=tuple(sources)
        self.query=nn.Linear(width,width,bias=False)
        self.layer_logits=nn.Parameter(torch.zeros(3))
        self.semantic_scale=nn.Parameter(torch.zeros(()))
        self.source_logits=nn.Parameter(torch.zeros(len(sources)))
        self.source_router=nn.Linear(width,len(sources),bias=False)
        nn.init.zeros_(self.source_router.weight)
        self.quality_scale=nn.Parameter(torch.zeros(()))
        # Start with a 0.1 relative update allowance; the hard maximum is0.5.
        self.update_budget=nn.Parameter(torch.tensor(-1.3862943611198906))
        self.transport_out=nn.Linear(width,channels,bias=False)
        nn.init.zeros_(self.transport_out.weight)
        self.refine=nn.Sequential(nn.Conv2d(channels,2*channels,1),nn.GELU(),
            nn.Conv2d(2*channels,2*channels,3,padding=1,groups=2*channels),nn.GELU(),
            nn.Conv2d(2*channels,channels,1))
        nn.init.zeros_(self.refine[-1].weight);nn.init.zeros_(self.refine[-1].bias)
        self.capture_diagnostics=False;self.last_diagnostics=None

    def forward(self,feature,receiver,bank,relations,distance):
        assert receiver not in self.sources,'Identity and inter-stage transport must remain separate'
        qk,qv=bank[receiver]
        query=F.normalize(self.query(qk),dim=-1,eps=1e-6)
        relation=(relations*self.layer_logits.softmax(0)[None,:,None,None]).sum(1)
        # A fixed task temperature gives a verifiable [-4,4] logit bound.
        semantic=2.*self.semantic_scale.sigmoid()*relation/.15-2.*distance[None]
        candidates=[];qualities=[];bounds=[]
        for source in self.sources:
            key,value=bank[source]
            key=F.normalize(key,dim=-1,eps=1e-6)
            content=(query@key.transpose(1,2))/.25
            affinity=(content+semantic).softmax(-1)
            # Match the latent value scale across different restoration stages.
            value=F.layer_norm(value,(self.width,))
            candidates.append(affinity@value)
            qualities.append((affinity*relation).sum(-1))
            if self.capture_diagnostics:bounds.append(float(content.detach().abs().max()))
        quality=torch.stack(qualities,-1)
        stage_logits=self.source_router(qk)+self.source_logits[None,None,:]+F.softplus(self.quality_scale)*quality
        gates=stage_logits.softmax(-1)
        mixture=(torch.stack(candidates,-2)*gates[...,None]).sum(-2)
        innovation=mixture-F.layer_norm(qv,(self.width,))
        message=self.transport_out(innovation).transpose(1,2).reshape(feature.shape[0],-1,16,16)
        message=F.interpolate(message,size=feature.shape[-2:],mode='bilinear',align_corners=False)
        raw_update=message+self.refine(feature+message)
        reference_rms=feature.detach().square().mean(1,keepdim=True).sqrt().clamp_min(.05)
        budget=.5*self.update_budget.sigmoid()
        update=budget*reference_rms*raw_update/(raw_update.square().mean(1,keepdim=True)+reference_rms.square()).sqrt()
        if self.capture_diagnostics:
            self.last_diagnostics=dict(receiver=receiver,donors=list(self.sources),receiver_excluded=True,
                task_logit_abs_max=max(bounds),source_mass=dict(zip(self.sources,gates.detach().mean((0,1)).cpu().tolist())),
                stage_mass_sum_max_error=float((gates.detach().sum(-1)-1.).abs().max()),
                relative_update_budget=float(budget.detach()),relative_update_RMS_max=float((update.detach().square().mean(1,keepdim=True).sqrt()/reference_rms).max()))
        return feature+update


class RMSDirectionInteraction(FactorizedInteraction):
    def __init__(self,width=64):
        super().__init__(width)
        # Existing block state is copied exactly; random native ISP init is loaded independently.
        for name,old in list(self.receivers.items()):
            receiver=RMSDirectionReceiver(self.CHANNELS[name],width,self.SOURCES[name])
            receiver.load_state_dict(old.state_dict(),strict=True)
            self.receivers[name]=receiver

class RMSDirectionUNet(FactorizedDinoUNet):
    def __init__(self,*args,relation_mode='stage_transport',**kwargs):
        assert relation_mode in ['stage_nodino','stage_transport','stage_disagreement']
        super().__init__(*args,use_dino=relation_mode!='stage_nodino',**kwargs)
        self.interaction=RMSDirectionInteraction()
        self.relation_mode=relation_mode
    def forward(self,x,return_pair=False):
        if self.relation_mode!='stage_disagreement' or not self.interaction_enabled:
            return super().forward(x,return_pair=return_pair)
        previous=self.relation_override
        raw=previous if previous is not None else teacher_relations(x)
        self.relation_override=centered_correlations(raw)
        try:return super().forward(x,return_pair=return_pair)
        finally:self.relation_override=previous
