"""DINO-guided exchange between restoration stages; the ISP implementation is untouched.

The frozen teacher defines relationships, while every transmitted value belongs
to the restoration network. One shared stage bank feeds three successive
receivers, so shallow/deep features interact rather than being smoothed only
within their own layer. The external teacher is independently frozen/hashed;
it is not redundantly averaged, checkpointed or copied into an EMA shadow.
"""
from pathlib import Path
import json
import torch
from torch import nn
from torch.nn import functional as F
from optical_regional_isp import RegionalISPUNet
from semantic_backbone import FrozenDINOFeatures

_TEACHERS={}

def teacher_relations(image):
    device=str(image.device)
    if device not in _TEACHERS:
        # Loading a frozen teacher must not perturb restoration RNG streams.
        with torch.random.fork_rng(devices=[]):
            model=FrozenDINOFeatures(weights_path=str(Path(__file__).resolve().parents[1]/'teachers/vitb14.pth'),layers=(2,6,10))
        provenance=json.loads((Path(__file__).resolve().parents[1]/'TEACHER_PROVENANCE.json').read_text())
        assert model.weights_sha256==provenance['sha256']
        _TEACHERS[device]=model.to(image.device).eval()
    model=_TEACHERS[device]
    assert not model.training and not any(p.requires_grad for p in model.parameters())
    with torch.no_grad():
        patches,_=model(image)
        tokens=[F.normalize(p.flatten(2).transpose(1,2),dim=-1,eps=1e-6) for p in patches]
        return torch.stack([t@t.transpose(1,2) for t in tokens],1).detach()


class TokenBank(nn.Module):
    def __init__(self,channels,width):
        super().__init__()
        self.project=nn.Linear(channels,2*width)
        self.key_norm=nn.LayerNorm(width)

    def forward(self,feature):
        tokens=F.adaptive_avg_pool2d(feature,(16,16)).flatten(2).transpose(1,2)
        key,value=self.project(tokens).chunk(2,-1)
        return self.key_norm(key),value


class StageReceiver(nn.Module):
    def __init__(self,channels,width,sources,mode):
        super().__init__()
        self.sources=tuple(sources);self.mode=mode;self.width=width
        self.query=nn.Linear(width,width,bias=False)
        self.source_logits=nn.Parameter(torch.zeros(len(sources)))
        self.layer_logits=nn.Parameter(torch.zeros(3))
        self.semantic_scale=nn.Parameter(torch.zeros(()))
        self.transport_out=nn.Linear(width,channels,bias=False)
        nn.init.zeros_(self.transport_out.weight)
        self.refine=nn.Sequential(nn.Conv2d(channels,2*channels,1),nn.GELU(),
            nn.Conv2d(2*channels,2*channels,3,padding=1,groups=2*channels),nn.GELU(),
            nn.Conv2d(2*channels,channels,1))
        nn.init.zeros_(self.refine[-1].weight);nn.init.zeros_(self.refine[-1].bias)

    def forward(self,feature,receiver,bank,relations,distance):
        qk,qv=bank[receiver]
        query=self.query(qk)
        keys=torch.cat([bank[s][0] for s in self.sources],1)
        values=torch.cat([bank[s][1] for s in self.sources],1)
        content=query@keys.transpose(1,2)/(self.width**.5)
        relation=(relations*self.layer_logits.softmax(0)[None,:,None,None]).sum(1)/.15
        if self.mode=='stage_disagreement':
            # Multi-depth agreement suppresses semantic links whose interpretation
            # changes between DINO layers; live restoration evidence stays active.
            trust=(1.-4.*relations.var(1,unbiased=False)).clamp(.1,1.)
            relation=relation*trust
        relation=F.softplus(self.semantic_scale)*relation-2.*distance[None]
        logits=content+relation.repeat(1,1,len(self.sources))
        logits=logits+self.source_logits.repeat_interleave(256)[None,None,:]
        affinity=logits.softmax(-1)
        # A receiver-relative innovation avoids replacing native features by a
        # low-resolution semantic average; sources contain actual restoration values.
        innovation=affinity@values-qv
        message=self.transport_out(innovation).transpose(1,2).reshape(feature.shape[0],-1,16,16)
        result=feature+F.interpolate(message,size=feature.shape[-2:],mode='bilinear',align_corners=False)
        return result+self.refine(result)


class CrossStageInteraction(nn.Module):
    CHANNELS={'e1':32,'e2':64,'b':128,'d1':64,'d2':32}
    SOURCES={'b':('e1','e2','b'),'d1':('e1','e2','b','d1'),'d2':('e1','e2','b','d1','d2')}

    def __init__(self,mode,width=64):
        super().__init__()
        self.banks=nn.ModuleDict({name:TokenBank(ch,width) for name,ch in self.CHANNELS.items()})
        self.receivers=nn.ModuleDict({name:StageReceiver(self.CHANNELS[name],width,sources,mode) for name,sources in self.SOURCES.items()})


class DinoStageUNet(RegionalISPUNet):
    def __init__(self,*args,interaction_mode='stage_transport',**kwargs):
        assert interaction_mode in ['stage_transport','stage_disagreement','stage_nodino']
        super().__init__(*args,regional_isp_mode='diagonal',**kwargs)
        self.interaction_mode=interaction_mode
        self.interaction=CrossStageInteraction(interaction_mode)
        self.interaction_enabled=True
        self._bank={};self._relations=None;self.relation_override=None
        yy,xx=torch.meshgrid(torch.linspace(0,1,16),torch.linspace(0,1,16),indexing='ij')
        coordinates=torch.stack([yy,xx],-1).reshape(256,2)
        self.register_buffer('stage_distance',(coordinates[:,None]-coordinates[None]).square().sum(-1))
        # The ISP manually evaluates encoder1; capture its completed output at
        # down1 instead of installing a hook that would never be called.
        self.down1.register_forward_pre_hook(self._capture_first)
        self.encoder2.register_forward_hook(self._capture_second)
        self.bottleneck.register_forward_hook(self._exchange_bottleneck)
        self.dark_dec1_v2.register_forward_hook(self._exchange_decoder1)
        self.decoder2.register_forward_hook(self._exchange_decoder2)

    def _capture_first(self,module,args):
        if self.interaction_enabled:self._bank['e1']=self.interaction.banks['e1'](args[0])

    def _capture_second(self,module,args,output):
        if self.interaction_enabled:self._bank['e2']=self.interaction.banks['e2'](output)

    def _exchange(self,name,output):
        if not self.interaction_enabled:return output
        self._bank[name]=self.interaction.banks[name](output)
        result=self.interaction.receivers[name](output,name,self._bank,self._relations,self.stage_distance)
        self._bank[name]=self.interaction.banks[name](result)
        return result

    def _exchange_bottleneck(self,module,args,output):return self._exchange('b',output)
    def _exchange_decoder1(self,module,args,output):return self._exchange('d1',output)
    def _exchange_decoder2(self,module,args,output):return self._exchange('d2',output)

    def forward(self,x,return_pair=False):
        if self.interaction_enabled:
            if self.relation_override is not None:self._relations=self.relation_override
            elif self.interaction_mode=='stage_nodino':self._relations=x.new_zeros(x.shape[0],3,256,256)
            else:self._relations=teacher_relations(x)
        try:return super().forward(x,return_pair=return_pair)
        finally:self._bank={};self._relations=None
