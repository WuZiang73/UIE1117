'''DINO structure relationships for existing inter-stage interaction; ISP untouched.'''
import torch
from factorized_cross_stage import FactorizedDinoUNet
from dino_stage_interaction import teacher_relations

def centered_correlations(relation):
    centered=relation-relation.mean(-1,keepdim=True)-relation.mean(-2,keepdim=True)+relation.mean((-1,-2),keepdim=True)
    norm=centered.diagonal(dim1=-2,dim2=-1).clamp_min(1e-6).sqrt()
    return (centered/(norm[..., :,None]*norm[...,None,:])).clamp(-1.,1.)

def reciprocal_correction(relation,temperature=.15):
    log_partition=torch.logsumexp(relation/temperature,-1)
    return relation-temperature*.5*(log_partition[..., :,None]+log_partition[...,None,:])

class RelationVariantUNet(FactorizedDinoUNet):
    def __init__(self,*args,relation_mode='stage_transport',**kwargs):
        assert relation_mode in ['stage_nodino','stage_transport','stage_disagreement']
        super().__init__(*args,use_dino=relation_mode!='stage_nodino',**kwargs)
        self.relation_mode=relation_mode
    def forward(self,x,return_pair=False):
        if self.relation_mode=='stage_nodino' or not self.interaction_enabled:
            return super().forward(x,return_pair=return_pair)
        original=self.relation_override
        raw=original if original is not None else teacher_relations(x)
        relation=centered_correlations(raw)
        if self.relation_mode=='stage_disagreement':relation=reciprocal_correction(relation)
        self.relation_override=relation
        try:return super().forward(x,return_pair=return_pair)
        finally:self.relation_override=original
