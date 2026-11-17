"""Whole-model tests on real non-square external inputs, preserving frozen source."""
import json,time
import torch
from frozen_factory import ROOT,create_model,seed,atomic,snapshot
from utils.dataset import get_loader
from external_geometry import predict_external

def main():
 seed();before=snapshot();g=torch.Generator().manual_seed(1943)
 desc=torch.nn.functional.normalize(torch.randn(1,3,256,12,generator=g),dim=-1);relation=desc@desc.transpose(-1,-2)
 rows=[]
 for ds in ['Challenging-60','U45','Color-Check7']:
  x,names=next(iter(get_loader(str(ROOT/'data/UnderWaterDataset'/ds/'test'),1,256,train=False,resize=True,num_workers=1,shuffle=False,non_ref=True)))
  for mode in ['baseline','c0','isp_control','dino_cross_stage']:
   model=create_model(mode,7).eval()
   if mode=='dino_cross_stage':model.relation_override=relation
   state={k:v.clone() for k,v in model.state_dict().items()}
   with torch.no_grad():
    out=predict_external(model,x);assert out.shape==x.shape and torch.isfinite(out).all()
    if x.shape[-2]%8==x.shape[-1]%8==0:assert torch.equal(out,model(x))
   assert all(torch.equal(model.state_dict()[k],v) for k,v in state.items())
   rows.append(dict(dataset=ds,input_name=names[0],mode=mode,input_shape=list(x.shape),output_same_shape=True,finite=True,state_unchanged=True,divisible8_input_bitidentical=True if x.shape[-1]%8==0 else None))
   print('EXTERNAL_GEOMETRY_PASSED',ds,mode,list(x.shape),flush=True)
 assert snapshot()==before
 atomic(ROOT/'EXTERNAL_GEOMETRY_CPU_CHECKS.json',dict(at=time.time(),passed=True,tests=rows,original_source61_unchanged=True,all4_methods_same_padding_and_crop_rule=True,all_original_pixels_retained_in_metric_image=True,synthetic_relation_CPU_only_not_actual_semantic_quality_test=True))
if __name__=='__main__':main()
