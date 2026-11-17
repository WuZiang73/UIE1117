"""Input-only Table2 evaluation; independent metric lock required before admission."""
import json,time
from pathlib import Path
import numpy as np
import torch
from frozen_factory import ROOT,PLAN,create_model,atomic,sha,state_sha
from external_geometry import predict_external

def evaluate_external(dataset,mode,number):
 from experiment import configure
 configure();dest=ROOT/'runs'/dataset/f'seed{number}'/mode;dest.mkdir(parents=True,exist_ok=True)
 try:
  ready=json.loads((ROOT/'EXTERNAL_READINESS.json').read_text());assert dataset in ready['ready_datasets']
  for rel,digest in json.loads((ROOT/'EXTERNAL_MATH_LOCK.json').read_text())['files'].items():assert sha(ROOT/rel)==digest
  from external_metrics import MetricSuite,color_chart_score
  from utils.dataset import get_loader
  source=ROOT/'runs/UIEB'/f'seed{number}'/mode;assert (source/'EVALUATED.json').exists()
  suite=MetricSuite(ROOT/'evaluators') if dataset!='Color-Check7' else None
  for weights in ['raw','ema']:
   output=dest/f'{weights}_external_per_image.json'
   if output.exists():continue
   checkpoint=source/('best_model.pth' if weights=='ema' else 'best_raw_model.pth');digest=sha(checkpoint)
   locked=json.loads((ROOT/'WEIGHTS_LOCK.json').read_text())[f'UIEB/{number}/{mode}'][weights];assert digest==locked
   model=create_model(mode,number,'cuda').eval();model.load_state_dict(torch.load(checkpoint,map_location='cpu'),strict=True);before=state_sha(model.state_dict());rows=[]
   if dataset=='Challenging-60' and number==7 and weights=='raw':
    from profile_efficiency import profile_efficiency
    profile_efficiency(model,mode,number)
   folder=ROOT/'data/UnderWaterDataset'/dataset/'test'
   for x,names in get_loader(str(folder),1,256,train=False,resize=True,num_workers=1,shuffle=False,pin_memory=True,non_ref=True):
    with torch.no_grad():raw=predict_external(model,x.cuda());assert torch.isfinite(raw).all();prediction=raw.clamp(0,1).cpu()
    values=color_chart_score(prediction[0],names[0],ROOT/'evaluators/colorcheck_annotations.json') if dataset=='Color-Check7' else suite(prediction)
    rows.append(dict(name=names[0],**values))
   assert len(rows)=={'Challenging-60':60,'U45':45,'Color-Check7':7}[dataset]
   assert state_sha(model.state_dict())==before and sha(checkpoint)==digest
   keys=['CIEDE2000'] if dataset=='Color-Check7' else ['UCIQE','NIQE','URanker']
   result=dict(evaluation_version='uniform_pad8_crop_v2',preprocessing='Author NR resize/crop-to4; all4 models receive right/bottom reflect-pad to8, outputs cropped back before scoring.',at=time.time(),dataset=dataset,mode=mode,initialization_seed=number,weights=weights,source_dataset='UIEB',source_checkpoint=str(checkpoint),checkpoint_sha256=digest,n=len(rows),images=rows,
    metric_lock_sha256=sha(ROOT/'EXTERNAL_MATH_LOCK.json'),input_only=True,model_state_unchanged=True)
   result.update({key:float(np.mean([row[key] for row in rows])) for key in keys});atomic(output,result)
  atomic(dest/'EVALUATED.json',dict(at=time.time(),dataset=dataset,source_dataset='UIEB',Raw_and_EMA=True,all_required_metrics_present=True))
 except BaseException as exc:
  import traceback
  atomic(dest/'EVALUATION_ERROR.json',dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()));raise
if __name__=='__main__':raise RuntimeError('Launch through the single owner and experiment.py --evaluate external')
