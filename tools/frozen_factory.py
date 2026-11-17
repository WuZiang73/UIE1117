"""Four explicitly named cumulative stages, with shared accepted native initialization."""
import hashlib,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];SOURCE=ROOT/'source';PLAN=json.loads((ROOT/'PLAN.json').read_text())
os.chdir(SOURCE);sys.path.insert(0,str(SOURCE))
import numpy as np
import torch
import train as original
from baseline_model import BaselineUNet
from optical_regional_isp import RegionalISPUNet
from rms_direction_interaction import RMSDirectionUNet
from dp_ema import configure_drop_path

def atomic(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)
def sha(path):
 d=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):d.update(b)
 return d.hexdigest()
def state_sha(state):
 d=hashlib.sha256()
 for name,value in sorted(state.items()):
  d.update(name.encode());d.update(str(value.dtype).encode());d.update(str(tuple(value.shape)).encode());d.update(value.detach().contiguous().cpu().numpy().tobytes())
 return d.hexdigest()
def seed():
 import random
 torch.set_num_threads(2);random.seed(7);np.random.seed(7);torch.manual_seed(7);torch.cuda.manual_seed_all(7)
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
 torch.use_deterministic_algorithms(True,warn_only=False)
def create_model(mode,number,device='cpu'):
 assert mode in PLAN['methods'] and number in PLAN['initialization_seeds']
 with torch.random.fork_rng(devices=[]):
  torch.random.default_generator.manual_seed(7)
  kwargs=dict(in_channels=3,feature_channels=32,variant=None if mode=='baseline' else 'M2v2R',semantic=None)
  if mode=='dino_cross_stage':model=RMSDirectionUNet(relation_mode='stage_disagreement',**kwargs)
  elif mode=='isp_control':model=RegionalISPUNet(regional_isp_mode='diagonal',**kwargs)
  else:model=BaselineUNet(**kwargs)
 native=torch.load(ROOT/'initializations'/f'native_seed{number}.pth',map_location='cpu')
 missing,unexpected=model.load_state_dict(native,strict=False)
 assert all(mode=='dino_cross_stage' and (k.startswith('interaction.') or k=='stage_distance') for k in missing)
 assert all(mode=='baseline' and k.startswith(('dark_context_v2.','dark_dec1_v2.')) for k in unexpected)
 assert all(torch.equal(model.state_dict()[k],v) for k,v in native.items() if k in model.state_dict())
 configure_drop_path(model,PLAN['dropout']['rates']);return model.to(device)
def snapshot():return {str(p.relative_to(SOURCE)):sha(p) for p in sorted(SOURCE.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and 'UnderWaterDataset' not in p.parts}
def initialize():
 seed();rows={};old=json.loads((ROOT.parent/'baseline_c0_isp_ssim_ablation_20261109/INITIALIZATIONS.json').read_text())['models'];latest=json.loads((ROOT.parent/'dino_rms_direction_20261110/INITIALIZATIONS.json').read_text())['models']
 for n in PLAN['initialization_seeds']:
  keys=set(create_model('baseline',n).state_dict());native=torch.load(ROOT/'initializations'/f'native_seed{n}.pth',map_location='cpu');common=state_sha({k:v for k,v in native.items() if k in keys})
  for mode in PLAN['methods']:
   m=create_model(mode,n);state=m.state_dict();count=sum(p.numel() for p in m.parameters());assert count=={'baseline':480995,'c0':485143,'isp_control':485143,'dino_cross_stage':645778}[mode]
   assert state_sha({k:v for k,v in state.items() if k in keys})==common
   prior=latest[f'{n}/stage_disagreement'] if mode=='dino_cross_stage' else old[f'{n}/{mode}']
   assert state_sha(state)==prior['full_sha'];assert common==prior['common_backbone_sha']
   rows[f'{n}/{mode}']=dict(full_sha=state_sha(state),common_backbone_sha=common,total_parameters=count,trainable_parameters=count,native_file_sha256=sha(ROOT/'initializations'/f'native_seed{n}.pth'))
 actual=snapshot();expected=json.loads((ROOT/'SOURCE_PROVENANCE.json').read_text())['files'];assert actual==expected and len(actual)==61
 atomic(ROOT/'INITIALIZATIONS.json',dict(at=time.time(),models=rows,all8_states_exactly_match_accepted_versions=True,common_backbone_exact_between_all4=True))
 print('FOUR_STAGES8_NATIVE_INITIALIZATIONS_EXACT',flush=True)
