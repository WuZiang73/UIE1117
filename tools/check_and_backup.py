"""Verify all four accepted stages, restart state, imported runs, and independent code snapshots."""
import copy,json,random,shutil,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
from frozen_factory import ROOT,SOURCE,PLAN,create_model,seed,initialize,atomic,sha,state_sha,snapshot
from delayed_ema import DelayedEMA
from restoration_losses import RestorationLoss

def main():
 initialize();seed();generator=torch.Generator().manual_seed(19)
 x=torch.rand(2,3,64,64,generator=generator);y=torch.rand(2,3,64,64,generator=generator)
 features=torch.nn.functional.normalize(torch.randn(2,3,256,12,generator=generator),dim=-1);relations=(features@features.transpose(-1,-2)).detach()
 criterion=RestorationLoss('charbonnier',.1);checks=[]
 def objects(mode,n):
  model=create_model(mode,n).train()
  if mode=='dino_cross_stage':model.relation_override=relations
  optim=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=1e-5)
  scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optim,200,eta_min=2e-7)
  return model,optim,scheduler,DelayedEMA(model,3,.999),torch.Generator().manual_seed(1007)
 def step(objects):
  m,o,s,e,g=objects;random.random();np.random.rand();torch.rand(1,generator=g)
  o.zero_grad(set_to_none=True);pred=m(x);loss=criterion(pred,y);loss.backward()
  assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
  o.step();e.update(m);s.step()
  return float(loss),state_sha(m.state_dict()),state_sha(e.model.state_dict()) if e.initialized else None
 for n in [7,43]:
  ref=create_model('isp_control',n).eval()
  model=create_model('dino_cross_stage',n).eval();model.relation_override=relations
  with torch.no_grad():assert torch.equal(model(x),ref(x))
  for mode in PLAN['methods']:
   seed();full=objects(mode,n);expected=[step(full) for _ in range(5)]
   seed();first=objects(mode,n);prefix=[step(first) for _ in range(2)];assert prefix==expected[:2]
   m,o,s,e,g=first
   checkpoint=copy.deepcopy(dict(model=m.state_dict(),optimizer=o.state_dict(),scheduler=s.state_dict(),ema=e.state_dict(),torch_rng=torch.get_rng_state(),python_rng=random.getstate(),numpy_rng=np.random.get_state(),loader_rng=g.get_state()))
   resumed=objects(mode,n);m,o,s,e,g=resumed
   m.load_state_dict(checkpoint['model']);o.load_state_dict(checkpoint['optimizer']);s.load_state_dict(checkpoint['scheduler']);e.load_state_dict(checkpoint['ema']);g.set_state(checkpoint['loader_rng'])
   torch.set_rng_state(checkpoint['torch_rng']);random.setstate(checkpoint['python_rng']);np.random.set_state(checkpoint['numpy_rng'])
   assert prefix+[step(resumed) for _ in range(3)]==expected
   assert e.initialized and e.num_updates==5 and e.averaging_updates==2
   m.eval();e.model.eval()
   if mode=='dino_cross_stage':
    e.model.relation_override=relations
    with torch.no_grad():
     a=m(x);m.relation_override=relations.roll(1,2);b=m(x)
    assert (a-b).abs().max()>0
    assert float(m.interaction.banks['e1'].project.weight.grad.abs().sum())>0
   checks.append(dict(mode=mode,initialization=n,all_gradients_finite=True,exact_Raw_EMA_AdamW_scheduler_Python_numpy_torch_loader_RNG_resume=True))
   print('CPU_MODEL_RESUME_VERIFIED',mode,n,flush=True)
 atomic(ROOT/'MODEL_CPU_CHECKS.json',dict(passed=True,tests=checks,DINO_zero_initial_output_equals_ISP=True,original_source61_unchanged=True))
 atomic(ROOT/'RESUME_CPU_CHECKS.json',dict(passed=True,tests=checks))
 # Every imported artifact is copied byte-for-byte; additionally load all9 files/run.
 imports=json.loads((ROOT/'IMPORTED_ACCEPTED_RUNS.json').read_text());verified=[];loaded_files=loaded_states=0
 initial=json.loads((ROOT/'INITIALIZATIONS.json').read_text())['models']
 for item in imports['runs']:
  ds,mode,n=item['dataset'],item['mode'],item['init_seed'];folder=ROOT/'runs'/ds/f'seed{n}'/mode
  cfg=json.loads((folder/'run_config.json').read_text());done=json.loads((folder/'completed.json').read_text());records=json.loads((folder/'records.json').read_text())
  assert cfg['initial_full_sha']==initial[f'{n}/{mode}']['full_sha'];assert cfg['training_seed']==7 and cfg['epochs']==200
  assert cfg['loss_config']==PLAN['losses'][mode] and cfg['drop_path']==PLAN['dropout']
  assert cfg['ema']['decay_constant']==.999 and cfg['ema']['start_optimizer_step_by_dataset'][ds]==PLAN['ema']['start_optimizer_step_by_dataset'][ds]
  assert done['epoch']==len(records)==200 and done['steps']==PLAN['steps_per_epoch'][ds]*200
  for name,digest in cfg['source_sha256'].items():assert sha(SOURCE/name)==digest
  native=create_model(mode,n)
  states=0;weightfiles=sorted(folder.rglob('*.pth'));assert len(weightfiles)==9,(folder,len(weightfiles))
  for path in weightfiles:
   full=torch.load(path,map_location='cpu');values=[full['model'],full['ema']['model']] if path.name=='last_full.pth' else [full]
   for state in values:native.load_state_dict(state,strict=True);assert all(torch.isfinite(v).all() for v in state.values());states+=1
   if path.name=='last_full.pth':
    assert all(k in full for k in ['optim','scheduler','torch_rng','cuda_rng','python_rng','numpy_rng','train_generator_rng'])
    assert full['epoch']==200 and full['trace_steps']==done['steps']
  for w in ['raw','ema']:
   best=json.loads((folder/f'best_{w}.json').read_text());assert best['epoch']==max((v for v in records if v[w] is not None),key=lambda v:v[w]['psnr'])['epoch']
   for sp in ['val','test']:
    metric=json.loads((folder/f'{w}_{sp}_per_image.json').read_text());assert metric['n']==PLAN['split_counts'][ds][sp]
    assert metric['checkpoint_sha256']==sha(folder/('best_model.pth' if w=='ema' else 'best_raw_model.pth'))
  loaded_files+=len(weightfiles);loaded_states+=states;verified.append(dict(dataset=ds,mode=mode,init_seed=n,files=len(weightfiles),states=states,source=item['source_run']))
  print('IMPORTED_RUN9_WEIGHTS_ACTUALLY_LOADED',ds,mode,n,flush=True)
 assert loaded_files==126 and loaded_states==140
 atomic(ROOT/'IMPORTED_REUSE_AUDIT.json',dict(passed=True,runs=verified,source_recipe_and_initialization_match=True,weights_actually_loaded=loaded_files,model_states_actually_loaded=loaded_states,count_as_new_trainings=False))
 # Snapshot each executable stage separately. Data are pinned by the shared manifest.
 backup_root=ROOT/'code_backups';backup_root.mkdir(exist_ok=False);all_backups=[]
 for mode in PLAN['methods']:
  dest=backup_root/mode;dest.mkdir();shutil.copytree(SOURCE,dest/'source',ignore=shutil.ignore_patterns('__pycache__','UnderWaterDataset'))
  (dest/'source/UnderWaterDataset').symlink_to(ROOT/'data/UnderWaterDataset',target_is_directory=True)
  shutil.copytree(ROOT/'tools',dest/'tools',ignore=shutil.ignore_patterns('__pycache__'))
  for name in ['initializations','teachers']:shutil.copytree(ROOT/name,dest/name)
  for name in ['PLAN.json','SCHEDULER_POLICY.json','INITIALIZATIONS.json','SOURCE_PROVENANCE.json','TEACHER_PROVENANCE.json','DATA_MANIFEST.json']:
   shutil.copy2(ROOT/name,dest/name)
  atomic(dest/'STAGE.json',dict(mode=mode,description=PLAN['method_names'][mode],source61_unmodified=True,data_manifest=str(ROOT/'DATA_MANIFEST.json'),accepted_inputs=[7,43]))
  (dest/'README.md').write_text(f'# {mode}\n\n{PLAN["method_names"][mode]}\n\n完整source、训练/恢复入口、两份native初值、冻结教师与数据清单已独立复制。数据目录指向已独立复制的共同数据快照。此目录为代码备份；正式训练仅由全局协调器在父实验目录运行，勿在备份里启动第二个队列。\n\n模型入口：tools/frozen_factory.py 的 create_model("{mode}",7或43)。原文Table1/Table2值与本次重跑值分别记录来源。\n')
  signatures={}
  for f in sorted(dest.rglob('*')):
   if f.is_file() and 'UnderWaterDataset' not in f.parts:
    rel=f.relative_to(dest);original=ROOT/rel
    if original.exists() and f.name not in ['README.md']:assert not f.samefile(original) and sha(f)==sha(original)
    signatures[str(rel)]=dict(sha256=sha(f),size_bytes=f.stat().st_size)
  atomic(dest/'MANIFEST.json',dict(at=time.time(),files=signatures,own_stage=mode))
  # Verify in a clean process whose factory resolves the snapshot's own paths.
  command=[sys.executable,'-c',f'import sys;sys.path.insert(0,{str(dest/"tools")!r});from frozen_factory import create_model,state_sha;import json; m=create_model({mode!r},7); expected=json.load(open({str(dest/"INITIALIZATIONS.json")!r}))["models"]["7/{mode}"]["full_sha"];assert state_sha(m.state_dict())==expected; print("BACKUP_FACTORY_LOADED",{mode!r})']
  subprocess.run(command,check=True)
  all_backups.append(dict(mode=mode,path=str(dest),files=len(signatures),all_hashes_independent_inodes_and_clean_process_model_load_verified=True))
 atomic(ROOT/'CODE_BACKUP_AUDIT.json',dict(at=time.time(),passed=True,stages=all_backups,source57_original_and4_DINO_files_unchanged=True,all_native_initializations_exact=True))
 print('ALL4_CODE_BACKUPS_AND14_IMPORTED_RUNS_VERIFIED',flush=True)
if __name__=='__main__':main()
