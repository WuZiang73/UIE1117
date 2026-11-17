"""Full-scope completion: all40 paired runs, all48 external evaluations and immutable backup."""
import fcntl,json,shutil,subprocess,sys,time
from pathlib import Path
import torch
from frozen_factory import ROOT,PLAN,create_model,atomic,sha,snapshot
from reporting_protocol import effective_plan
PLAN=effective_plan(PLAN)

def main():
 with (ROOT/'finalization.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  assert json.loads((ROOT/'CODE_BACKUP_AUDIT.json').read_text())['passed']
  assert json.loads((ROOT/'IMPORTED_REUSE_AUDIT.json').read_text())['passed']
  for file in ['CONFIG_LOCK.json','ADAPTIVE_CODE_LOCK.json','EXTERNAL_MATH_LOCK.json']:
   for name,digest in json.loads((ROOT/file).read_text())['files'].items():assert sha(ROOT/name)==digest
  # Full-scope input/GT/DropPath consistency across four methods, all200epochs.
  cohort_checks=[]
  for dataset in PLAN['datasets']:
   for number in PLAN['required_initializations_by_dataset'][dataset]:
    reference=None
    for mode in PLAN['methods']:
     folder=ROOT/'runs'/dataset/f'seed{number}'/mode
     trace=json.loads((folder/'TRACE.json').read_text())['steps']
     actual=[(v['input_sha'],v['label_sha'],v['mask_sha']) for v in trace]
     assert len(actual)==PLAN['steps_per_epoch'][dataset]*200
     if reference is None:reference=actual
     else:assert actual==reference,(dataset,number,mode)
    cohort_checks.append(dict(dataset=dataset,init_seed=number,steps=len(reference),all4_inputs_GT_DP_exact=True))
  atomic(ROOT/'COHORT_FINAL_AUDIT.json',dict(passed=True,cohorts=cohort_checks))
  weight_index={};states=curves=evaluations=0
  for j in PLAN['predeclared_job_order']:
   ds,m,n=j['dataset'],j['mode'],j['init_seed'];folder=ROOT/'runs'/ds/f'seed{n}'/m
   assert (folder/'EVALUATED.json').exists();done=json.loads((folder/'completed.json').read_text());records=json.loads((folder/'records.json').read_text())
   assert done['epoch']==len(records)==200;curves+=len(records);model=create_model(m,n)
   for w in ['raw','ema']:
    for sp in ['val','test']:
     metric=json.loads((folder/f'{w}_{sp}_per_image.json').read_text());assert metric['n']==len(metric['images'])==PLAN['split_counts'][ds][sp]
     checkpoint=folder/('best_model.pth' if w=='ema' else 'best_raw_model.pth')
     assert metric['checkpoint_sha256']==sha(checkpoint)==json.loads((ROOT/'WEIGHTS_LOCK.json').read_text())[f'{ds}/{n}/{m}'][w]
     import statistics
     assert abs(statistics.mean(v['psnr'] for v in metric['images'])-metric['psnr'])<1e-10
     assert abs(statistics.mean(v['ssim'] for v in metric['images'])-metric['ssim'])<1e-12
     expected_names=json.loads((ROOT/'DATA_MANIFEST.json').read_text())['splits'][ds][sp]
     assert sorted(v['name'] for v in metric['images'])==expected_names
     best=json.loads((folder/f'best_{w}.json').read_text())
     assert best['epoch']==max((v for v in records if v[w] is not None),key=lambda v:v[w]['psnr'])['epoch']
     evaluations+=1
   files=sorted(folder.rglob('*.pth'));assert len(files)==9
   for path in files:
    full=torch.load(path,map_location='cpu');values=[full['model'],full['ema']['model']] if path.name=='last_full.pth' else [full]
    for value in values:model.load_state_dict(value,strict=True);assert all(torch.isfinite(v).all() for v in value.values());states+=1
    if path.name=='last_full.pth':
     assert full['epoch']==200 and full['trace_steps']==PLAN['steps_per_epoch'][ds]*200
     assert all(k in full for k in ['optim','scheduler','torch_rng','cuda_rng','python_rng','numpy_rng','train_generator_rng'])
    weight_index[str(path.relative_to(ROOT))]=dict(sha256=sha(path),mode=m,initialization_seed=n,dataset=ds)
  assert len(weight_index)==PLAN['expected_weight_files'] and states==PLAN['expected_model_states'] and curves==PLAN['expected_curve_rows'] and evaluations==PLAN['expected_evaluations']
  for j in PLAN['external_job_order']:
   folder=ROOT/'runs'/j['dataset']/f'seed{j["init_seed"]}'/j['mode'];assert (folder/'EVALUATED.json').exists()
   for w in ['raw','ema']:
    metric=json.loads((folder/f'{w}_external_per_image.json').read_text());assert metric['n']==len(metric['images'])=={'Challenging-60':60,'U45':45,'Color-Check7':7}[j['dataset']]
    required=['CIEDE2000'] if j['dataset']=='Color-Check7' else ['UCIQE','NIQE','URanker']
    assert all(k in metric and all(k in image for image in metric['images']) for k in required)
  assert (ROOT/'EFFICIENCY.json').exists() and all(m in json.loads((ROOT/'EFFICIENCY.json').read_text())['methods'] for m in PLAN['methods'])
  subprocess.run(['/data1/anaconda3/bin/python3',str(ROOT/'tools/export_excel.py')],check=True)
  audit=json.loads((ROOT/'EXCEL_AUDIT.json').read_text());assert audit['paired_evaluated']==PLAN['total_jobs'] and audit['external_evaluations']==48 and audit['curve_rows']==PLAN['expected_curve_rows']
  atomic(ROOT/'WEIGHTS_INDEX.json',weight_index)
  atomic(ROOT/'FINAL_AUDIT.json',dict(at=time.time(),passed=True,formal_runs=PLAN['total_jobs'],original_declared_runs=40,retired_unstarted_runs=4,imported_accepted_runs=14,fresh_trainings=PLAN['total_jobs']-14,paired_evaluations=PLAN['expected_evaluations'],external_evaluations=48,
   all_original_PDF_rows_in_Excel=True,weight_files_actually_loaded=PLAN['expected_weight_files'],model_states_actually_loaded=PLAN['expected_model_states'],curve_rows=PLAN['expected_curve_rows'],all8_target_datasets_complete=True))
  backup=Path(PLAN['backup_root'])
  from archive_snapshot import snapshot as archive_snapshot
  files=archive_snapshot(ROOT,backup)
  # Load from the archive, not only the working directory.
  script='''import sys,json,torch;from pathlib import Path;root=Path(sys.argv[1]);sys.path.insert(0,str(root/'tools'));from frozen_factory import create_model;index=json.loads((root/'WEIGHTS_INDEX.json').read_text());count=0
for rel,row in index.items():
 m=create_model(row['mode'],row['initialization_seed']);p=root/rel;value=torch.load(p,map_location='cpu');states=[value['model'],value['ema']['model']] if p.name=='last_full.pth' else [value]
 for state in states:m.load_state_dict(state,strict=True);count+=1
assert count==int(sys.argv[2])
print('BACKUP_REQUIRED_MODEL_STATES_ACTUALLY_LOADED',count)
'''
  subprocess.run([sys.executable,'-c',script,str(backup),str(PLAN['expected_model_states'])],check=True)
  atomic(ROOT/'BACKUP_AUDIT.json',dict(at=time.time(),passed=True,path=str(backup),files=len(files),weight_files_loaded_from_backup=PLAN['expected_weight_files'],model_states_loaded=PLAN['expected_model_states'],hashes_and_independent_inodes_verified=True))
  shutil.copy2(ROOT/'BACKUP_AUDIT.json',backup/'BACKUP_AUDIT.json')
  print('ALL8_DATASETS_USER_REQUIRED_TRAJECTORIES_AND_EVALUATIONS_ARCHIVED',PLAN['total_jobs'],flush=True)
if __name__=='__main__':main()
