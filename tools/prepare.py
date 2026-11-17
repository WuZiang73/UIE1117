"""Independent four-stage snapshots and the complete Table1/Table2 task matrix."""
from pathlib import Path
import ast,hashlib,json,math,shutil,time
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT.parent
RMS=BASE/'dino_rms_direction_20261110';OLD=BASE/'baseline_c0_isp_ssim_ablation_20261109';OWNER=BASE/'dino_factorized_cross_stage_20261110'
def read(p):return json.loads(p.read_text())
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def sha(p):
 d=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):d.update(b)
 return d.hexdigest()
def copytree(a,b):shutil.copytree(a,b,ignore=shutil.ignore_patterns('__pycache__','UnderWaterDataset'))
def main():
 assert not (ROOT/'PLAN.json').exists()
 for d in ['logs','checks','references','initializations','teachers','data/UnderWaterDataset','workers','external','paper']:(ROOT/d).mkdir(parents=True,exist_ok=True)
 copytree(RMS/'source',ROOT/'source');(ROOT/'source/UnderWaterDataset').symlink_to('../data/UnderWaterDataset',target_is_directory=True)
 for name in ['base_equivalent_operators.py','strict_math.py','delayed_ema.py','restoration_losses.py','experiment.py']:
  shutil.copy2(RMS/'tools'/name,ROOT/'tools'/name)
 shutil.copy2(RMS/'TEACHER_PROVENANCE.json',ROOT/'TEACHER_PROVENANCE.json')
 shutil.copy2(RMS/'teachers/vitb14.pth',ROOT/'teachers/vitb14.pth')
 t=read(ROOT/'TEACHER_PROVENANCE.json');t.update(local=str(ROOT/'teachers/vitb14.pth'),source=str(RMS/'teachers/vitb14.pth'));assert sha(ROOT/'teachers/vitb14.pth')==t['sha256'];write(ROOT/'TEACHER_PROVENANCE.json',t)
 for n in [7,43]:
  a=RMS/'initializations'/f'native_seed{n}.pth';b=ROOT/'initializations'/a.name;shutil.copy2(a,b);assert sha(a)==sha(b) and not a.samefile(b)
 for name in ['PARENT_SOURCE_PROVENANCE.json','PARENT_INITIALIZATIONS.json']:
  shutil.copy2(RMS/'references'/name,ROOT/'references'/name)
 for ds in ['UIEB','LSUI']:
  shutil.copy2(OLD/'references'/f'{ds}_seed7_TRACE.json',ROOT/'references'/f'{ds}_seed7_TRACE.json')
  for n in [7,43]:
   dst=ROOT/'references/historical'/ds/f'seed{n}/isp_control';dst.mkdir(parents=True)
   shutil.copy2(OLD/'runs'/ds/f'seed{n}/isp_control/TRACE.json',dst/'TRACE.json')
 original=read(ROOT/'references/PARENT_SOURCE_PROVENANCE.json')['files'];assert len(original)==57
 for name,digest in original.items():assert sha(ROOT/'source'/name)==digest
 source={str(p.relative_to(ROOT/'source')):sha(p) for p in sorted((ROOT/'source').rglob('*')) if p.is_file() and '__pycache__' not in p.parts and 'UnderWaterDataset' not in p.parts}
 assert len(source)==61
 write(ROOT/'SOURCE_PROVENANCE.json',dict(files=source,original57_bitidentical=True,source_parent=str(RMS)))
 names=['UIEB','LSUI','UFO-120','EUVP-Scene','EUVP-Dark'];counts={};files={};splits={};official=BASE/'UnderWaterDataset_official'
 for ds in names+['Challenging-60','U45']:
  shutil.copytree(official/ds,ROOT/'data/UnderWaterDataset'/ds)
  splits[ds]={}
  for sp in ['train','val','test'] if ds in names else ['test']:
   folder=ROOT/'data/UnderWaterDataset'/ds/sp
   source=folder/'input' if ds in names else folder
   ns=sorted(f.name for f in source.iterdir() if f.is_file() and f.suffix.lower() in ['.jpg','.jpeg','.png','.bmp'])
   if ds in names:assert ns==sorted(f.name for f in (folder/'GT').iterdir() if f.is_file() and f.suffix.lower() in ['.jpg','.jpeg','.png','.bmp'])
   splits[ds][sp]=ns
  counts[ds]={sp:len(v) for sp,v in splits[ds].items()}
  for f in sorted((ROOT/'data/UnderWaterDataset'/ds).rglob('*')):
   if f.is_file():
    rel=str(f.relative_to(ROOT));other=official/ds/f.relative_to(ROOT/'data/UnderWaterDataset'/ds)
    digest=sha(f);assert digest==sha(other) and not f.samefile(other)
    files[rel]=dict(sha256=digest,size_bytes=f.stat().st_size)
  print('COPIED_AND_HASHED_DATA',ds,counts[ds],flush=True)
 write(ROOT/'DATA_MANIFEST.json',dict(files=files,splits=splits,split_counts=counts,official_source=str(official)))
 write(ROOT/'DATA_COPY_AUDIT.json',dict(passed=True,files=len(files),all_hashes_and_independent_inodes_verified=True,all_paired_filenames_match=True))
 plan=read(OLD/'PLAN.json');methods=['baseline','c0','isp_control','dino_cross_stage'];steps={ds:math.ceil(counts[ds]['train']/24) for ds in names}
 plan.update(created_at=time.time(),scope='Four cumulative versions on all5 paired datasets and all3 Table2 external sets; current accepted recipe, all outcomes retained.',
  datasets=names,methods=methods,total_jobs=40,initialization_seeds=[7,43],physical_gpu=1,authorized_gpus=[1,4],evaluation_authorized_gpus=[1,4],max_active_physical_gpus=2,
  coordinator_root=str(OWNER),first_batch_root=str(RMS),data_root=str(ROOT/'data/UnderWaterDataset'),backup_root=str(BASE/'four_stage_all_datasets_backup_20261116'),
  excel=str(ROOT/'four_stage_paper_comparison.xlsx'),source_reference=str(RMS/'source'),steps_per_epoch=steps,split_counts={ds:counts[ds] for ds in names},
  immediate_evaluation=True,full_external_evaluation_required=True,expected_weight_files=360,expected_model_states=400,expected_evaluations=160,expected_curve_rows=8000,
  external_datasets=['Challenging-60','U45','Color-Check7'],external_source_dataset='UIEB',external_expected_evaluations=48,
  selection=dict(rule='All four fixed cumulative versions are reported. Each Raw/EMA checkpoint chosen independently by full Val PSNR. No architecture selection by Test.'),
  method_names=dict(baseline='Baseline: all proposed points removed',c0='Baseline+C0',isp_control='Baseline+C0+aligned ISP',dino_cross_stage='Baseline+C0+aligned ISP+centered-DINO RMS cross-stage interaction'),
  checkpoint_selection='Immediate Raw/EMA full Val/Test after each run; preserve all seeds, failures and recovery attempts.',
  gpu_authorization='User-owned GPUs1/4 only, maximum2 physical cards. WWE first; genuine CLUIE resumes after ready WWE.',
  allow_parallel_with_prerequisite=False)
 plan['ema']['start_optimizer_step_by_dataset']={'UIEB':2100,'LSUI':10000,'UFO-120':3500,'EUVP-Scene':70*steps['EUVP-Scene'],'EUVP-Dark':70*steps['EUVP-Dark']}
 plan['ema']['extension_to_new_datasets']='Accepted approximatelyepoch70 start, fixed0.999: preserve2100/10000/3500, extend to5110/12950; same across four methods.'
 plan['losses']={m:dict(loss_id='charb_s100',pixel='charbonnier',ssim_weight=.1) for m in methods}
 plan['recipe']['dino_weights']=str(ROOT/'teachers/vitb14.pth')
 plan['initialization_gpus']={'7':1,'43':4};plan['predeclared_job_order']=[dict(dataset=ds,mode=m,init_seed=n,gpu=1 if n==7 else 4) for ds in ['UIEB','EUVP-Scene','UFO-120','EUVP-Dark','LSUI'] for n in [7,43] for m in methods]
 for key in ['cohorts','prerequisite_scope','historical_reference_scope','dataset_gpus']:plan.pop(key,None)
 plan['external_completion_gate']='All3 named external datasets x4models x2initializations xRaw/EMA; absence of Color-Check7 or a metric is incomplete, never silently skipped.'
 write(ROOT/'PLAN.json',plan);shutil.copy2(OWNER/'SCHEDULER_POLICY.json',ROOT/'SCHEDULER_POLICY.json')
 imports=[]
 for j in plan['predeclared_job_order']:
  ds,mode,n=j['dataset'],j['mode'],j['init_seed'];parent=None;oldmode=mode
  if ds in ['UIEB','LSUI'] and mode!='dino_cross_stage':parent=OLD
  elif ds=='LSUI' and mode=='dino_cross_stage':parent=RMS;oldmode='stage_disagreement'
  if parent is None:continue
  source=parent/'runs'/ds/f'seed{n}'/oldmode;target=ROOT/'runs'/ds/f'seed{n}'/mode
  assert read(source/'completed.json')['epoch']==200 and (source/'EVALUATED.json').exists()
  shutil.copytree(source,target)
  hashes={}
  for f in sorted(target.rglob('*')):
   if f.is_file():
    a=source/f.relative_to(target);assert not f.samefile(a) and sha(f)==sha(a)
    hashes[str(f.relative_to(target))]=sha(f)
  row=dict(**j,source_root=str(parent),source_mode=oldmode,source_run=str(source),copied_files=hashes,
   reused_completed_accepted_recipe=True,count_as_new_training=False,prior_Val_selection_preserved=True)
  write(target/'IMPORTED_PROVENANCE.json',row);imports.append(row)
 assert len(imports)==14
 write(ROOT/'IMPORTED_ACCEPTED_RUNS.json',dict(at=time.time(),runs=imports,count=14,new_training_count=26,still_requires_CPU_model_recipe_and_source_verification=True))
 shutil.copy2(BASE/'WWE-UIE_A_Wavelet_amp_White_Balance_Efficient_Network_for_Underwater_Image_Enhancement.pdf',ROOT/'paper/WWE-UIE.pdf')
 for name in ['paper_layout.txt','table1_page6.png','table2_page7.png']:
  shutil.copy2(BASE/'paper_protocol_review_20261116'/name,ROOT/'paper'/name)
 write(ROOT/'PREPARATION_STATUS.json',dict(at=time.time(),data_and_import_copies_complete=True,ready_for_GPU=False,remaining=['factory/engine adaptation','independent CPU/recovery checks','four stage code snapshots with actual loads','lock science and integrate single-owner queue']))
 print('DATA_SOURCE_INITIALIZATIONS_AND14_ACCEPTED_RUNS_COPIED; 26_FRESH_RUNS_PENDING',flush=True)
if __name__=='__main__':main()
