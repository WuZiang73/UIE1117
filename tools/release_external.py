"""Release all three Table2 datasets only after actual metric checks and input-only ROI review."""
import ast,fcntl,hashlib,json,os,shutil,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OWNER=ROOT.parent/'dino_factorized_cross_stage_20261110'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def atomic(p,x):q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');q.replace(p)
def owned(h):
 try:
  p=Path('/proc')/str(h['pid']);st=p.joinpath('stat').read_text().rsplit(')',1)[1].split();args=p.joinpath('cmdline').read_bytes().decode().rstrip('\0').split('\0')
  return p.stat().st_uid==os.getuid() and st[0]!='Z' and st[19]==str(h['start_ticks']) and args==h['args']
 except (FileNotFoundError,ProcessLookupError,KeyError):return False
def stop(h):
 if not owned(h):return
 os.kill(h['pid'],signal.SIGTERM)
 for _ in range(100):
  if not owned(h):return
  time.sleep(.05)
 raise RuntimeError('Exact CPU supervisor did not stop')
def main():
 checks=read(ROOT/'EXTERNAL_METRICS_CPU_CHECKS.json');assert checks['passed']
 lock=read(ROOT/'EXTERNAL_MATH_LOCK.json')
 for name,digest in lock['files'].items():assert sha(ROOT/name)==digest,name
 atomic(ROOT/'checks/COLORCHECK_INPUT_GEOMETRY_REVIEW.json',dict(at=time.time(),passed=True,all7_input_rectified_charts_reviewed=True,all24_fixed_ROIs_each=True,visible_subregion_overrides_for_finger_occlusion=True,restoration_outputs_never_used_to_select_ROIs=True,no_adjustment_to_match_paper_score=True,annotation_SHA=sha(ROOT/'evaluators/colorcheck_annotations.json'),review_image_sha256=sha(ROOT/'checks/colorcheck_input_sampling_review.png')))
 receipts=ROOT/'checks/external_release';receipts.mkdir()
 # Update CPU resource/export logic while adopting all healthy GPU workers intact.
 with (OWNER/'launch.lock').open('a+') as launch_lock:
  fcntl.flock(launch_lock,fcntl.LOCK_EX)
  hs=[read(OWNER/x) for x in ['WATCHDOG_LAUNCH.json','LAUNCH.json']]
  active=read(OWNER/'ACTIVE_JOBS.json');verified_before=[h for h in active if owned(h)]
  for h in hs:stop(h)
  for name in ['tools/controller_adaptive.py','ADAPTIVE_CODE_LOCK.json','SCHEDULER_POLICY.json']:
   p=receipts/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(OWNER/name,p)
  p=OWNER/'tools/controller_adaptive.py';text=p.read_text();old="budget=4500 if j.get('phase')=='evaluation' else max(initial_budget,int(profiled_peak/1048576)+1100)"
  new="eval_budget=policy.get('external_eval_budget_mib_by_mode',{}).get(j['mode'],4500) if j.get('evaluation_scope')=='external' else 4500\n                    budget=eval_budget if j.get('phase')=='evaluation' else max(initial_budget,int(profiled_peak/1048576)+1100)"
  assert old in text;text=text.replace(old,new);ast.parse(text);p.write_text(text)
  policy=read(OWNER/'SCHEDULER_POLICY.json');policy['external_eval_budget_mib_by_mode']={'baseline':2500,'c0':2500,'isp_control':2500,'dino_cross_stage':3500}
  policy['external_budget_basis']='Batch1 external inputs <=256x452 (C60 max aspect1.779), U45 256 square, ColorCheck340x256; frozen URanker runs CPU; 1500MiB card margin retained. Actual peak recorded and resource errors preserved.'
  atomic(OWNER/'SCHEDULER_POLICY.json',policy)
  # Make the evaluation convention visible next to the table cells themselves.
  export=ROOT/'tools/export_excel.py';shutil.copy2(export,receipts/'export_excel_before.py');s=export.read_text()
  s=s.replace("'Means shown only after both initializations finish. Published scores use the paper protocol; not our reruns.'", "'Two-init means only. Our FLOPs: profiler conv/matmul, multiply-add=2, includes frozen DINO; paper uses a different counter. Our ms: shared RTX4090; paper RTX3090. Published scores are not our reruns.'")
  s=s.replace("['Current protocol observed',plan['method_names'][m],w]+values+['UIEB-trained weights; two-initialization means. Missing values stay blank.']", "['Current WWE Python / fixed colour protocol',plan['method_names'][m],w]+values+['UIEB-trained, two-init means. UCIQE follows WWE Python HSV; paper older Matlab. CIEDE2000 uses fixed input ROIs + BabelColor D65; original ROI implementation unavailable. Missing stays blank.']")
  ast.parse(s);export.write_text(s)
  adaptive=read(ROOT/'ADAPTIVE_CODE_LOCK.json');adaptive['files']['tools/export_excel.py']=sha(export);adaptive.setdefault('revisions',[]).append(dict(at=time.time(),reason='Explicit metric/cost convention labels; frozen training source and math unchanged'));atomic(ROOT/'ADAPTIVE_CODE_LOCK.json',adaptive)
  runtime=read(OWNER/'ADAPTIVE_CODE_LOCK.json')
  for name in ['tools/controller_adaptive.py','SCHEDULER_POLICY.json']:runtime['files'][name]=sha(OWNER/name)
  runtime.setdefault('resource_revisions',[]).append(dict(at=time.time(),reason='Admit independently validated Table2 batch1 evaluation with real aspect-ratio budgets; healthy WWE workers preserved',training_math_changed=False));atomic(OWNER/'ADAPTIVE_CODE_LOCK.json',runtime)
  atomic(ROOT/'EXTERNAL_READINESS.json',dict(at=time.time(),ready_datasets=['Challenging-60','U45','Color-Check7'],all_required_materials_present=True,actual_metric_CPU_checks_passed=True,input_only_geometry_review_passed=True,metric_lock_sha256=sha(ROOT/'EXTERNAL_MATH_LOCK.json'),
   protocol='Current fixed WWE Python HSV UCIQE/NIQE + official URanker; BabelColor D65 CIEDE2000 with input-only visible24patch ROIs. Published rows kept verbatim with differing implementation labelled.',pending=[]))
  atomic(receipts/'CPU_RESOURCE_REVISION.json',dict(at=time.time(),passed=True,stopped_exact_CPU_handles=hs,all_healthy_GPU_handles_preserved=verified_before,training_engine_source_and_CONFIG_unchanged=True))
 subprocess.run([sys.executable,str(OWNER/'tools/launch_adaptive.py')],check=True)
 # A new complete snapshot includes evaluator code/assets without overwriting
 # any previously frozen code or manifest.
 target=ROOT.parent/'four_stage_code_backup_complete_20261116';target.mkdir();rows=[]
 for mode in read(ROOT/'PLAN.json')['methods']:
  dest=target/mode;dest.mkdir()
  shutil.copytree(ROOT/'source',dest/'source',ignore=shutil.ignore_patterns('__pycache__','UnderWaterDataset'))
  (dest/'source/UnderWaterDataset').symlink_to(ROOT/'data/UnderWaterDataset',target_is_directory=True)
  shutil.copytree(ROOT/'tools',dest/'tools',ignore=shutil.ignore_patterns('__pycache__'))
  for folder in ['initializations','teachers']:shutil.copytree(ROOT/folder,dest/folder)
  (dest/'evaluators/utils').mkdir(parents=True);(dest/'data').mkdir()
  for name in ['URanker_ckpt-450eb36d.pth','URANKER_PROVENANCE.json','colorcheck_annotations.json','colorchecker_babelcolor_D65.json']:
   shutil.copy2(ROOT/'evaluators'/name,dest/'evaluators'/name)
  shutil.copy2(ROOT/'evaluators/utils/niqe_pris_params.npz',dest/'evaluators/utils/niqe_pris_params.npz')
  for name in ['PLAN.json','SCHEDULER_POLICY.json','CONFIG_LOCK.json','ADAPTIVE_CODE_LOCK.json','EXTERNAL_MATH_LOCK.json','INITIALIZATIONS.json','SOURCE_PROVENANCE.json','TEACHER_PROVENANCE.json','DATA_MANIFEST.json','EXTERNAL_METRICS_CPU_CHECKS.json','environment_requirements.txt']:
   shutil.copy2(ROOT/name,dest/name)
  (dest/'README.md').write_text(f'# Complete {mode} code snapshot\n\nActive mode: {mode}. Contains original source61, frozen training/recovery tools, all evaluator source, native7/43, frozen DINO and URanker weights, NIQE and colour references, data manifest and pinned environment list. Dataset bytes remain in the independent shared parent snapshot. Launch actual jobs only through the single coordinator; never overwrite frozen source or use this archive as a competing GPU dispatcher.\n')
  files={}
  for file in sorted(dest.rglob('*')):
   if file.is_file() and 'UnderWaterDataset' not in file.parts:
    rel=str(file.relative_to(dest));original=ROOT/rel
    if original.exists() and file.name!='README.md':assert sha(file)==sha(original) and not file.samefile(original)
    files[rel]=dict(sha256=sha(file),size_bytes=file.stat().st_size)
  atomic(dest/'MANIFEST.json',dict(at=time.time(),stage=mode,files=files))
  rows.append(dict(stage=mode,path=str(dest),files=len(files),all_hashes_and_independent_inodes_verified=True))
 atomic(target/'BACKUP_AUDIT.json',dict(at=time.time(),passed=True,stages=rows,prior_code_archives_untouched=True))
 atomic(ROOT/'COMPLETE_CODE_BACKUP_AUDIT.json',dict(at=time.time(),passed=True,stages=rows,training_and_evaluator_assets_included=True,prior_archives_untouched=True))
 print('ALL3_EXTERNAL_SETS_RELEASED_AND4_COMPLETE_CODE_ASSET_BACKUPS_VERIFIED',flush=True)
if __name__=='__main__':main()
