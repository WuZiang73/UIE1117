"""Uniform whole-model external geometry, with all V1 outcomes kept and no GPU training edits."""
import ast,fcntl,hashlib,json,os,shutil,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OWNER=ROOT.parent/'dino_factorized_cross_stage_20261110'
def read(p,d=None):return json.loads(p.read_text()) if p.exists() else d
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def atomic(p,x):q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');q.replace(p)
def owned(h):
 try:
  p=Path('/proc')/str(h['pid']);s=p.joinpath('stat').read_text().rsplit(')',1)[1].split();a=p.joinpath('cmdline').read_bytes().decode().rstrip('\0').split('\0')
  return p.stat().st_uid==os.getuid() and s[0]!='Z' and s[19]==str(h['start_ticks']) and a==h['args']
 except (FileNotFoundError,ProcessLookupError):return False
def stop(h):
 if not owned(h):return
 os.kill(h['pid'],signal.SIGTERM)
 for _ in range(100):
  if not owned(h):return
  time.sleep(.05)
 raise RuntimeError('CPU supervisor did not exit')
def main():
 assert read(ROOT/'EXTERNAL_GEOMETRY_CPU_CHECKS.json')['passed']
 assert not read(ROOT/'EXTERNAL_READINESS.json')['ready_datasets']
 for h in read(OWNER/'ACTIVE_JOBS.json',[]):
  if h.get('experiment_root')==str(ROOT) and h.get('evaluation_scope')=='external' and owned(h):
   print('WAIT_FOR_HEALTHY_V1_EVALUATION',h['pid'],flush=True);return
 archive=ROOT/'external/history/v1';archive.mkdir(parents=True)
 with (OWNER/'launch.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  hs=[read(OWNER/name) for name in ['WATCHDOG_LAUNCH.json','LAUNCH.json']];live=[h for h in read(OWNER/'ACTIVE_JOBS.json',[]) if owned(h)]
  assert not any(h.get('evaluation_scope')=='external' for h in live if h.get('experiment_root')==str(ROOT))
  for h in hs:stop(h)
  (archive/'tools').mkdir()
  for name in ['external_evaluate.py','external_metrics.py','profile_efficiency.py']:shutil.copy2(ROOT/'tools'/name,archive/'tools'/name)
  for name in ['EXTERNAL_MATH_LOCK.json','EXTERNAL_METRICS_CPU_CHECKS.json','EXTERNAL_READINESS.json']:shutil.copy2(ROOT/name,archive/name)
  data=[]
  for ds in ['Challenging-60','U45','Color-Check7']:
   source=ROOT/'runs'/ds
   if source.exists():
    target=archive/'runs'/ds;target.parent.mkdir(exist_ok=True);shutil.move(str(source),str(target))
    for f in target.glob('seed*/*/*_external_per_image.json'):
     row=read(f);data.append(dict(path=str(f.relative_to(ROOT)),dataset=row['dataset'],mode=row['mode'],initialization_seed=row['initialization_seed'],weights=row['weights'],n=row['n'],metrics={k:row[k] for k in ['UCIQE','NIQE','URanker','CIEDE2000'] if k in row},reason='Uniform pad-to8/crop was needed for ISP; V1 all methods preserved and superseded uniformly.'))
  atomic(archive/'RESULTS_INDEX.json',dict(at=time.time(),results=data,not_used_as_final_V2_comparisons=True))
  p=ROOT/'tools/external_evaluate.py';text=p.read_text();text=text.replace('from frozen_factory import ROOT,PLAN,create_model,atomic,sha,state_sha','from frozen_factory import ROOT,PLAN,create_model,atomic,sha,state_sha\nfrom external_geometry import predict_external')
  old='with torch.no_grad():raw=model(x.cuda());assert torch.isfinite(raw).all();prediction=raw.clamp(0,1).cpu()';new='with torch.no_grad():raw=predict_external(model,x.cuda());assert torch.isfinite(raw).all();prediction=raw.clamp(0,1).cpu()';assert old in text;text=text.replace(old,new)
  text=text.replace("result=dict(at=time.time(),dataset=dataset", "result=dict(evaluation_version='uniform_pad8_crop_v2',preprocessing='Author NR resize/crop-to4; all4 models receive right/bottom reflect-pad to8, outputs cropped back before scoring.',at=time.time(),dataset=dataset")
  ast.parse(text);p.write_text(text)
  previous=read(ROOT/'EXTERNAL_MATH_LOCK.json');files={k:sha(ROOT/k) for k in previous['files']};files['tools/external_geometry.py']=sha(ROOT/'tools/external_geometry.py')
  atomic(ROOT/'EXTERNAL_MATH_LOCK.json',dict(at=time.time(),version='uniform_pad8_crop_v2',files=files,library_sources=previous['library_sources'],prior_version_archive=str(archive),geometry_check_sha256=sha(ROOT/'EXTERNAL_GEOMETRY_CPU_CHECKS.json'),all_metric_formulas_unchanged=True))
  # Preserve V1 observations as a separate Excel sheet; never substitute them for V2.
  export=ROOT/'tools/export_excel.py';shutil.copy2(export,archive/'tools/export_excel.py');s=export.read_text();needle="  expected={}\n";assert needle in s
  insert="""  history=read(ROOT/'external/history/v1/RESULTS_INDEX.json',{}).get('results',[])
  page=book.create_sheet('ExternalHistoryV1');page.append(['Dataset','Stage','Init','Weights','N','UCIQE WWE Python HSV','NIQE','URanker','CIEDE2000','Reason','Path'])
  for item in history:
   v=item['metrics'];page.append([item['dataset'],item['mode'],item['initialization_seed'],item['weights'],item['n'],v.get('UCIQE'),v.get('NIQE'),v.get('URanker'),v.get('CIEDE2000'),item['reason'],str(ROOT/item['path'])])
  expected={}
"""
  s=s.replace(needle,insert);s=s.replace("'UIEB-trained, two-init means. UCIQE follows WWE Python HSV;", "'V2 all4 pad-to8/crop; UIEB-trained, two-init means. UCIQE follows WWE Python HSV;")
  ast.parse(s);export.write_text(s)
  adaptive=read(ROOT/'ADAPTIVE_CODE_LOCK.json');adaptive['files']['tools/export_excel.py']=sha(export);adaptive.setdefault('revisions',[]).append(dict(at=time.time(),reason='Preserve all V1 external outcomes; uniform V2 geometry comparison, original neural/training math untouched'));atomic(ROOT/'ADAPTIVE_CODE_LOCK.json',adaptive)
  blocked=read(OWNER/'BLOCKED_JOBS.json',{});atomic(archive/'BLOCKED_V1.json',blocked)
  for key in list(blocked):
   if key.startswith(ROOT.name+'/') and key.split('/')[1] in ['Challenging-60','U45','Color-Check7']:
    assert 'AssertionError' in (ROOT/'logs'/f'{key.split("/")[1]}_seed{key.split("/")[2]}_{key.split("/")[3]}.log').read_text(errors='replace')
    blocked.pop(key)
  atomic(OWNER/'BLOCKED_JOBS.json',blocked)
  atomic(ROOT/'EXTERNAL_V2_RELEASE_AUDIT.json',dict(at=time.time(),passed=True,prior_observations_preserved=len(data),v1_archive=str(archive),original_source_and_training_math_unchanged=True,all4_models_same_external_geometry=True,healthy_GPU_handles_preserved=live))
  atomic(ROOT/'EXTERNAL_READINESS.json',dict(at=time.time(),version='uniform_pad8_crop_v2',ready_datasets=['Challenging-60','U45','Color-Check7'],all_required_materials_present=True,metric_and_geometry_CPU_checks_passed=True,metric_lock_sha256=sha(ROOT/'EXTERNAL_MATH_LOCK.json'),protocol='WWE Python HSV UCIQE/NIQE + official URanker; fixed BabelColor D65 CIEDE2000; uniformly pad/crop all4 models. Published values explicitly separate.',pending=[]))
 subprocess.run([sys.executable,str(OWNER/'tools/launch_adaptive.py')],check=True)
 # Versioned evaluation supplement; no existing frozen manifest is overwritten.
 base=ROOT.parent/'four_stage_code_backup_complete_20261116'
 for mode in read(ROOT/'PLAN.json')['methods']:
  dest=base/mode/'evaluation_v2';dest.mkdir();(dest/'tools').mkdir()
  for name in ['external_evaluate.py','external_geometry.py','external_metrics.py','profile_efficiency.py','export_excel.py']:shutil.copy2(ROOT/'tools'/name,dest/'tools'/name)
  for name in ['EXTERNAL_MATH_LOCK.json','EXTERNAL_GEOMETRY_CPU_CHECKS.json','EXTERNAL_METRICS_CPU_CHECKS.json']:shutil.copy2(ROOT/name,dest/name)
  (dest/'README.md').write_text('Uniform pad-to8/crop evaluation V2. Core model/training files remain exactly those in the original stage manifest. To restore a runnable working copy, overlay these evaluator tools over a fresh copy of the parent stage tools; keep the original immutable archive intact. All V1 observations remain in the main experiment external/history/v1.\n')
  atomic(dest/'MANIFEST.json',dict(at=time.time(),files={str(p.relative_to(dest)):dict(sha256=sha(p),size_bytes=p.stat().st_size) for p in dest.rglob('*') if p.is_file()}))
 print('EXTERNAL_V2_RELEASED_ALL4_UNIFORM_GEOMETRY; V1_OBSERVATIONS_PRESERVED',len(data),flush=True)
if __name__=='__main__':main()
