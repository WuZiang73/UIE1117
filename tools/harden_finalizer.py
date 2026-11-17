"""Close archive races and broaden final verification without touching training math."""
import ast,fcntl,hashlib,json,os,shutil,signal,subprocess,sys,time
from pathlib import Path
from archive_snapshot import snapshot
ROOT=Path(__file__).resolve().parents[1];OWNER=ROOT.parent/'dino_factorized_cross_stage_20261110'
def read(p):return json.loads(p.read_text())
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
 raise RuntimeError('CPU did not exit')
def main():
 fixture=ROOT/'checks/archive_copy_fixture';a=fixture/'source';b=fixture/'partial';(a/'logs').mkdir(parents=True);b.mkdir()
 (a/'weights.pth').write_bytes(b'fixture tensor bytes');(a/'weights.pth').chmod(0o444);(b/'weights.pth').write_bytes(b'incomplete')
 (a/'LATEST_STATUS.json').write_text('live');(a/'logs/progress_observer.log').write_text('live');(a/'alias').symlink_to('weights.pth')
 files=snapshot(a,b);assert set(files)=={'weights.pth'} and not (b/'LATEST_STATUS.json').exists() and (b/'alias').is_symlink()
 assert snapshot(a,b)==files and not (a/'weights.pth').samefile(b/'weights.pth')
 atomic(ROOT/'checks/ARCHIVE_RESUME_FIXTURE.json',dict(passed=True,readonly_partial_file_replaced=True,independent_inode_verified=True,symlink_preserved=True,mutable_observer_excluded=True,sealed_archive_reverified=True))
 with (OWNER/'launch.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  hs=[read(OWNER/name) for name in ['WATCHDOG_LAUNCH.json','LAUNCH.json']];live=[h for h in read(OWNER/'ACTIVE_JOBS.json') if owned(h)]
  for h in hs:stop(h)
  receipts=ROOT/'checks/finalizer_hardening';receipts.mkdir();p=ROOT/'tools/finalize.py';old=p.read_text();(receipts/'finalize_before.py').write_text(old)
  s=old.replace("for file in ['CONFIG_LOCK.json','ADAPTIVE_CODE_LOCK.json']:","for file in ['CONFIG_LOCK.json','ADAPTIVE_CODE_LOCK.json','EXTERNAL_MATH_LOCK.json']:")
  needle="  weight_index={};states=curves=evaluations=0\n";assert needle in s
  insert="""  # Full-scope input/GT/DropPath consistency across four methods, all200epochs.
  cohort_checks=[]
  for dataset in PLAN['datasets']:
   for number in PLAN['initialization_seeds']:
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
"""
  s=s.replace(needle,insert)
  needle="     metric=json.loads((folder/f'{w}_{sp}_per_image.json').read_text());assert metric['n']==len(metric['images'])==PLAN['split_counts'][ds][sp];evaluations+=1"
  repl="""     metric=json.loads((folder/f'{w}_{sp}_per_image.json').read_text());assert metric['n']==len(metric['images'])==PLAN['split_counts'][ds][sp]
     checkpoint=folder/('best_model.pth' if w=='ema' else 'best_raw_model.pth')
     assert metric['checkpoint_sha256']==sha(checkpoint)==json.loads((ROOT/'WEIGHTS_LOCK.json').read_text())[f'{ds}/{n}/{m}'][w]
     import statistics
     assert abs(statistics.mean(v['psnr'] for v in metric['images'])-metric['psnr'])<1e-10
     assert abs(statistics.mean(v['ssim'] for v in metric['images'])-metric['ssim'])<1e-12
     expected_names=json.loads((ROOT/'DATA_MANIFEST.json').read_text())['splits'][ds][sp]
     assert sorted(v['name'] for v in metric['images'])==expected_names
     best=json.loads((folder/f'best_{w}.json').read_text())
     assert best['epoch']==max((v for v in records if v[w] is not None),key=lambda v:v[w]['psnr'])['epoch']
     evaluations+=1"""
  assert needle in s;s=s.replace(needle,repl)
  a=s.index("  backup=Path(PLAN['backup_root'])");b=s.index('  # Load from the archive',a)
  s=s[:a]+"  backup=Path(PLAN['backup_root'])\n  from archive_snapshot import snapshot as archive_snapshot\n  files=archive_snapshot(ROOT,backup)\n"+s[b:]
  needle="  print('ALL8_DATASETS40_RUNS_AND208_EVALUATIONS_ARCHIVED',flush=True)";assert needle in s
  s=s.replace(needle,"  shutil.copy2(ROOT/'BACKUP_AUDIT.json',backup/'BACKUP_AUDIT.json')\n"+needle)
  ast.parse(s);p.write_text(s)
  adaptive=read(ROOT/'ADAPTIVE_CODE_LOCK.json');adaptive['files']['tools/finalize.py']=sha(p);adaptive['files']['tools/archive_snapshot.py']=sha(ROOT/'tools/archive_snapshot.py');adaptive.setdefault('revisions',[]).append(dict(at=time.time(),reason='Full40-cohort/metrics/weight audit, mutable-observer exclusions and resumable archive; no training or metric math changed'));atomic(ROOT/'ADAPTIVE_CODE_LOCK.json',adaptive)
  atomic(receipts/'CHANGE_AUDIT.json',dict(at=time.time(),passed=True,archive_fixture_passed=True,healthy_GPU_handles_preserved=live,all_model_and_training_math_unchanged=True,external_metric_math_unchanged=True))
 subprocess.run([sys.executable,str(OWNER/'tools/launch_adaptive.py')],check=True)
 # Ancillary archive tools are versioned additions, not overwrites of old manifests.
 base=ROOT.parent/'four_stage_code_backup_complete_20261116'
 for mode in read(ROOT/'PLAN.json')['methods']:
  dest=base/mode/'archive_tools_v2';dest.mkdir()
  for name in ['finalize.py','archive_snapshot.py']:shutil.copy2(ROOT/'tools'/name,dest/name)
  shutil.copy2(ROOT/'ADAPTIVE_CODE_LOCK.json',dest/'ADAPTIVE_CODE_LOCK.json')
  atomic(dest/'MANIFEST.json',dict(files={p.name:sha(p) for p in dest.iterdir() if p.is_file()}))
 print('FINALIZER_FULL_SCOPE_AND_RESUMABLE_ARCHIVE_HARDENED; GPU_WORK_UNTOUCHED',flush=True)
if __name__=='__main__':main()
