"""Preserve failed admissions; fix prerequisite setup and diagnose a memory-starved preflight."""
import ast,fcntl,importlib.util,json,os,shutil,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OWNER=ROOT.parent/'dino_factorized_cross_stage_20261110'
def read(p,d=None):return json.loads(p.read_text()) if p.exists() else d
def sha(p):import hashlib;return hashlib.sha256(p.read_bytes()).hexdigest()
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
 locks=read(ROOT/'WEIGHTS_LOCK.json');assert len(locks)>=14
 folder=ROOT/'checks/admission_repair';folder.mkdir()
 with (OWNER/'launch.lock').open('a+') as launch_lock:
  fcntl.flock(launch_lock,fcntl.LOCK_EX)
  previous=[read(OWNER/name) for name in ['WATCHDOG_LAUNCH.json','LAUNCH.json']];live=[h for h in read(OWNER/'ACTIVE_JOBS.json',[]) if owned(h)]
  for h in previous:stop(h)
  blocked=read(OWNER/'BLOCKED_JOBS.json',{});atomic(folder/'BLOCKED_BEFORE.json',blocked)
  p=OWNER/'tools/controller_adaptive.py';shutil.copy2(p,folder/'controller_before.py');s=p.read_text()
  needle="    resource=(code<0 or error and (error['type']=='OutOfMemoryError' or\n"
  replacement="    prior_resource=read(ROOT/'RETRY_STATE.json',{}).get(key(j),{}).get('resource_failures',0)\n    init_memory_failure=(error and 'CUDNN_STATUS_NOT_INITIALIZED' in error.get('message','') and prior_resource>0 and not (dest(j)/'last_full.pth').exists())\n    resource=(init_memory_failure or code<0 or error and (error['type']=='OutOfMemoryError' or\n"
  assert needle in s;s=s.replace(needle,replacement);ast.parse(s);p.write_text(s)
  runtime=read(OWNER/'ADAPTIVE_CODE_LOCK.json');runtime['files']['tools/controller_adaptive.py']=sha(p);runtime.setdefault('resource_revisions',[]).append(dict(at=time.time(),reason='Classify cuDNN initial-context failure only with prior same-run OOM and no full epoch as resource; no mathematical training changes.'))
  atomic(OWNER/'ADAPTIVE_CODE_LOCK.json',runtime)
  sys.path.insert(0,str(OWNER/'tools'));import controller_adaptive as controller
  removed=[]
  for k,entry in list(blocked.items()):
   if not k.startswith(ROOT.name+'/'):continue
   _,dataset,seed,mode=k.split('/');n=int(seed);dest=ROOT/'runs'/dataset/f'seed{n}'/mode
   if dataset in ['Challenging-60','U45','Color-Check7']:
    log=ROOT/'logs'/f'{dataset}_seed{n}_{mode}.log';text=log.read_text(errors='replace')
    assert 'FileNotFoundError' in text and 'WEIGHTS_LOCK.json' in text
    assert not list(dest.glob('*_external_per_image.json'))
    if (dest/'EVALUATION_ERROR.json').exists():shutil.move(str(dest/'EVALUATION_ERROR.json'),str(folder/f'{dataset}_{n}_{mode}_EVALUATION_ERROR.json'))
    removed.append(dict(key=k,reason='Missing initial WEIGHTS_LOCK before any inference;14 existing Val-selected runs now locked. No model/metric/source changed.'));blocked.pop(k)
   elif dataset=='EUVP-Scene' and mode=='dino_cross_stage' and n==7:
    # This exact attempt had an earlier documented OOM on the same job/card,
    # then failed while initializing the first frozen-teacher convolution.
    state=read(OWNER/'RETRY_STATE.json')[k];assert state['resource_failures']>0
    assert entry['error']['type']=='RuntimeError' and 'CUDNN_STATUS_NOT_INITIALIZED' in entry['error']['message']
    assert not (dest/'last_full.pth').exists()
    job=dict(dataset=dataset,mode=mode,init_seed=n,gpu=1,experiment_root=str(ROOT),phase='pipeline')
    assert controller.recover(job,1,read(OWNER/'SCHEDULER_POLICY.json'))
    removed.append(dict(key=k,reason='Same job prior OOM + no committed epoch + frozen-teacher initialization context; actual exit1 preserved. Same fixed init retry after safe capacity.'));blocked.pop(k)
  atomic(OWNER/'BLOCKED_JOBS.json',blocked)
  atomic(folder/'REPAIR_AUDIT.json',dict(at=time.time(),passed=True,diagnosed_and_requeued=removed,all_healthy_GPU_handles_preserved=live,CPU_handles=previous,no_training_or_metric_math_changed=True,failed_logs_retained=True))
 subprocess.run([sys.executable,str(OWNER/'tools/launch_adaptive.py')],check=True)
 print('ADMISSION_PREREQUISITES_REPAIRED_AND_FAILED_ATTEMPTS_PRESERVED',len(removed),flush=True)
if __name__=='__main__':main()
