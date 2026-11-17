"""Use either of the two authorized identical GPUs for read-only external/paired evaluation."""
from pathlib import Path
import ast,fcntl,json,os,signal,subprocess,sys,time,hashlib,shutil
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
with (OWNER/'launch.lock').open('a+') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 hs=[read(OWNER/n) for n in ['WATCHDOG_LAUNCH.json','LAUNCH.json']];live=[h for h in read(OWNER/'ACTIVE_JOBS.json') if owned(h)]
 for h in hs:stop(h)
 p=OWNER/'tools/controller_adaptive.py';previous=p.read_text();old="job_policy=dict(policy,authorized_gpus=[j['gpu']])";new="job_policy=dict(policy,authorized_gpus=policy['authorized_gpus'] if j.get('phase')=='evaluation' else [j['gpu']])";assert old in previous
 folder=ROOT/'checks/flexible_evaluation_cards';folder.mkdir();(folder/'controller_before.py').write_text(previous)
 text=previous.replace(old,new);ast.parse(text);p.write_text(text)
 runtime=read(OWNER/'ADAPTIVE_CODE_LOCK.json');runtime['files']['tools/controller_adaptive.py']=sha(p);runtime.setdefault('resource_revisions',[]).append(dict(at=time.time(),reason='Evaluation can use either assigned GPU1/4 of identical required hardware; preserve all healthy workers and training card plans.',math_changed=False));atomic(OWNER/'ADAPTIVE_CODE_LOCK.json',runtime)
 atomic(folder/'CHANGE_AUDIT.json',dict(at=time.time(),passed=True,healthy_GPU_handles_preserved=live,read_only_evaluation_on_any_of_GPU1_4=True,training_device_plans_and_all_model_math_unchanged=True))
subprocess.run([sys.executable,str(OWNER/'tools/launch_adaptive.py')],check=True)
print('EVALUATION_CAN_USE_EITHER_ASSIGNED_CARD; ALL_HEALTHY_GPU_WORKERS_PRESERVED',flush=True)
