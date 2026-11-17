"""Checkpoint owned CLUIE, then add all forty paired runs and external tasks to the single owner."""
import ast,fcntl,hashlib,importlib.util,json,os,shutil,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OWNER=ROOT.parent/'dino_factorized_cross_stage_20261110'
def read(p,d=None):return json.loads(p.read_text()) if p.exists() else d
def write(p,x):q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');q.replace(p)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def owned(h):
 try:
  p=Path('/proc')/str(h['pid']);s=(p/'stat').read_text().rsplit(')',1)[1].split();args=(p/'cmdline').read_bytes().decode().rstrip('\0').split('\0')
  return p.stat().st_uid==os.getuid() and s[0]!='Z' and s[19]==str(h['start_ticks']) and args==h['args']
 except (FileNotFoundError,KeyError,ProcessLookupError):return False
def stop(h):
 if not owned(h):return dict(status='not_live',handle=h)
 os.kill(h['pid'],signal.SIGTERM)
 for _ in range(100):
  if not owned(h):return dict(status='stopped_exact_owned_CPU',handle=h)
  time.sleep(.1)
 raise RuntimeError('Exact owned CPU supervisor did not exit')
def main():
 assert read(ROOT/'PREPARED.json')['passed']
 assert not (ROOT/'ACTIVATION.json').exists()
 assert read(ROOT.parent/'gpu24_dispatch_20261104/CURRENT_WWE_OWNER.json')['root']==str(OWNER)
 for filename in ['CONFIG_LOCK.json','ADAPTIVE_CODE_LOCK.json']:
  for name,digest in read(ROOT/filename)['files'].items():assert sha(ROOT/name)==digest
 folder=ROOT/'checks/coordinator_handoff';folder.mkdir(parents=True);(ROOT/'cluie_handoff').mkdir()
 with (OWNER/'launch.lock').open('a+') as launch_lock:
  fcntl.flock(launch_lock,fcntl.LOCK_EX)
  with (OWNER/'cluie_gpu5/lifecycle.lock').open('a+') as filler_lock:
   fcntl.flock(filler_lock,fcntl.LOCK_EX)
   stopped=[stop(read(OWNER/name,{})) for name in ['WATCHDOG_LAUNCH.json','LAUNCH.json']]
   assert not any(owned(h) for h in read(OWNER/'ACTIVE_JOBS.json',[])),'Keep healthy WWE; unexpected active run requires adoption review'
   # Preserve exact old resource code/locks before modifying only scheduler orchestration.
   for name in ['tools/controller_adaptive.py','SCHEDULER_POLICY.json','SUPPLEMENTARY_WWE_QUEUE.json','ADAPTIVE_CODE_LOCK.json']:
    p=folder/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(OWNER/name,p)
   path=OWNER/'tools/controller_adaptive.py';text=path.read_text()
   text=text.replace("assert d in job_plan['datasets'] and m in job_plan['methods']", "assert d in job_plan['datasets']+job_plan.get('external_datasets',[]) and m in job_plan['methods']")
   text=text.replace("all_jobs.extend(dict(job,experiment_root=str(folder)) for job in read(folder/'PLAN.json')['predeclared_job_order'])", "job_plan=read(folder/'PLAN.json')\n            all_jobs.extend(dict(job,experiment_root=str(folder)) for job in job_plan['predeclared_job_order'])\n            ready=set(read(folder/'EXTERNAL_READINESS.json',{}).get('ready_datasets',[]))\n            all_jobs.extend(dict(job,experiment_root=str(folder)) for job in job_plan.get('external_job_order',[]) if job['dataset'] in ready)")
   a=text.index('def supplementary_work():');b=text.index('\ndef verify_locks():',a)
   replacement='''def supplementary_work():
    work=[]
    for folder in supplementary_roots():
        if (folder/'COMPLETE.json').exists():continue
        plan=read(folder/'PLAN.json')
        assert read(folder/'ACTIVATION.json')['authorized_after_complete_first_batch_review']
        for filename in ['CONFIG_LOCK.json','ADAPTIVE_CODE_LOCK.json']:
            for name,signature in read(folder/filename)['files'].items():assert sha(folder/name)==signature,(folder,name)
        jobs=[dict(job,experiment_root=str(folder)) for job in plan['predeclared_job_order']]
        if plan.get('immediate_evaluation'):
            for job in jobs:
                if (dest(job)/'completed.json').exists():
                    work.append(dict(job,phase='evaluation',evaluation_scope='all'))
                else:work.append(job)
        elif all((dest(job)/'completed.json').exists() for job in jobs):
            if not (folder/'METHOD_SELECTION.json').exists():
                subprocess.run([sys.executable,str(folder/'tools/select_method.py')],check=True,env=ENV)
            selection=read(folder/'METHOD_SELECTION.json')
            work.extend(dict(job,phase='evaluation',evaluation_scope='all' if job['mode'] in selection['test_modes'] else 'val_only') for job in jobs)
        else:work.extend(jobs)
        ready=set(read(folder/'EXTERNAL_READINESS.json',{}).get('ready_datasets',[]))
        for job in plan.get('external_job_order',[]):
            source=folder/'runs'/job.get('source_dataset','UIEB')/f"seed{job['init_seed']}"/job['mode']
            if job['dataset'] in ready and (source/'EVALUATED.json').exists():
                work.append(dict(job,experiment_root=str(folder)))
    return work

'''
   text=text[:a]+replacement+text[b:]
   old="budget=4500 if j.get('phase')=='evaluation' else max(7500 if j['mode']=='isp_control' else policy['default_job_budget_mib'],int(profiled_peak/1048576)+1100)"
   new="initial_budget=experiment_plan(j).get('initial_job_budget_mib_by_mode',{}).get(j['mode'],7500 if j['mode']=='isp_control' else policy['default_job_budget_mib'])\n                    budget=4500 if j.get('phase')=='evaluation' else max(initial_budget,int(profiled_peak/1048576)+1100)"
   assert old in text;text=text.replace(old,new)
   old="                    if all(complete(job) for job in jobs) and not any(experiment_root(job)==folder for job in active):\n                        subprocess.run([sys.executable,str(folder/'tools/finalize.py')],env=ENV,check=True)"
   new="                    external_jobs=[dict(job,experiment_root=str(folder)) for job in read(folder/'PLAN.json').get('external_job_order',[])]\n                    if all(complete(job) for job in jobs+external_jobs) and not any(experiment_root(job)==folder for job in active):\n                        subprocess.run([sys.executable,str(folder/'tools/finalize.py')],env=ENV,check=True)"
   assert old in text;text=text.replace(old,new)
   ast.parse(text);path.write_text(text)
   policy=read(OWNER/'SCHEDULER_POLICY.json');policy.update(max_parallel_per_gpu=5,priority_reason='Complete four fixed cumulative versions and all Table1/Table2 data. Stage-specific memory budgets use accepted first-epoch profiles; no training math changed.')
   plan=read(ROOT/'PLAN.json');priority=[f'{ROOT.name}/{j["dataset"]}/{j["init_seed"]}/{j["mode"]}' for j in plan['predeclared_job_order']]
   policy['job_priority']=priority+policy.get('job_priority',[]);write(OWNER/'SCHEDULER_POLICY.json',policy)
   queue=read(OWNER/'SUPPLEMENTARY_WWE_QUEUE.json');assert str(ROOT) not in queue['roots'];queue['roots'].append(str(ROOT));write(OWNER/'SUPPLEMENTARY_WWE_QUEUE.json',queue)
   runtime=read(OWNER/'ADAPTIVE_CODE_LOCK.json')
   for name in ['tools/controller_adaptive.py','SCHEDULER_POLICY.json','SUPPLEMENTARY_WWE_QUEUE.json']:runtime['files'][name]=sha(OWNER/name)
   runtime.setdefault('resource_revisions',[]).append(dict(at=time.time(),reason='User requests four stage backups and all Table1/Table2 training/evaluation; immediate completed-run Test, external task admission and completion gate.',math_source_changed=False,GPU_training_engine_changed=False));write(OWNER/'ADAPTIVE_CODE_LOCK.json',runtime)
   # Checkpoint/pause only exact registered owned filler process groups.
   spec=importlib.util.spec_from_file_location('safe_cluie_handoff',ROOT.parent/'baseline_c0_isp_ssim_ablation_20261109/tools/handoff.py');handoff=importlib.util.module_from_spec(spec);spec.loader.exec_module(handoff);handoff.ROOT=ROOT
   registry=read(OWNER/'cluie_gpu5/ACTIVE.json',{});write(ROOT/'cluie_handoff/ORIGINAL_REGISTRY.json',registry);paused=[]
   for name,h in registry.items():
    assert h['physical_gpu'] in [1,4]
    for _ in range(15):
     value=handoff.pause_one(name,h)
     if value is not None:paused.append(value);break
     time.sleep(2)
    else:raise RuntimeError(f'No valid complete checkpoint for owned CLUIE{name}; keep evidence')
   write(ROOT/'cluie_handoff/COMPLETE.json',dict(at=time.time(),passed=True,paused=paused,third_party_untouched=True,CPU_supervisors=stopped))
   write(ROOT/'ACTIVATION.json',dict(at=time.time(),authorized_after_complete_first_batch_review=True,scope='All4 cumulative versions x5paired plus all3 Table2 externals',coordinator_root=str(OWNER),training_plan_SHA=sha(ROOT/'PLAN.json'),cards=[1,4],max_cards=2))
   write(folder/'ACTIVATION_AUDIT.json',dict(passed=True,coordinator_resource_only_revision=True,all_GPU_math_sources_unchanged=True,old_CPU_handles=stopped))
 # The existing cron and idempotent launcher retain ownership and restart exactly one controller.
 subprocess.run([sys.executable,str(OWNER/'tools/launch_adaptive.py')],check=True)
 print('FOUR_STAGE26_NEW_TRAININGS_AND24_EXTERNAL_TASKS_ADMITTED_TO_SINGLE_OWNER',flush=True)
if __name__=='__main__':main()
