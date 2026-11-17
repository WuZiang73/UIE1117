"""Seal exact training math and refresh complete stage-specific code backups before GPU admission."""
import ast,json,shutil,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
from frozen_factory import sha,atomic

def main():
 for name in ['MODEL_CPU_CHECKS.json','RESUME_CPU_CHECKS.json','CODE_BACKUP_AUDIT.json','IMPORTED_REUSE_AUDIT.json','DATA_COPY_AUDIT.json','checks/ENGINE_INHERITANCE.json']:
  assert json.loads((ROOT/name).read_text())['passed'],name
 for f in (ROOT/'tools').glob('*.py'):ast.parse(f.read_text())
 plan=json.loads((ROOT/'PLAN.json').read_text());plan['initial_job_budget_mib_by_mode']={'baseline':7500,'c0':7500,'isp_control':7500,'dino_cross_stage':12000};atomic(ROOT/'PLAN.json',plan)
 # Source + mathematical trainer helpers freeze here. External evaluator math will
 # be independently locked before it is admitted, without changing these files.
 scientific=['PLAN.json','SOURCE_PROVENANCE.json','INITIALIZATIONS.json','TEACHER_PROVENANCE.json','DATA_MANIFEST.json']
 scientific += [str(p.relative_to(ROOT)) for p in (ROOT/'source').rglob('*') if p.is_file() and 'UnderWaterDataset' not in p.parts and '__pycache__' not in p.parts]
 scientific += [f'tools/{name}' for name in ['experiment.py','frozen_factory.py','base_equivalent_operators.py','strict_math.py','delayed_ema.py','restoration_losses.py']]
 scientific += ['initializations/native_seed7.pth','initializations/native_seed43.pth','teachers/vitb14.pth']
 atomic(ROOT/'CONFIG_LOCK.json',dict(at=time.time(),files={rel:sha(ROOT/rel) for rel in scientific},scientific_math_frozen=True))
 resources=['SCHEDULER_POLICY.json','tools/select_method.py','tools/export_excel.py','tools/finalize.py']
 atomic(ROOT/'ADAPTIVE_CODE_LOCK.json',dict(at=time.time(),files={rel:sha(ROOT/rel) for rel in resources}))
 audit=json.loads((ROOT/'CODE_BACKUP_AUDIT.json').read_text())
 for row in audit['stages']:
  dest=Path(row['path'])
  for f in (ROOT/'tools').glob('*.py'):
   target=dest/'tools'/f.name
   if target.exists():target.chmod(target.stat().st_mode|0o200)
   shutil.copy2(f,target)
  for name in ['PLAN.json','CONFIG_LOCK.json','ADAPTIVE_CODE_LOCK.json']:
   target=dest/name
   if target.exists():target.chmod(target.stat().st_mode|0o200)
   shutil.copy2(ROOT/name,target)
  files={}
  for f in sorted(dest.rglob('*')):
   if f.is_file() and 'UnderWaterDataset' not in f.parts and f.name!='MANIFEST.json':
    rel=str(f.relative_to(dest));p=ROOT/rel
    if p.exists() and f.name not in ['README.md','STAGE.json']:assert sha(f)==sha(p) and not f.samefile(p)
    files[rel]=dict(sha256=sha(f),size_bytes=f.stat().st_size)
  atomic(dest/'MANIFEST.json',dict(at=time.time(),files=files,own_stage=row['mode']))
  row['files']=len(files);row['final_frozen_training_tools_and_plan_copied']=True
 atomic(ROOT/'CODE_BACKUP_AUDIT.json',audit)
 external=ROOT.parent/'four_stage_code_backup_20261116';assert not external.exists()
 shutil.copytree(ROOT/'code_backups',external,symlinks=True)
 for row in audit['stages']:
  src=Path(row['path']);dest=external/row['mode'];manifest=json.loads((dest/'MANIFEST.json').read_text())
  for rel,entry in manifest['files'].items():assert sha(dest/rel)==entry['sha256'] and not (dest/rel).samefile(src/rel)
 atomic(external/'BACKUP_AUDIT.json',dict(at=time.time(),passed=True,four_stages=[x['mode'] for x in audit['stages']],all_files_SHA_and_independent_inodes_verified=True,training_math_lock_sha256=sha(ROOT/'CONFIG_LOCK.json')))
 atomic(ROOT/'SECOND_CODE_BACKUP_AUDIT.json',dict(passed=True,path=str(external),all4_stage_manifests_verified=True))
 atomic(ROOT/'PREPARED.json',dict(at=time.time(),passed=True,four_code_backups_and_second_independent_copy_verified=True,imports=14,new_trainings=26,total40=True,external_all3_required=True))
 print('ALL_TRAINING_MATH_FROZEN_AND4_STAGE_BACKUPS_PLUS_SECOND_COPY_COMPLETE',flush=True)
if __name__=='__main__':main()
