"""CPU-only independent backups for finished runs while the fixed GPU queue runs.

This supplements the final whole-experiment archive. It never mutates run files,
training code, metrics, scheduling policy, or existing sealed manifests.
"""
import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

from archive_snapshot import snapshot, sha
from reporting_protocol import effective_plan

ROOT = Path(__file__).resolve().parents[1]
BACKUP = ROOT.parent / 'four_stage_completed_runs_backup_20261116'
CODE = ROOT.parent / 'four_stage_code_backup_complete_20261116'


def read(path):
    return json.loads(path.read_text())


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


LOAD = r'''
import json,sys,torch
from pathlib import Path
torch.set_num_threads(1)
code,folder,mode,number=Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3],int(sys.argv[4])
sys.path.insert(0,str(code/'tools'))
from frozen_factory import create_model
model=create_model(mode,number,'cpu')
files=sorted(folder.rglob('*.pth'));assert len(files)==9
states=0
for path in files:
 value=torch.load(path,map_location='cpu')
 if path.name=='last_full.pth':
  assert value['epoch']==200
  assert all(k in value for k in ['optim','scheduler','torch_rng','cuda_rng','python_rng','numpy_rng','train_generator_rng'])
  values=[value['model'],value['ema']['model']]
 else:values=[value]
 for state in values:
  model.load_state_dict(state,strict=True)
  assert all(torch.isfinite(t).all() for t in state.values())
  states+=1
assert states==10
print(json.dumps(dict(passed=True,files_loaded=9,model_states_loaded=states,actual_backup_paths_loaded=True,full_resume_components_present=True)))
'''


def backup_once():
    plan = effective_plan(read(ROOT/'PLAN.json'))
    backed_up = []
    for job in plan['predeclared_job_order']:
        dataset, mode, number = job['dataset'], job['mode'], job['init_seed']
        relative = Path('runs')/dataset/f'seed{number}'/mode
        source, destination = ROOT/relative, BACKUP/relative
        if not (source/'EVALUATED.json').exists():
            continue
        code = CODE/mode
        locks = read(ROOT/'WEIGHTS_LOCK.json')
        # Read only immutable, completed trajectories. A snapshot can resume if
        # an earlier CPU copy was interrupted before its manifest was sealed.
        manifest = snapshot(source, destination)
        weight_files = [name for name in manifest if name.endswith('.pth')]
        assert len(weight_files) == 9, relative
        for weights, filename in [('raw','best_raw_model.pth'), ('ema','best_model.pth')]:
            digest = sha(destination/filename)
            assert locks[f'{dataset}/{number}/{mode}'][weights] == digest
            for split in ['val','test']:
                metric = read(destination/f'{weights}_{split}_per_image.json')
                assert metric['checkpoint_sha256'] == digest
                assert metric['n'] == len(metric['images']) == plan['split_counts'][dataset][split]
        receipt = destination/'MODEL_LOAD_AUDIT.json'
        if not receipt.exists():
            environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
            process = subprocess.run([sys.executable,'-c',LOAD,str(code),str(destination),mode,str(number)],
                                     check=True,capture_output=True,text=True,env=environment)
            result = json.loads(process.stdout.strip().splitlines()[-1])
            result.update(at=time.time(),code_backup=str(code),code_backup_manifest_sha256=sha(code/'MANIFEST.json'),
                          run_manifest_sha256=sha(destination/'MANIFEST.json'),dataset=dataset,mode=mode,initialization_seed=number,
                          original_run=str(source),imported=(source/'IMPORTED_PROVENANCE.json').exists())
            atomic(receipt,result)
            print('COMPLETED_RUN_BACKUP_LOADED',dataset,mode,number,flush=True)
        result = read(receipt)
        assert result['passed'] and result['files_loaded']==9 and result['model_states_loaded']==10
        assert result['run_manifest_sha256'] == sha(destination/'MANIFEST.json')
        assert result['code_backup_manifest_sha256'] == sha(code/'MANIFEST.json')
        backed_up.append(dict(dataset=dataset,mode=mode,init_seed=number,path=str(destination),
                              model_states_loaded=10,manifest_sha256=sha(destination/'MANIFEST.json')))
    result = dict(at=time.time(),passed_for_completed_runs=True,completed_runs_backed_up=len(backed_up),
                  expected_formal_runs=plan['total_jobs'],original_declared_runs=40,weight_files_independently_backed_up=9*len(backed_up),
                  model_states_actually_loaded_from_backup=10*len(backed_up),runs=backed_up,
                  code_backups=str(CODE),final_full_experiment_archive=plan['backup_root'],
                  final_full_archive_not_replaced_by_this_partial_backup=True,
                  scope='Completed run weights/metrics/config/trace/RNG/optimizer backup; datasets and full code are in their separately verified archives.')
    atomic(BACKUP/'BACKUP_INDEX.json',result)
    atomic(ROOT/'LATEST_COMPLETED_RUN_BACKUP_STATUS.json',result)
    print('COMPLETED_RUN_BACKUPS',len(backed_up),'/',plan['total_jobs'],flush=True)
    return len(backed_up)==plan['total_jobs']


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--watch',action='store_true');args=parser.parse_args()
    with (ROOT/'completed_backup.lock').open('a+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return
        while True:
            try:
                done=backup_once()
            except BaseException as exc:
                atomic(ROOT/'checks'/f'COMPLETED_BACKUP_ERROR_{time.time_ns()}.json',
                       dict(at=time.time(),type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
                raise
            if done or not args.watch:return
            time.sleep(1200)


if __name__=='__main__':
    main()
