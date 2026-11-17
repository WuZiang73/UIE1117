"""Preserve the accepted optimizer/EMA/loss math; generalize dataset references and immediate evaluation."""
from pathlib import Path
import ast,json
ROOT=Path(__file__).resolve().parents[1];PARENT=ROOT.parent/'dino_rms_direction_20261110'
def main():
 text=(PARENT/'tools/experiment.py').read_text()
 text=text.replace("data_reference = json.loads((ROOT / 'references' / f'{dataset}_seed7_TRACE.json').read_text())['steps']\n    assert len(data_reference) == per_epoch * PLAN['epochs']", "reference_path = ROOT / 'references' / f'{dataset}_seed7_TRACE.json'\n    data_reference = json.loads(reference_path.read_text())['steps'] if reference_path.exists() else []\n    assert not data_reference or len(data_reference) == per_epoch * PLAN['epochs']")
 text=text.replace("old_control=json.loads((ROOT/'references/historical'/dataset/f'seed{number}/isp_control/TRACE.json').read_text())['steps'] if mode=='isp_control' else None", "control_path=ROOT/'references/historical'/dataset/f'seed{number}/isp_control/TRACE.json'\n    old_control=json.loads(control_path.read_text())['steps'] if mode=='isp_control' and control_path.exists() else None")
 text=text.replace("execution_revision='dino_cross_stage_v1'","execution_revision='four_cumulative_stages_v1'")
 a=text.index('def evaluate_completed(');b=text.index("\n\nif __name__=='__main__':",a)
 text=text[:a]+'''def evaluate_completed(dataset,mode,number):
    # All four architectures are fixed before this new batch. Lock each full-Val
    # best before its Test; completed runs need not wait for other datasets.
    from select_method import select
    select()
    configure()
    for weight in ['raw','ema']:
        for split in ['val','test']:
            if not (run_path(dataset,mode,number)/f'{weight}_{split}_per_image.json').exists():
                evaluate(dataset,mode,number,split,weight,configured=True)
    atomic(run_path(dataset,mode,number)/'EVALUATED.json',dict(at=time.time(),splits=4,Test_authorized=True,immediate_after_training=True))
'''+text[b:]
 text=text.replace("choices=PLAN['datasets']", "choices=PLAN['datasets']+PLAN['external_datasets']")
 text=text.replace("choices=['val','test','all','val_only']", "choices=['val','test','all','val_only','external']")
 text=text.replace("    if args.pipeline:\n", "    if args.evaluate=='external':\n        from external_evaluate import evaluate_external\n        evaluate_external(args.dataset,args.mode,args.init_seed)\n    elif args.pipeline:\n")
 old=ast.parse((PARENT/'tools/experiment.py').read_text());new=ast.parse(text)
 fixed=['configure','tensor_sha','key','run_path','save_torch','copy_state','compare_epoch_inputs','barrier','validate','evaluate']
 def node(tree,name):return next(x for x in tree.body if isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef)) and x.name==name)
 for name in fixed:assert ast.dump(node(old,name),include_attributes=False)==ast.dump(node(new,name),include_attributes=False),name
 # The optimizer/loss/backprop/update loop must remain verbatim at the AST level.
 def optimizer_loop(tree):
  fun=node(tree,'train')
  return next(x for x in ast.walk(fun) if isinstance(x,ast.For) and isinstance(x.target,ast.Tuple) and [getattr(y,'id',None) for y in x.target.elts]==['x','y','names'])
 assert ast.dump(optimizer_loop(old),include_attributes=False)==ast.dump(optimizer_loop(new),include_attributes=False)
 (ROOT/'tools/experiment.py').write_text(text)
 (ROOT/'checks/ENGINE_INHERITANCE.json').write_text(json.dumps(dict(passed=True,unchanged_helpers=fixed,optimizer_backprop_EMA_step_loop_AST_exact=True,
  changes=['optional historical reference on newly introduced datasets','four explicit mode labels in factory','immediate per-run Val lock and Raw/EMA Val/Test','separate external evaluation dispatch'],parent=str(PARENT)),indent=2)+'\n')
 print('ENGINE_MATH_INHERITANCE_PASSED',flush=True)
if __name__=='__main__':main()
