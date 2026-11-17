"""Versioned reporting/completion scope; never mutates the frozen training PLAN."""
import copy
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def effective_plan(original):
    plan=copy.deepcopy(original)
    path=ROOT/'TRAINING_SEED_POLICY.json'
    if not path.exists():
        plan['required_initializations_by_dataset']={d:list(plan['initialization_seeds']) for d in plan['datasets']}
        plan['retired_jobs']=[]
        return plan
    policy=json.loads(path.read_text())
    required=set(policy['required_job_keys'])
    key=lambda j:f"{j['dataset']}/{j['init_seed']}/{j['mode']}"
    assert required.issubset({key(j) for j in original['predeclared_job_order']})
    plan['predeclared_job_order']=[j for j in original['predeclared_job_order'] if key(j) in required]
    plan['original_declared_total']=original['total_jobs']
    plan['total_jobs']=len(plan['predeclared_job_order'])
    plan['required_initializations_by_dataset']=policy['required_initializations_by_dataset']
    plan['retired_jobs']=policy['retired_unstarted_jobs']
    plan['primary_reporting_initialization']=7
    plan['seed_selection_after_observed_UIEB_Test']=True
    plan['seed_policy_version']=policy['version']
    plan['expected_weight_files']=plan['total_jobs']*9
    plan['expected_model_states']=plan['total_jobs']*10
    plan['expected_evaluations']=plan['total_jobs']*4
    plan['expected_curve_rows']=plan['total_jobs']*plan['epochs']
    return plan
