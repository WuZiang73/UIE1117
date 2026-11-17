"""Low-frequency observer; no GPU tasks are restarted by this process."""
import json,os,subprocess,time,traceback
from pathlib import Path
from reporting_protocol import effective_plan
ROOT=Path(__file__).resolve().parents[1];OWNER=ROOT.parent/'dino_factorized_cross_stage_20261110'
def read(p,d=None):
 try:return json.loads(p.read_text())
 except FileNotFoundError:return d
def atomic(p,x):q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');q.replace(p)
def owned(h):
 try:
  p=Path('/proc')/str(h['pid']);st=p.joinpath('stat').read_text().rsplit(')',1)[1].split();args=p.joinpath('cmdline').read_bytes().decode().rstrip('\0').split('\0')
  return p.stat().st_uid==os.getuid() and st[0]!='Z' and st[19]==str(h['start_ticks']) and args==h['args']
 except (FileNotFoundError,ProcessLookupError):return False
def observe():
 plan=effective_plan(read(ROOT/'PLAN.json'));registered=read(OWNER/'ACTIVE_JOBS.json',[]);live=[h for h in registered if h.get('experiment_root')==str(ROOT) and owned(h)]
 trained=evaluated=0;rows=[]
 for j in plan['predeclared_job_order']:
  folder=ROOT/'runs'/j['dataset']/f'seed{j["init_seed"]}'/j['mode'];progress=read(folder/'PROGRESS.json',{});done=(folder/'completed.json').exists();tested=(folder/'EVALUATED.json').exists();trained+=done;evaluated+=tested
  h=next((x for x in live if x['dataset']==j['dataset'] and x['mode']==j['mode'] and x['init_seed']==j['init_seed']),None)
  rows.append(dict(**j,trained=done,evaluated=tested,epoch=progress.get('epoch',0),PID=h['pid'] if h else None,PID_handle_verified=bool(h)))
 external=sum((ROOT/'runs'/j['dataset']/f'seed{j["init_seed"]}'/j['mode']/f'{w}_external_per_image.json').exists() for j in plan['external_job_order'] for w in ['raw','ema'])
 blocked={k:v for k,v in read(OWNER/'BLOCKED_JOBS.json',{}).items() if k.startswith(ROOT.name+'/')}
 cards=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=10)
 result=dict(at=time.time(),observer_pid=os.getpid(),interval_seconds=1200,trained=trained,paired_evaluated=evaluated,total_required=plan['total_jobs'],original_declared40=True,imported=14,new_trainings_required=plan['total_jobs']-14,
  real_live_workers=live,live_count=len(live),external_evaluations=external,external_required=48,external_readiness=read(ROOT/'EXTERNAL_READINESS.json'),blocked=blocked,jobs=rows,GPU_actual_snapshot=cards,
  all_complete=(ROOT/'COMPLETE.json').exists(),goal_completion_still_requires_independent_full_scope_audit=True)
 atomic(ROOT/'LATEST_STATUS.json',result)
 text=f'# 四阶段全数据集进度\n\n更新时间Unix {result["at"]}；每20分钟一次。\n\n配对训练 {trained}/{plan["total_jobs"]}，其中复用14项；配对评测{evaluated}/{plan["total_jobs"]}；外部评测{external}/48；真实核验worker {len(live)}。\n\n| 数据 | 模型 | 初值 | 正式epoch | PID | 训练完 | 评测完 |\n|---|---|---:|---:|---:|---|---|\n'
 for row in rows:text+=f'|{row["dataset"]}|{row["mode"]}|{row["init_seed"]}|{row["epoch"]}|{row["PID"] or ""}|{row["trained"]}|{row["evaluated"]}|\n'
 text+='\nGPU快照包含第三方，不代表全是本方占用。PID均按UID/start_ticks/argv核验；observer不修改训练、不重启GPU任务。\n'
 (ROOT/'LATEST_STATUS.md').write_text(text);print('OBSERVED',trained,evaluated,external,'live',len(live),'blocked',len(blocked),flush=True)
 return result['all_complete']
def main():
 while True:
  try:
   if observe():break
  except Exception as exc:print('OBSERVER_RETRY',repr(exc),traceback.format_exc(),flush=True)
  time.sleep(1200)
if __name__=='__main__':main()
