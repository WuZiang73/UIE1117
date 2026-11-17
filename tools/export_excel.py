"""Merge every PDF Table1/Table2 row with all four cumulative-stage observations."""
import fcntl,hashlib,json,math,statistics,sys,time
from pathlib import Path
sys.modules['numpy']=None
from openpyxl import Workbook,load_workbook
from openpyxl.styles import Font,PatternFill
from reporting_protocol import effective_plan
ROOT=Path(__file__).resolve().parents[1]
def read(p,default=None):return json.loads(p.read_text()) if p.exists() else default
def main():
 with (ROOT/'excel.lock').open('a+') as lock:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:return
  plan=effective_plan(read(ROOT/'PLAN.json'));paper=read(ROOT/'paper/TABLE1_TABLE2_REFERENCE.json');book=Workbook();book.remove(book.active)
  headers={'Table1':['Source','Method','Weights','Init count']+[f'{d} {k}' for d in plan['datasets'] for k in ['PSNR','SSIM']]+['Params M','FLOPs G','Runtime ms','Notes'],
   'Table2':['Source','Method','Weights','Challenging-60 UCIQE','Challenging-60 NIQE','Challenging-60 URanker','U45 UCIQE','U45 NIQE','U45 URanker','Color-Check7 CIEDE2000','Notes'],
   'Runs':['Dataset','Stage','Init','Status','Imported from','Epoch','Raw best epoch','EMA best epoch','Raw Test PSNR','EMA Test PSNR','Raw Test SSIM','EMA Test SSIM','Path'],
   'External':['Dataset','Stage','UIEB init','Weights','Status','N','UCIQE','NIQE','URanker','CIEDE2000','Path'],
   'Curves':['Dataset','Stage','Init','Epoch','Loss','Gradient norm','LR','Raw Val PSNR','EMA Val PSNR','Raw Val SSIM','EMA Val SSIM'],
   'PerImage':['Source','Dataset','Stage','Init','Weights','Split','Image','PSNR','SSIM','UCIQE','NIQE','URanker','CIEDE2000'],
   'Weights':['Dataset','Stage','Init','Path','SHA256','Imported'],
   'CodeBackups':['Stage','Path','Verified','Files'],
   'Config':['Key','Value'], 'Provenance':['Item','Value']}
  pages={key:book.create_sheet(key) for key in headers}
  for key,h in headers.items():pages[key].append(h)
  for r in paper['table1']:
   pages['Table1'].append(['Published PDF',r['method'],'Paper',None]+[r['metrics'][d][k] for d in plan['datasets'] for k in ['psnr','ssim']]+[r['params_M'],r['FLOPs_G'],r['runtime_ms'],r['source']])
  for r in paper['table2']:
   pages['Table2'].append(['Published PDF',r['method'],'Paper']+[r['metrics'][d][k] for d in ['Challenging-60','U45'] for k in ['UCIQE','NIQE','URanker']]+[r['metrics']['Color-Check7']['CIEDE2000'],r['source']])
  grouped={};extgroup={};tested=trained=imports=0
  for job in plan['predeclared_job_order']:
   ds,m,n=job['dataset'],job['mode'],job['init_seed'];folder=ROOT/'runs'/ds/f'seed{n}'/m
   progress=read(folder/'PROGRESS.json',{});records=read(folder/'records.json',[]);done=(folder/'completed.json').exists();evaluated=(folder/'EVALUATED.json').exists()
   imported=read(folder/'IMPORTED_PROVENANCE.json',{});trained+=done;tested+=evaluated;imports+=bool(imported)
   metrics={w:read(folder/f'{w}_test_per_image.json',{}) for w in ['raw','ema']};best={w:read(folder/f'best_{w}.json',{}) for w in ['raw','ema']}
   pages['Runs'].append([ds,m,n,'Evaluated' if evaluated else 'Trained; evaluation pending' if done else f'Training epoch{progress["epoch"]}' if progress else 'Queued',imported.get('source_run'),len(records),best['raw'].get('epoch'),best['ema'].get('epoch'),metrics['raw'].get('psnr'),metrics['ema'].get('psnr'),metrics['raw'].get('ssim'),metrics['ema'].get('ssim'),str(folder)])
   for w in ['raw','ema']:
    if metrics[w]:grouped.setdefault((ds,m,w),[]).append(metrics[w])
    for split in ['val','test']:
     met=read(folder/f'{w}_{split}_per_image.json')
     if not met:continue
     assert met['n']==len(met['images'])==plan['split_counts'][ds][split]
     assert abs(statistics.mean(v['psnr'] for v in met['images'])-met['psnr'])<1e-10
     for image in met['images']:pages['PerImage'].append(['Observed',ds,m,n,w,split,image['name'],image['psnr'],image['ssim']])
   for r in records:
    e=r['ema'] or {};pages['Curves'].append([ds,m,n,r['epoch'],r['train_loss'],r.get('gradient_norm'),r.get('learning_rate'),r['raw']['psnr'],e.get('psnr'),r['raw']['ssim'],e.get('ssim')])
   for f in sorted(folder.rglob('*.pth')):
    digest=hashlib.sha256(f.read_bytes()).hexdigest();pages['Weights'].append([ds,m,n,str(f),digest,bool(imported)])
  efficiency=read(ROOT/'EFFICIENCY.json',{}).get('methods',{})
  for m in plan['methods']:
   for w in ['raw','ema']:
    vals=[];ns=[]
    for ds in plan['datasets']:
     observations=grouped.get((ds,m,w),[]);ns.append(len(observations))
     vals += [statistics.mean(r[k] for r in observations) if len(observations)==len(plan['required_initializations_by_dataset'][ds]) else None for k in ['psnr','ssim']]
    ef=efficiency.get(m,{})
    pages['Table1'].append(['Current protocol observed',plan['method_names'][m],w,str(dict(zip(plan['datasets'],ns)))]+vals+[ef.get('total_parameters_M'),ef.get('FLOPs_G'),ef.get('runtime_ms'),'Required-init summaries: prior datasets retain7/43 means; Dark uses7 only per user change after UIEB Test. All completed43 retained. Our FLOPs: profiler conv/matmul, multiply-add=2, includes frozen DINO; paper uses a different counter. Our ms: shared RTX4090; paper RTX3090. Published scores are not our reruns.'])
  extdone=0
  for job in plan['external_job_order']:
   ds,m,n=job['dataset'],job['mode'],job['init_seed'];folder=ROOT/'runs'/ds/f'seed{n}'/m
   for w in ['raw','ema']:
    path=folder/f'{w}_external_per_image.json';met=read(path,{})
    if met:
     extdone+=1;extgroup.setdefault((ds,m,w),[]).append(met)
     for image in met['images']:pages['PerImage'].append(['Observed',ds,m,n,w,'external',image['name'],None,None,image.get('UCIQE'),image.get('NIQE'),image.get('URanker'),image.get('CIEDE2000')])
    pages['External'].append([ds,m,n,w,'Evaluated' if met else 'Pending',met.get('n'),met.get('UCIQE'),met.get('NIQE'),met.get('URanker'),met.get('CIEDE2000'),str(path)])
  for m in plan['methods']:
   for w in ['raw','ema']:
    values=[]
    for ds in ['Challenging-60','U45','Color-Check7']:
     observations=extgroup.get((ds,m,w),[])
     for key in (['CIEDE2000'] if ds=='Color-Check7' else ['UCIQE','NIQE','URanker']):values.append(statistics.mean(x[key] for x in observations) if len(observations)==2 else None)
    pages['Table2'].append(['Current WWE Python / fixed colour protocol',plan['method_names'][m],w]+values+['V2 all4 pad-to8/crop; UIEB-trained, two-init means. UCIQE follows WWE Python HSV; paper older Matlab. CIEDE2000 uses fixed input ROIs + BabelColor D65; original ROI implementation unavailable. Missing stays blank.'])
  for stage in read(ROOT/'CODE_BACKUP_AUDIT.json',{}).get('stages',[]):pages['CodeBackups'].append([stage['mode'],stage['path'],stage['all_hashes_independent_inodes_and_clean_process_model_load_verified'],stage['files']])
  for k,v in plan.items():pages['Config'].append([k,json.dumps(v,ensure_ascii=False)[:30000]])
  for name in ['CODE_BACKUP_AUDIT.json','IMPORTED_REUSE_AUDIT.json','DATA_COPY_AUDIT.json','EXTERNAL_READINESS.json','SOURCE_PROVENANCE.json','DATASET_COMPATIBILITY.json','DATA_COMPATIBILITY_LOCK.json','checks/DARK_EXTENSION_CPU_AUDIT.json','checks/dark_extension_repair_20261117/REPAIR_AUDIT.json']:
   pages['Provenance'].append([name,json.dumps(read(ROOT/name),ensure_ascii=False)[:30000]])
  pages['Provenance'].append(['Interpretation','Paper rows are transcribed published values, not retrained by us. Current baseline is the accepted stripped BaselineUNet, not automatically the paper 0.734M model. Raw/EMA share a training trajectory. Imported accepted results are explicitly identified. No missing metric is replaced by0.'])
  history=read(ROOT/'external/history/v1/RESULTS_INDEX.json',{}).get('results',[])
  page=book.create_sheet('ExternalHistoryV1');page.append(['Dataset','Stage','Init','Weights','N','UCIQE WWE Python HSV','NIQE','URanker','CIEDE2000','Reason','Path'])
  for item in history:
   v=item['metrics'];page.append([item['dataset'],item['mode'],item['initialization_seed'],item['weights'],item['n'],v.get('UCIQE'),v.get('NIQE'),v.get('URanker'),v.get('CIEDE2000'),item['reason'],str(ROOT/item['path'])])
  seedpage=book.create_sheet('Seed7');seedpage.append(['Dataset','Stage','Init','Weights','Test PSNR','Test SSIM','Status','Selection note'])
  for ds in plan['datasets']:
   for mode in plan['methods']:
    for w in ['raw','ema']:
     value=read(ROOT/'runs'/ds/'seed7'/mode/f'{w}_test_per_image.json',{})
     seedpage.append([ds,mode,7,w,value.get('psnr'),value.get('ssim'),'Evaluated' if value else 'Pending','User selected7 after UIEB Test; retrospective view for completed data, prospective default for new training.'])
  retired=book.create_sheet('RetiredJobs');retired.append(['Dataset','Stage','Init','Status','Reason'])
  for j in plan['retired_jobs']:retired.append([j['dataset'],j['mode'],j['init_seed'],'Not trained; removed from future queue',j['reason']])
  pages['Provenance'].append(['TRAINING_SEED_POLICY.json',json.dumps(read(ROOT/'TRAINING_SEED_POLICY.json'),ensure_ascii=False)[:30000]])
  expected={}
  for page in book:
   page.freeze_panes='A2';page.auto_filter.ref=page.dimensions
   for cell in page[1]:cell.font=Font(bold=True,color='FFFFFF');cell.fill=PatternFill('solid',fgColor='24476B')
   for col in page.columns:page.column_dimensions[col[0].column_letter].width=22
   for line in page:
    for cell in line:
     if cell.value is not None and cell.value!='':expected[(page.title,cell.coordinate)]=cell.value
  path=ROOT/'four_stage_paper_comparison.xlsx';tmp=path.with_suffix('.tmp.xlsx');book.save(tmp);loaded=load_workbook(tmp,data_only=True)
  for (sheet,address),v in expected.items():
   actual=loaded[sheet][address].value
   if isinstance(v,float):assert math.isclose(actual,v,rel_tol=1e-12,abs_tol=1e-12),(sheet,address)
   else:assert actual==v,(sheet,address)
  loaded.close();tmp.replace(path)
  audit=dict(at=time.time(),passed=True,original_Table1_rows=10,original_Table2_rows=10,all_nonempty_cells_readback_verified=len(expected),formal_runs=plan['total_jobs'],original_declared_runs=40,retired_unstarted_runs=len(plan['retired_jobs']),trained=trained,paired_evaluated=tested,imported_runs=imports,external_evaluations=extdone,external_required=48,
   curve_rows=pages['Curves'].max_row-1,weight_rows=pages['Weights'].max_row-1,per_image_rows=pages['PerImage'].max_row-1,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
  (ROOT/'EXCEL_AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n');print('EXCEL_VERIFIED',json.dumps(audit),flush=True)
if __name__=='__main__':main()
