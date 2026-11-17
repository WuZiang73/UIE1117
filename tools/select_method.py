"""Incrementally lock every fixed stage's full-Val best before its own Test."""
import fcntl,json,time
from frozen_factory import ROOT,PLAN,atomic,sha

def select():
 with (ROOT/'selection.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  locks=json.loads((ROOT/'WEIGHTS_LOCK.json').read_text()) if (ROOT/'WEIGHTS_LOCK.json').exists() else {}
  entries={}
  if (ROOT/'METHOD_SELECTION.json').exists():entries=json.loads((ROOT/'METHOD_SELECTION.json').read_text()).get('entries',{})
  for job in PLAN['predeclared_job_order']:
   ds,m,n=job['dataset'],job['mode'],job['init_seed'];folder=ROOT/'runs'/ds/f'seed{n}'/m;k=f'{ds}/{n}/{m}'
   if not (folder/'completed.json').exists():continue
   if k in locks:
    for w,digest in locks[k].items():assert sha(folder/('best_model.pth' if w=='ema' else 'best_raw_model.pth'))==digest
    continue
   imported=(folder/'IMPORTED_PROVENANCE.json').exists()
   assert imported or not list(folder.glob('*_test_per_image.json'))
   rows=json.loads((folder/'records.json').read_text());assert len(rows)==PLAN['epochs']==200
   values={}
   for w in ['raw','ema']:
    expected=max((v for v in rows if v[w] is not None),key=lambda v:v[w]['psnr'])
    best=json.loads((folder/f'best_{w}.json').read_text());assert expected['epoch']==best['epoch']
    values[w]=sha(folder/('best_model.pth' if w=='ema' else 'best_raw_model.pth'))
    if imported:
     for split in ['val','test']:
      metric=json.loads((folder/f'{w}_{split}_per_image.json').read_text());assert metric['checkpoint_sha256']==values[w]
   locks[k]=values;entries[k]=dict(at=time.time(),imported_previous_Val_locked_artifacts=imported,full_Val_only_checkpoint_selection=True)
  result=dict(at=time.time(),test_modes=PLAN['methods'],selected_mode=None,locked_before_new_Test=True,entries=entries,
   rule='All four cumulative versions fixed in advance; each Raw/EMA full-Val best checkpoint independently locked before its Test. No Test-selected candidate.')
  atomic(ROOT/'WEIGHTS_LOCK.json',locks);atomic(ROOT/'METHOD_SELECTION.json',result)
  return result
if __name__=='__main__':select()
