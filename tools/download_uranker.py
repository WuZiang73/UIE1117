"""Resume interrupted public checkpoint downloads and verify the upstream SHA prefix."""
from pathlib import Path
import hashlib,json,requests,time
ROOT=Path(__file__).resolve().parents[1];folder=ROOT/'evaluators';dest=folder/'URanker_ckpt-450eb36d.pth';part=dest.with_suffix('.pth.part');expected=86526613
url='https://huggingface.co/chaofengc/IQA-PyTorch-Weights/resolve/main/URanker_ckpt-450eb36d.pth?download=true'
for attempt in range(8):
 if dest.exists():break
 offset=part.stat().st_size if part.exists() else 0
 try:
  reply=requests.get(url,headers={'Range':f'bytes={offset}-'} if offset else {},stream=True,timeout=(15,30));reply.raise_for_status()
  append=offset>0 and reply.status_code==206
  if append:assert reply.headers['Content-Range'].startswith(f'bytes {offset}-')
  with part.open('ab' if append else 'wb') as f:
   for block in reply.iter_content(512*1024):f.write(block)
  assert part.stat().st_size==expected
  digest=hashlib.sha256(part.read_bytes()).hexdigest();assert digest.startswith('450eb36d'),digest
  part.rename(dest)
 except (requests.RequestException,AssertionError) as exc:
  print('DOWNLOAD_RETRY',attempt,type(exc).__name__,str(exc)[:180],'bytes',part.stat().st_size if part.exists() else 0,flush=True);time.sleep(min(5,attempt+1))
assert dest.exists();digest=hashlib.sha256(dest.read_bytes()).hexdigest();assert digest.startswith('450eb36d')
(folder/'URANKER_PROVENANCE.json').write_text(json.dumps(dict(passed=True,sha256=digest,bytes=dest.stat().st_size,url=url,upstream_name=dest.name,official_pyiqa_checkpoint_hash_prefix_verified=True),indent=2)+'\n')
print('URANKER_WEIGHT_VERIFIED',digest,flush=True)
