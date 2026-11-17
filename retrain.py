"""New UIEB seed7 training, using the unchanged accepted training engine."""
import argparse
import json
import os
from pathlib import Path
import time
from runtime import ROOT, sha

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['baseline','c0','isp_control','dino_cross_stage'],required=True)
    parser.add_argument('--resume',action='store_true')
    a=parser.parse_args()
    assert os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8'
    # This wrapper removes only the old machine's global scheduling entry guard.
    # experiment.train itself, factory, losses, EMA and deterministic operators are copied unchanged.
    import experiment as engine
    import torch
    assert int(os.environ['CUDA_VISIBLE_DEVICES']) in engine.RESOURCE_POLICY['authorized_gpus']
    assert torch.cuda.get_device_name(0)==engine.RESOURCE_POLICY['hardware_requirement']['name']
    assert torch.cuda.get_device_capability(0)==(8,9)
    assert (ROOT/'source/UnderWaterDataset/UIEB/train/input').is_dir()
    dest=engine.run_path('UIEB',a.mode,7)
    if a.resume:
        checkpoint=dest/'last_full.pth'
        assert checkpoint.is_file() and not (dest/'completed.json').exists()
        saved=torch.load(checkpoint,map_location='cpu')
        assert saved['trace_steps']==saved['epoch']*30
        for name,sig in saved['trace_chunks_sha256'].items(): assert sha(dest/'trace_epochs'/name)==sig
        path=ROOT/'RECOVERIES.json'
        recoveries=json.loads(path.read_text()) if path.exists() else {}
        key='UIEB/7/'+a.mode
        recovery=dict(at=time.time(),checkpoint_relative=str(checkpoint.relative_to(ROOT)),checkpoint_sha256=sha(checkpoint),attempt=recoveries.get(key,{}).get('attempt',0)+1)
        recoveries[key]=recovery;engine.atomic(path,recoveries)
        engine.train('UIEB',a.mode,7,resume=True)
    else:
        if dest.exists(): raise FileExistsError('Use --resume with a valid full checkpoint; do not overwrite '+str(dest))
        preflight=engine.run_path('UIEB',a.mode,7,True)
        if not (preflight/'completed.json').exists():
            if preflight.exists(): raise RuntimeError('An incomplete preflight must be preserved and diagnosed before retrying')
            engine.train('UIEB',a.mode,7,preflight=True)
        engine.train('UIEB',a.mode,7)

if __name__ == '__main__': main()
