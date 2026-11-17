"""Repeat the original UIEB Test protocol without writing into archived results."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from runtime import ROOT, load_model, sha

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['baseline','c0','isp_control','dino_cross_stage'], required=True)
    parser.add_argument('--data-root', type=Path, required=True, help='UIEB directory containing test/input and test/GT')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    a = parser.parse_args()
    data_root = a.data_root.resolve()
    output = a.output.resolve()
    if output.exists(): raise FileExistsError(output)
    manifest = json.loads((ROOT/'DATA_MANIFEST.json').read_text())['files']
    prefix = 'data/UnderWaterDataset/UIEB/test/'
    expected = {name[len(prefix):]:v for name,v in manifest.items() if name.startswith(prefix)}
    assert len(expected) == 178
    for name, value in expected.items():
        path = data_root/'test'/name
        assert sha(path) == value['sha256'] and path.stat().st_size == value['size_bytes'], path
    for directory in ['input','GT']:
        actual = {p.name for p in (data_root/'test'/directory).iterdir() if p.is_file()}
        names = {Path(name).name for name in expected if name.startswith(directory+'/')}
        assert actual == names, ('Test split differs',directory)
    model = load_model(a.mode, a.device)
    from utils.dataset import get_loader
    from utils.metrics import getPSNR, getSSIM
    rows = []; input_hash = hashlib.sha256(); target_hash = hashlib.sha256()
    with torch.no_grad():
        for x,y,names in get_loader(str(data_root/'test'),24,256,train=False,resize=True,num_workers=4,shuffle=False,pin_memory=True):
            input_hash.update(x.numpy().tobytes()); target_hash.update(y.numpy().tobytes())
            raw = model(x.to(a.device)); assert torch.isfinite(raw).all()
            prediction = raw.clamp(0,1).cpu().numpy().astype(np.float32).transpose(0,2,3,1)
            target = y.numpy().astype(np.float32).transpose(0,2,3,1)
            for i,name in enumerate(names):
                rows.append(dict(name=name,psnr=float(getPSNR(prediction[i:i+1],target[i:i+1],1.)),ssim=float(getSSIM(prediction[i:i+1],target[i:i+1],1.))))
    accepted = json.loads((ROOT/'results'/a.mode/'ema_test_per_image.json').read_text())
    assert len(rows) == 89
    assert input_hash.hexdigest() == accepted['input_sha256']
    assert target_hash.hexdigest() == accepted['target_sha256']
    assert [r['name'] for r in rows] == [r['name'] for r in accepted['images']]
    result = dict(mode=a.mode,weights='EMA Val-best',seed=7,n=89,device=a.device,psnr=float(np.mean([r['psnr'] for r in rows])),ssim=float(np.mean([r['ssim'] for r in rows])),images=rows,input_sha256=input_hash.hexdigest(),target_sha256=target_hash.hexdigest(),checkpoint_sha256=accepted['checkpoint_sha256'])
    result['delta_from_archived'] = dict(psnr=result['psnr']-accepted['psnr'],ssim=result['ssim']-accepted['ssim'])
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='images'},ensure_ascii=False))

if __name__ == '__main__': main()
