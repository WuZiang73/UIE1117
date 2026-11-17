"""Verify package hashes, four exact model loads and optional CPU forward replay."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from runtime import ROOT, load_model, sha

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--forward',action='store_true')
    a=parser.parse_args()
    manifest=json.loads((ROOT/'EXPORT_MANIFEST.json').read_text())
    for name,value in manifest['files'].items():
        p=ROOT/name
        assert p.is_file() and not p.is_symlink(), name
        assert p.stat().st_size==value['size_bytes'] and sha(p)==value['sha256'], name
    data=np.load(ROOT/'validation/cpu_forward_reference.npz') if a.forward else None
    results={}
    for mode in json.loads((ROOT/'MODELS.json').read_text()):
        model=load_model(mode,'cpu')
        if a.forward:
            with torch.no_grad(): output=model(torch.from_numpy(data['input']))
            assert torch.isfinite(output).all()
            assert np.array_equal(output.numpy(),data[mode]), ('CPU forward changed',mode)
        results[mode]=dict(strict_load=True,forward_bitexact=a.forward)
        del model
    print(json.dumps(dict(passed=True,files=len(manifest['files']),models=results),ensure_ascii=False))

if __name__ == '__main__': main()
