"""Portable entry points; the archived 61 model files and training math stay unchanged."""
from pathlib import Path
import hashlib
import json
import os
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'tools'))
_cwd = Path.cwd()
try:
    from frozen_factory import create_model, seed
finally:
    os.chdir(_cwd)

_configured = False

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()

def configure():
    global _configured
    seed()
    if not _configured:
        import strict_math
        strict_math.install()
        _configured = True

def load_model(mode, device='cpu'):
    import torch
    configure()
    metadata = json.loads((ROOT / 'MODELS.json').read_text())[mode]
    checkpoint = ROOT / metadata['checkpoint_relative']
    assert sha(checkpoint) == metadata['sha256'], checkpoint
    model = create_model(mode, 7, 'cpu')
    state = torch.load(checkpoint, map_location='cpu')
    # Bypass the legacy optional-PSF compatibility relaxation for an exact audit.
    torch.nn.Module.load_state_dict(model, state, strict=True)
    assert all(torch.equal(model.state_dict()[k], v) for k, v in state.items())
    return model.to(device).eval()
