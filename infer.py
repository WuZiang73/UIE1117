"""RGB256 inference with one of the four archived UIEB seed7 EMA models."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from runtime import load_model

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['baseline','c0','isp_control','dino_cross_stage'], required=True)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cpu')
    a = p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    image = Image.open(a.input).convert('RGB').resize((256,256), Image.BILINEAR)
    x = torch.from_numpy(np.asarray(image, dtype=np.float32).copy().transpose(2,0,1) / 255.)[None]
    model = load_model(a.mode, a.device)
    with torch.no_grad(): y = model(x.to(a.device))
    assert y.shape == x.shape and torch.isfinite(y).all()
    output = y[0].clamp(0,1).cpu().numpy().transpose(1,2,0)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.rint(output*255).astype(np.uint8)).save(a.output)

if __name__ == '__main__': main()
