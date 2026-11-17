"""Fixed, provenance-recorded Table2 metrics; no restoration/training code changes."""
import json,os
from pathlib import Path
import cv2
import numpy as np
import torch
from skimage.color import rgb2lab,deltaE_ciede2000

def uciqe_wwe_python(rgb):
 # Vectorized exact definition of the existing author's Python HSV implementation.
 # It is deliberately labelled separately from the paper's older Matlab numbers.
 hsv=cv2.cvtColor(np.asarray(rgb*255,dtype=np.uint8),cv2.COLOR_RGB2HSV)
 h,s,v=cv2.split(hsv);values=v.reshape(-1).astype(np.float64)/255.;n=values.size//100
 assert n>0
 top=np.partition(values,values.size-n)[-n:].mean();bottom=np.partition(values,n-1)[:n].mean()
 return float(.4680*(np.std(h)/180.)+.2745*(top-bottom)+.2576*(np.mean(s)/255.))

class MetricSuite:
 def __init__(self,folder):
  from pyiqa.archs.uranker_arch import URanker
  self.folder=Path(folder).resolve()
  self.model=URanker(pretrained=False).cpu()
  state=torch.load(folder/'URanker_ckpt-450eb36d.pth',map_location='cpu')
  self.model.load_state_dict(state,strict=True);self.model.eval()
  for p in self.model.parameters():p.requires_grad_(False)
 def __call__(self,image):
  from utils.metrics import getNIQE
  assert image.ndim==4 and image.shape[0]==1 and torch.isfinite(image).all()
  rgb=image.permute(0,2,3,1).contiguous().numpy()
  with torch.no_grad():score=float(self.model(image).reshape(-1)[0])
  previous=Path.cwd()
  try:
   os.chdir(self.folder)
   niqe=float(getNIQE(rgb))
  finally:os.chdir(previous)
  result=dict(UCIQE=uciqe_wwe_python(rgb[0]),NIQE=niqe,URanker=score)
  assert all(np.isfinite(v) for v in result.values());return result

def chart_samples(image,name,annotation_path):
 annotation=json.loads(Path(annotation_path).read_text());entry=annotation['images'][name]
 original_w,original_h=entry['original_size'];size=256
 if original_w<original_h:w,h=size,int(size*original_h/original_w)
 else:h,w=size,int(size*original_w/original_h)
 rgb=image.permute(1,2,0).contiguous().numpy().astype(np.float64)
 assert rgb.shape[:2]==(h//4*4,w//4*4)
 quad=np.array(entry['quad_original_pixels'],np.float32)*np.array([w/original_w,h/original_h],np.float32)
 destination=np.array([[0,0],[600,0],[600,400],[0,400]],np.float32)
 transform=cv2.getPerspectiveTransform(quad,destination);warped=cv2.warpPerspective(rgb,transform,(600,400),flags=cv2.INTER_LINEAR)
 fraction=annotation['central_cell_fraction'];margin=(1-fraction)/2
 samples=[]
 for row in range(4):
  for col in range(6):
   box=entry.get('visible_patch_boxes',{}).get(str(row*6+col+1),[margin,margin,1-margin,1-margin])
   left,top,right,bottom=box
   assert 0<=left<right<=1 and 0<=top<bottom<=1
   region=warped[int((row+top)*100):int((row+bottom)*100),int((col+left)*100):int((col+right)*100)]
   samples.append(region.mean((0,1)))
 return np.array(samples),quad,warped

def color_chart_score(image,name,annotation_path):
 annotation=json.loads(Path(annotation_path).read_text());reference_path=Path(annotation_path).parent/annotation['reference_file']
 reference=np.array(json.loads(reference_path.read_text())['Lab_D65'],np.float64)
 sampled,_,_=chart_samples(image,name,annotation_path)
 observed=rgb2lab(sampled.reshape(1,24,3)).reshape(24,3)
 delta=deltaE_ciede2000(observed,reference)
 assert delta.shape==(24,) and np.isfinite(delta).all()
 return dict(CIEDE2000=float(delta.mean()),patch_deltaE00=delta.tolist(),patch_mean_RGB=sampled.tolist())
