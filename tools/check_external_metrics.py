"""Check real input metrics and freeze the new Table2 pipeline before any model comparison."""
import hashlib,inspect,json,os,shutil,sys,time
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw
from skimage.color import deltaE_ciede2000
from frozen_factory import ROOT,SOURCE,atomic,sha
from external_metrics import uciqe_wwe_python,MetricSuite,color_chart_score,chart_samples
from utils.dataset import get_loader
from utils.metrics import get_uciqe

def main():
 torch.set_num_threads(2)
 checkpoint=ROOT/'evaluators/URanker_ckpt-450eb36d.pth';assert checkpoint.exists() and sha(checkpoint).startswith('450eb36d')
 assert np.isclose(deltaE_ciede2000(np.array([[50,2.6772,-79.7751]]),np.array([[50,0,-82.7485]]))[0],2.0425,atol=5e-5)
 assert np.isclose(deltaE_ciede2000(np.array([[50,3.1571,-77.2803]]),np.array([[50,0,-82.7485]]))[0],2.8615,atol=5e-5)
 ref=np.array(json.loads((ROOT/'evaluators/colorchecker_babelcolor_D65.json').read_text())['Lab_D65']);assert np.max(deltaE_ciede2000(ref,ref))==0
 # Real images: exact numerical definition of current author's HSV implementation.
 tested=[]
 for dataset in ['Challenging-60','U45']:
  loader=get_loader(str(ROOT/'data/UnderWaterDataset'/dataset/'test'),1,256,train=False,resize=True,num_workers=1,shuffle=False,non_ref=True)
  for index,(x,names) in enumerate(loader):
   if index>=2:break
   rgb=x[0].permute(1,2,0).numpy();a=get_uciqe(rgb);b=uciqe_wwe_python(rgb);assert abs(a-b)<1e-12,(a,b)
   tested.append(dict(dataset=dataset,name=names[0],original=a,vectorized=b,error=abs(a-b)))
 suite=MetricSuite(ROOT/'evaluators');metrics=[]
 for dataset in ['Challenging-60','U45']:
  x,names=next(iter(get_loader(str(ROOT/'data/UnderWaterDataset'/dataset/'test'),1,256,train=False,resize=True,num_workers=1,shuffle=False,non_ref=True)))
  first=suite(x);second=suite(x);assert first==second
  metrics.append(dict(dataset=dataset,name=names[0],scores=first,deterministic_replay_exact=True))
 # Annotation overlay drawn from input geometry only. This is a review artifact,
 # not a processed restoration output used to tune the metric.
 display=Image.new('RGB',(1200,4*450),'white');draw=ImageDraw.Draw(display);color=[]
 loader=get_loader(str(ROOT/'data/UnderWaterDataset/Color-Check7/test'),1,256,train=False,resize=True,num_workers=1,shuffle=False,non_ref=True)
 annotation_path=ROOT/'evaluators/colorcheck_annotations.json'
 for index,(x,names) in enumerate(loader):
  values=color_chart_score(x[0],names[0],annotation_path);assert len(values['patch_deltaE00'])==24
  samples,quad,warped=chart_samples(x[0],names[0],annotation_path)
  img=Image.fromarray(np.uint8(np.clip(warped,0,1)*255));d=ImageDraw.Draw(img)
  for row in range(4):
   for col in range(6):
    entry=json.loads(annotation_path.read_text())['images'][names[0]]
    box=entry.get('visible_patch_boxes',{}).get(str(row*6+col+1),[.275,.275,.725,.725])
    d.rectangle((int((col+box[0])*100),int((row+box[1])*100),int((col+box[2])*100),int((row+box[3])*100)),outline=(255,0,0),width=2)
    d.text((col*100+2,row*100+2),str(row*6+col+1),fill=(255,255,255))
  base_x=index%2*600;base_y=index//2*450;display.paste(img,(base_x,base_y+25));draw.text((base_x+3,base_y+4),names[0]+' fixed input ROIs',fill='black')
  color.append(dict(name=names[0],input_CIEDE2000=values['CIEDE2000'],all24patches=True))
 display.save(ROOT/'checks/colorcheck_input_sampling_review.png')
 assert len(color)==7
 files=['tools/external_evaluate.py','tools/external_metrics.py','tools/profile_efficiency.py','evaluators/URanker_ckpt-450eb36d.pth','evaluators/utils/niqe_pris_params.npz','evaluators/colorcheck_annotations.json','evaluators/colorchecker_babelcolor_D65.json','evaluators/colorcheck_source/DOWNLOAD_AUDIT.json']
 for folder in ['Challenging-60','U45','Color-Check7']:
  files += [str(p.relative_to(ROOT)) for p in sorted((ROOT/'data/UnderWaterDataset'/folder/'test').glob('*')) if p.is_file()]
 # Pin imported implementation files, too; site-packages are read-only evidence.
 import pyiqa.archs.uranker_arch as uranker_module
 import skimage,cv2
 library_sources={str(Path(uranker_module.__file__)):sha(Path(uranker_module.__file__))}
 result=dict(at=time.time(),passed=True,real_image_HSV_UCIQE_matches_existing_definition=tested,actual_pretrained_URanker_and_NIQE_replays=metrics,
  CIEDE2000_Sharma_examples_and_identity_pass=True,input_colorcheck_scores=color,reference='BabelColor Average D50 with Bradford adaptation to D65',
  ROIs_fixed_from_input_only=True,all24patches_equal_weight=True,paper_exact_colour_implementation_available=False,
  protocol_limitations='Published Table2 entries are older Matlab/proprietary ROI protocol values. Current HSV UCIQE/NIQE Python and fixed BabelColor D65 ROI scores are explicitly current-protocol observations; not claimed exact reproduction of published Input row.',
  library_sources=library_sources,torch=torch.__version__,skimage=skimage.__version__,cv2=cv2.__version__)
 atomic(ROOT/'EXTERNAL_METRICS_CPU_CHECKS.json',result)
 atomic(ROOT/'EXTERNAL_MATH_LOCK.json',dict(at=time.time(),files={rel:sha(ROOT/rel) for rel in files},library_sources=library_sources))
 # Readiness is released after manual review of the input-only sampling overlay.
 print('EXTERNAL_METRICS_CHECKS_PASSED; INPUT_OVERLAY_REVIEW_REQUIRED_BEFORE_READY',flush=True)
if __name__=='__main__':main()
