"""Keep the source ISP unchanged: uniform external pad/crop for all four models."""
import torch
from torch.nn import functional as F

def predict_external(model,image):
    height,width=image.shape[-2:]
    pad_h=(-height)%8;pad_w=(-width)%8
    padded=F.pad(image,(0,pad_w,0,pad_h),mode='reflect') if pad_h or pad_w else image
    result=model(padded)
    assert result.shape[:2]==image.shape[:2] and result.shape[-2:]==padded.shape[-2:]
    return result[...,:height,:width]
