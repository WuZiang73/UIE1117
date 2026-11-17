"""Diagnostic alternatives only; do not modify the archived candidate code.

Reflection padding uses the exact same pixel mapping, expressed as slice/flip/cat.
Dark-context median uses sorted middle values; descriptors consume fixed inputs.
Both are mathematical equivalents, not numerical bitwise backward equivalence.
"""
import torch
import torch.nn.functional as F
import dark_context_v2

def reflection_by_cat(x,pads):
    left,right,top,bottom=pads
    if left or right:
        chunks=[]
        if left:chunks.append(x[...,1:left+1].flip(-1))
        chunks.append(x)
        if right:chunks.append(x[...,-right-1:-1].flip(-1))
        x=torch.cat(chunks,dim=-1)
    if top or bottom:
        chunks=[]
        if top:chunks.append(x[...,1:top+1,:].flip(-2))
        chunks.append(x)
        if bottom:chunks.append(x[...,-bottom-1:-1,:].flip(-2))
        x=torch.cat(chunks,dim=-2)
    return x

def install():
    old_pad=F.pad
    def pad(x,pad,mode='constant',value=0.):
        if mode=='reflect' and x.ndim==4 and len(pad)==4 and all(v>=0 for v in pad):
            assert max(pad[:2])<x.shape[-1] and max(pad[2:])<x.shape[-2]
            return reflection_by_cat(x,pad)
        return old_pad(x,pad,mode,value)
    def median3(x):
        padded=F.pad(x,(1,1,1,1),mode='reflect')
        patches=F.unfold(padded,kernel_size=3).view(x.shape[0],x.shape[1],9,-1)
        return patches.sort(dim=2).values[:,:,4].view_as(x)
    F.pad=pad;dark_context_v2.median3=median3
