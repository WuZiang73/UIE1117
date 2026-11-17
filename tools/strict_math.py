"""Deterministic mathematical equivalents for native and added fusion operators."""
import torch
import torch.nn.functional as F
import base_equivalent_operators

COUNTS={'grad_bilinear':0,'grad_block_pool':0}
_CACHE={}


def interpolation_matrix(n,m,device,dtype):
    key=(n,m,str(device),dtype)
    if key not in _CACHE:
        p=(torch.arange(m,dtype=torch.float64)+0.5)*(n/m)-0.5
        lo=p.floor().long();a=p-lo
        matrix=torch.zeros(m,n,dtype=torch.float64)
        for i in range(m):
            matrix[i,int(lo[i].clamp(0,n-1))]+=1-float(a[i])
            matrix[i,int((lo[i]+1).clamp(0,n-1))]+=float(a[i])
        _CACHE[key]=matrix.to(device=device,dtype=dtype)
    return _CACHE[key]


def bilinear_matmul(x,size):
    h,w=x.shape[-2:];oh,ow=size
    yh=interpolation_matrix(h,oh,x.device,x.dtype)
    xw=interpolation_matrix(w,ow,x.device,x.dtype)
    return (yh@x)@xw.transpose(0,1)


def block_mean(x,size):
    h,w=x.shape[-2:];oh,ow=size
    assert h%oh==w%ow==0
    return x.reshape(*x.shape[:-2],oh,h//oh,ow,w//ow).mean((3,5))


def install(base_only=False):
    base_equivalent_operators.install()
    if base_only:return
    old_interp=F.interpolate;old_pool=F.adaptive_avg_pool2d
    def interp(input,size=None,scale_factor=None,mode='nearest',align_corners=None,recompute_scale_factor=None,antialias=False):
        if input.requires_grad and input.ndim==4 and mode=='bilinear' and align_corners is False and size is not None and not antialias:
            COUNTS['grad_bilinear']+=1
            target=(size,size) if isinstance(size,int) else tuple(size)
            return bilinear_matmul(input,target)
        return old_interp(input,size=size,scale_factor=scale_factor,mode=mode,align_corners=align_corners,recompute_scale_factor=recompute_scale_factor,antialias=antialias)
    def pool(input,output_size):
        target=(output_size,output_size) if isinstance(output_size,int) else tuple(output_size)
        if input.requires_grad and input.ndim==4 and target==(16,16) and input.shape[-2]%16==input.shape[-1]%16==0:
            COUNTS['grad_block_pool']+=1
            return block_mean(input,target)
        return old_pool(input,output_size)
    F.interpolate=interp;F.adaptive_avg_pool2d=pool
