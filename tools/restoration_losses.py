"""Controlled pixel+SSIM losses matching the existing RGB/window5 metric definition."""
import torch
from torch import nn
import torch.nn.functional as F


def _box5(x):
    """Valid separable uniform mean, with no cuDNN/TF32 or overlapping pool backward."""
    h,w=x.shape[-2:]
    vertical=(x[...,:h-4,:]+x[...,1:h-3,:]+x[...,2:h-2,:]+x[...,3:h-1,:]+x[...,4:,:])*.2
    return (vertical[...,:w-4]+vertical[...,1:w-3]+vertical[...,2:w-2]+vertical[...,3:w-1]+vertical[...,4:])*.2


def ssim_rgb5(prediction,target):
    """RGB, valid5x5, range1, K1=.01/K2=.03, sample covariance25/24.

    Float64 local statistics avoid precision loss in nearly constant patches.
    The official skimage float32 evaluation stays unchanged; definitions match,
    while floating-point implementations need not be bitwise identical.
    """
    assert prediction.shape==target.shape and prediction.ndim==4
    assert prediction.shape[1]==3 and min(prediction.shape[-2:])>=5
    x=prediction.clamp(0.,1.).double();y=target.double()
    ux,uy=_box5(x),_box5(y)
    vx=(_box5(x*x)-ux*ux)*(25./24.)
    vy=(_box5(y*y)-uy*uy)*(25./24.)
    vxy=(_box5(x*y)-ux*uy)*(25./24.)
    numerator=(2*ux*uy+.01**2)*(2*vxy+.03**2)
    denominator=(ux*ux+uy*uy+.01**2)*(vx+vy+.03**2)
    return (numerator/denominator).mean()


class RestorationLoss(nn.Module):
    def __init__(self,pixel='charbonnier',ssim_weight=0.):
        super().__init__()
        assert pixel in ['charbonnier','l1'] and ssim_weight>=0
        self.pixel=pixel;self.ssim_weight=float(ssim_weight)
        self.last_components={}

    def forward(self,prediction,target):
        if self.pixel=='l1':pixel=F.l1_loss(prediction,target)
        else:
            diff=prediction-target
            pixel=torch.mean(torch.sqrt(diff*diff+1e-6))
        if self.ssim_weight:
            structure=1.-ssim_rgb5(prediction,target)
            weighted=self.ssim_weight*structure
            total=pixel+weighted
            self.last_components=dict(pixel=pixel.detach(),ssim_loss=structure.detach(),weighted_ssim=weighted.detach())
            return total
        # Returning the unmodified native scalar is required for bitwise
        # reproduction of the old Charbonnier-only training trajectory.
        self.last_components=dict(pixel=pixel.detach(),ssim_loss=None,weighted_ssim=0.)
        return pixel

    def component_values(self):
        return {name:float(value) if value is not None else None for name,value in self.last_components.items()}
