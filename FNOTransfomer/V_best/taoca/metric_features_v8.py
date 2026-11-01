"""History-only physical mesh metrics; zero-initialized modulation of fine features."""
import torch
from torch import nn
import torch.nn.functional as F

def metric_features(history):
    # history [N,8,H,W]; last two coordinates use the unchanged [-1,1] scaling.
    xy=history[:,6:8]*history.new_tensor([55.,35.])[None,:,None,None]
    eta,xi=torch.gradient(xy,dim=(-2,-1));a=xi.square().sum(1,keepdim=True).sqrt().clamp_min(1e-6);b=eta.square().sum(1,keepdim=True).sqrt().clamp_min(1e-6)
    ua=xi/a;ub=eta/b;cos=(ua*ub).sum(1,keepdim=True);jac=xi[:,0:1]*eta[:,1:2]-xi[:,1:2]*eta[:,0:1]
    # Unitless lengths/Jacobian relative to the common physical-grid spacing.
    scale=(110/439*70/279)**.5
    return torch.cat([ua,ub,torch.log(a/scale).clamp(-6,6),torch.log(b/scale).clamp(-6,6),cos.clamp(-1,1),torch.log(jac.abs().clamp_min(1e-8)/(scale*scale)).clamp(-10,10)],1)

class MetricEncoder(nn.Module):
    def __init__(self,width):
        super().__init__()
        # Preserve the baseline CPU RNG stream, including its shuffled window order.
        with torch.random.fork_rng(devices=[]):
            self.net=nn.Sequential(nn.Conv2d(8,width,1),nn.GELU(),nn.Conv2d(width,width,1));nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,history,size):
        features=metric_features(history);features=F.interpolate(features,size=size,mode='bilinear',align_corners=False)
        return self.net(features)
