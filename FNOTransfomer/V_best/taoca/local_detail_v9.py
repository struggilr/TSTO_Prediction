"""Full-resolution local candidate; no labels or future true geometry in forward."""
import torch
from torch import nn
import torch.nn.functional as F
from .metric_model_v8 import MetricPredictor
from .metric_features_v8 import metric_features

class LocalDetail(nn.Module):
    def __init__(self,width=16):
        super().__init__()
        self.c1=nn.Conv2d(22,width,3)
        self.c2=nn.Conv2d(width,width,3)
        self.out=nn.Conv2d(width,5,1)
        nn.init.zeros_(self.out.weight);nn.init.zeros_(self.out.bias)
    def forward(self,x,periodic=False):
        def pad(z):
            z=F.pad(z,(0,0,1,1),mode='replicate')
            return F.pad(z,(1,1,0,0),mode='circular' if periodic else 'replicate')
        return self.out(F.gelu(self.c2(pad(F.gelu(self.c1(pad(x)))))))

class LocalPredictor(MetricPredictor):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        with torch.random.fork_rng(devices=[]):self.local_detail=LocalDetail()
    def forward(self,history,mach,pose_history,dt,last_xy,templates,horizon,route,last_base=None,last_valid=None):
        out=super().forward(history,mach,pose_history,dt,last_xy,templates,horizon,route,last_base,last_valid)
        fields=out['fields'];b,m,k,_,hmax,w=fields.shape;result=[]
        for j in range(k):
            h=self.heights[j] if route=='taoca' else hmax
            q=fields[:,:,j,:,:h].reshape(b*m,5,h,w)
            prior=out['persistence'][:,:,j,:,:h].reshape(b*m,5,h,w)
            geo=out['geometry'][:,:,j,:,:h].reshape(b*m,3,h,w)
            xy=out['xy'][:,:,j,:h].reshape(b*m,h,w,2)
            metric=metric_features(torch.cat([q.new_zeros(b*m,5,h,w),geo],1))
            lead=torch.arange(1,m+1,device=q.device,dtype=q.dtype)[None,:,None,None,None].expand(b,m,1,h,w).reshape(b*m,1,h,w)/m
            x=torch.cat([q,prior,geo,metric,lead],1)
            valid=geo[:,:1]
            delta=self.local_detail(x*valid,periodic=route=='taoca' and j<3)*valid
            corrected=(q+delta).reshape(b,m,5,h,w)
            result.append(F.pad(corrected,(0,0,0,hmax-h)))
        out['fields']=torch.stack(result,2)
        return out

