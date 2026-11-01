"""V2 spatial FNO/Transformer: history-only shared motion, geometry-conditioned residuals."""
import math
import torch
from torch import nn
import torch.nn.functional as F
from .staged import FrozenTrajectory
from .joint import future_xy,normalize_xy,solid_mask,sample_grid

class RealSpectral(nn.Module):
    def __init__(self,width,modes=6):
        super().__init__();self.modes=modes
        self.positive=nn.Parameter(torch.randn(width,width,modes,modes,2)*.02)
        self.negative=nn.Parameter(torch.randn(width,width,modes,modes,2)*.02)
    def forward(self,x):
        h,w=x.shape[-2:];m=min(self.modes,h//2,w//2+1)
        xf=torch.fft.rfft2(x.float(),norm='ortho');out=torch.zeros_like(xf)
        out[:,:,:m,:m]=torch.einsum('bixy,ioxy->boxy',xf[:,:,:m,:m],torch.view_as_complex(self.positive[:,:,:m,:m].contiguous()))
        out[:,:,-m:,:m]=torch.einsum('bixy,ioxy->boxy',xf[:,:,-m:,:m],torch.view_as_complex(self.negative[:,:,:m,:m].contiguous()))
        return torch.fft.irfft2(out,s=(h,w),norm='ortho')

class ConvGRU(nn.Module):
    def __init__(self,c):
        super().__init__();self.gates=nn.Conv2d(c*2,c*2,3,padding=1);self.candidate=nn.Conv2d(c*2,c,3,padding=1)
    def forward(self,x,h):
        reset,update=self.gates(torch.cat([x,h],1)).sigmoid().chunk(2,1)
        candidate=torch.tanh(self.candidate(torch.cat([x,reset*h],1)))
        return (1-update)*h+update*candidate

class SpatialPredictor(nn.Module):
    def __init__(self,trajectory_path,width=24,modes=6,history=20):
        super().__init__();self.trajectory=FrozenTrajectory(trajectory_path);self.width=width;self.history=history
        self.fine=nn.Sequential(nn.Conv2d(8,width,3,stride=2,padding=1),nn.GELU(),nn.Conv2d(width,width,3,padding=1),nn.GELU())
        self.down=nn.Conv2d(width,width,3,stride=2,padding=1)
        self.spectral=nn.ModuleList([RealSpectral(width,modes) for _ in range(3)])
        self.local=nn.ModuleList([nn.Conv2d(width,width,1) for _ in range(3)])
        self.condition=nn.Linear(1,width);self.roles=nn.Embedding(4,width)
        layer=nn.TransformerEncoderLayer(width,4,width*4,dropout=0.,batch_first=True)
        self.temporal=nn.TransformerEncoder(layer,2)
        position=torch.arange(history)[:,None];freq=torch.exp(torch.arange(0,width,2)*(-math.log(10000.)/width));pe=torch.zeros(history,width);pe[:,0::2]=torch.sin(position*freq);pe[:,1::2]=torch.cos(position*freq);self.register_buffer('position',pe)
        self.geometry_encoder=nn.Sequential(nn.Conv2d(10,width,3,padding=1),nn.GELU())
        self.advance=ConvGRU(width)
        self.up_half=nn.Sequential(nn.Conv2d(width*2,width,3,padding=1),nn.GELU(),nn.Conv2d(width,width,3,padding=1),nn.GELU())
        self.decode=nn.Sequential(nn.Conv2d(width+10+5+1,width,3,padding=1),nn.GELU(),nn.Conv2d(width,width,3,padding=1),nn.GELU(),nn.Conv2d(width,5,1))
        nn.init.zeros_(self.decode[-1].weight);nn.init.zeros_(self.decode[-1].bias)

    def forward(self,history,mach,pose_history,dt,last_xy,templates,horizon,route,last_base=None,last_valid=None):
        b,t,k,_,h,w=history.shape
        pose=self.trajectory(pose_history,dt,horizon)
        with torch.no_grad():
            xy=future_xy(last_xy,pose_history[:,-1],pose) if route=='taoca' else last_xy[:,None].expand(-1,horizon,-1,-1,-1,-1)
            nc=normalize_xy(xy);inside=solid_mask(xy,pose,templates)
            valid=((nc.abs()<=1+2e-6).all(-1)&~inside)
            geo=torch.cat([valid[:,:,:,None].float(),nc.permute(0,1,2,5,3,4)],3)
            relative=(xy[:,:,:,None]-pose.reshape(b,horizon,2,3)[:,:,None,:,None,None,:2])/xy.new_tensor([110.,70.])
            relative=relative.permute(0,1,2,3,6,4,5).reshape(b,horizon,k,4,h,w)
            angle=(pose-pose_history[:,-1,None]).reshape(b,horizon,2,3)[...,2]
            angles=angle[:,:,None,:,None,None].expand(-1,-1,k,-1,h,w)
            lead=torch.arange(1,horizon+1,device=history.device,dtype=history.dtype)[None,:,None,None,None,None].expand(b,-1,k,1,h,w)/horizon
            motion=torch.cat([geo,relative,angles,lead],3)
            if last_base is None:raise ValueError('V2 requires common last observed physical-grid fields, not future fields')
            persistence=sample_grid(last_base[:,None].expand(-1,horizon,-1,-1,-1),xy)
            observed=sample_grid(last_valid[:,None,None].expand(-1,horizon,-1,-1,-1),xy)>.999
            persistence*=observed
        fine=self.fine(history.reshape(b*t*k,8,h,w));hh,ww=fine.shape[-2:]
        coarse=F.gelu(self.down(fine));hc,wc=coarse.shape[-2:]
        cond=self.condition((mach/7)[:,None])[:,None,None].expand(-1,t,k,-1).reshape(b*t*k,self.width,1,1)
        coarse=coarse+cond
        for spec,local in zip(self.spectral,self.local):coarse=F.gelu(coarse+spec(coarse)+local(coarse))
        roles=torch.arange(k,device=history.device) if k==4 else torch.tensor([2],device=history.device)
        maps=coarse.reshape(b,t,k,self.width,hc,wc)+self.roles(roles)[None,None,:,:,None,None]
        context=maps.mean(2,keepdim=True);maps=maps+.25*context
        tokens=F.adaptive_avg_pool2d(maps.reshape(b*t*k,self.width,hc,wc),(4,4)).reshape(b,t,k,self.width,16).permute(0,2,4,1,3).reshape(b*k*16,t,self.width)
        encoded=self.temporal(tokens+self.position[None,:t])[:,-1]
        temporal=encoded.reshape(b,k,4,4,self.width).permute(0,1,4,2,3).reshape(b*k,self.width,4,4)
        hidden=maps[:,-1].reshape(b*k,self.width,hc,wc)+F.interpolate(temporal,size=(hc,wc),mode='bilinear',align_corners=False)
        skip=fine.reshape(b,t,k,self.width,hh,ww)[:,-1].reshape(b*k,self.width,hh,ww)
        outputs=[]
        for step in range(horizon):
            features=motion[:,step].reshape(b*k,10,h,w)
            g=F.interpolate(features,size=(hc,wc),mode='bilinear',align_corners=False)
            hidden=self.advance(self.geometry_encoder(g),hidden)
            half=self.up_half(torch.cat([F.interpolate(hidden,size=(hh,ww),mode='bilinear',align_corners=False),skip],1))
            high=F.interpolate(half,size=(h,w),mode='bilinear',align_corners=False)
            prior=persistence[:,step].reshape(b*k,5,h,w);known=observed[:,step].reshape(b*k,1,h,w).float()
            output=prior+self.decode(torch.cat([high,features,prior,known],1))
            outputs.append(output.reshape(b,k,5,h,w)*valid[:,step,:,None])
        return dict(fields=torch.stack(outputs,1),pose=pose,xy=xy,geometry=geo,persistence=persistence)
