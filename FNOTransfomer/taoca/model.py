"""Shared FNO/Transformer backbone adapted from FNOTransfomerCDDIM_V2.0.

Original FNOEncoder and TemporalPredictor are preserved in adjacent modules.
The portable decoder accepts arbitrary chart shapes and explicit future geometry.
"""
import torch
from torch import nn
import torch.nn.functional as F
from .fno_blocks import FNOEncoder
from .temporal_net import TemporalPredictor


class Predictor(nn.Module):
    def __init__(self,width=32,modes=12,latent=128,layers=3,temporal_layers=2):
        super().__init__()
        if latent%4 or modes<1:raise ValueError('latent must be divisible by four; modes must be positive')
        self.encoder=FNOEncoder(8,32,modes,width,latent,layers)
        self.condition=nn.Linear(1,32)
        self.temporal=TemporalPredictor(latent,4,temporal_layers,0.1)
        self.role=nn.Embedding(4,latent)
        self.context=nn.Linear(latent,latent)
        self.decode=nn.Sequential(nn.Linear(latent,512),nn.GELU(),nn.Linear(512,16*12*16))
        self.refine=nn.Sequential(nn.Conv2d(16+5+3,32,3,padding=1),nn.GELU(),nn.Conv2d(32,5,1))

    def forward(self,history,mach,geometry):
        b,t,k,c,h,w=history.shape;horizon=geometry.shape[1]
        # Common fourfold spatial downsampling limits activation memory for 20 frames.
        small=F.interpolate(history.reshape(b*t*k,c,h,w),size=(max(12,h//4),max(24,w//4)),mode='bilinear',align_corners=False)
        if self.encoder.spectral_blocks[0].modes>min(small.shape[-2:]):raise ValueError('Fourier modes exceed downsampled spatial size')
        cond=self.condition((mach/7).reshape(b,1));cond=cond[:,None,None].expand(b,t,k,32).reshape(b*t*k,32)
        z=self.encoder(small,cond).reshape(b,t,k,-1)
        roles=torch.arange(k,device=z.device) if k==4 else torch.tensor([2],device=z.device)
        z=z+self.role(roles)[None,None]+self.context(z.mean(dim=2,keepdim=True))
        z=self.temporal(z.permute(0,2,1,3).reshape(b*k,t,-1),horizon)
        latent=self.decode(z.reshape(b*k*horizon,-1)).reshape(-1,16,12,16)
        latent=F.interpolate(latent,size=(h,w),mode='bilinear',align_corners=False)
        last=history[:,-1,:,:5].unsqueeze(2).expand(b,k,horizon,5,h,w).reshape(-1,5,h,w)
        geo=geometry.permute(0,2,1,3,4,5).reshape(-1,3,h,w)
        out=(last+self.refine(torch.cat([latent,last,geo],dim=1)))*geo[:,:1]
        return out.reshape(b,k,horizon,5,h,w).permute(0,2,1,3,4,5)
