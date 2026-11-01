"""Fixed-budget chart resolution allocation; no flow labels or future geometry inputs."""
import math
import torch
import torch.nn.functional as F

def allocate(coords,heights):
    # coords [N,4,70,W,2]; return four [N,Hk,W,2] arrays.
    if sum(heights)!=280 or len(heights)!=4:raise ValueError('Total radial rows must remain 280')
    result=[];w=coords.shape[-2]
    for k,h in enumerate(heights):
        old=coords[:,k]
        if h==old.shape[1]:result.append(old);continue
        if k==2:
            eta=torch.linspace(0,1,h,device=coords.device,dtype=coords.dtype)[None,:,None,None]
            stretch=.05*eta+.95*(1-torch.cos(math.pi*eta))*.5
            result.append(old[:,:1]+stretch*(old[:,-1:]-old[:,:1]))
        else:result.append(F.interpolate(old.permute(0,3,1,2),size=(h,w),mode='bilinear',align_corners=True).permute(0,2,3,1))
    return result

def pad_coordinates(coords,height):
    # Linear extension is only for masked loss bookkeeping; never encoded by the network.
    n,h,w,c=coords.shape
    if h==height:return coords
    steps=torch.arange(1,height-h+1,device=coords.device,dtype=coords.dtype)[None,:,None,None]
    return torch.cat([coords,coords[:,-1:]+steps*(coords[:,-1:]-coords[:,-2:-1])],1)
