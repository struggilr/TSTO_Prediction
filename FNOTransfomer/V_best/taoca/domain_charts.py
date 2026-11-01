"""Domain-aware V2 chart repair for the fixed 2D TSTO rectangle.

The background annulus contains masked bodies; it is not claimed to be a solid-free
outer envelope. Its small inner hole is covered by the gap chart in this dataset.
"""
import math
import torch

def orientation(xy):
    a=xy[:-1];b=torch.roll(xy,shifts=-1,dims=1)[:-1];c=torch.roll(xy,shifts=-1,dims=1)[1:];d=xy[1:]
    cross=lambda u,v:u[...,0]*v[...,1]-u[...,1]*v[...,0]
    return torch.minimum(cross(b-a,c-a),cross(c-a,d-a))

def repair_charts(coords):
    # coords [N,4,H,W,2], chronological historical geometry only.
    repaired=coords.clone();n,k,h,w,_=coords.shape
    for frame in range(n):
        for chart in [0,1]:
            p=coords[frame,chart];boundary=p[:1];offset=p-boundary
            thickness=offset[-1].square().sum(-1).sqrt().median().clamp_min(1e-4)
            factor=1.5/thickness
            direction=offset[-1]/offset[-1].norm(dim=-1,keepdim=True).clamp_min(1e-6)
            for _ in range(30):
                candidate=boundary+factor*offset+.002*direction[None]
                if orientation(candidate).min()>1e-10:break
                factor=factor*.8
            else:raise ValueError('Cannot repair body chart with positive cells')
            repaired[frame,chart]=candidate
    # Fixed, history-independent background anchor inside the known fixed domain.
    center=coords.new_tensor([45.,15.]);corners=coords.new_tensor([[-10.,-20.],[-10.,50.],[100.,50.],[100.,-20.]])
    counts=[round(w*70/360),round(w*110/360),round(w*70/360)];counts.append(w-sum(counts))
    perimeter=torch.cat([corners[i]+torch.arange(counts[i],device=coords.device,dtype=coords.dtype)[:,None]/counts[i]*(corners[(i+1)%4]-corners[i]) for i in range(4)])
    outer=(perimeter-center).norm(dim=-1);direction=(perimeter-center)/outer[:,None]
    eta=torch.linspace(0,1,h,device=coords.device,dtype=coords.dtype)[:,None]
    stretch=.05*eta+.95*(1-torch.cos(math.pi*eta))*.5
    radius=1.5+(outer[None]-1.5)*stretch
    background=center+radius[...,None]*direction[None]
    background[...,0].clamp_(-10,100);background[...,1].clamp_(-20,50)
    if orientation(background).min()<=0:raise ValueError('Nonpositive domain background chart')
    repaired[:,2]=background
    return repaired
