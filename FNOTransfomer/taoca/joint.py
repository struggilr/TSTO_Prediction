"""History-only joint pose/flow forecasting. Targets never enter forward()."""
import json
from pathlib import Path
import h5py
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from .model import Predictor

BOUNDS=(-10.,100.,-20.,50.)


def normalize_xy(xy):
    lo=xy.new_tensor([-10.,-20.]);span=xy.new_tensor([110.,70.])
    return 2*(xy-lo)/span-1


def sample_grid(base,xy):
    # base [B,T,C,H,W], xy [B,T,K,Hc,Wc,2]
    b,t,k,h,w,_=xy.shape
    x=base[:,:,None].expand(-1,-1,k,-1,-1,-1).reshape(b*t*k,base.shape[2],*base.shape[-2:])
    return F.grid_sample(x,normalize_xy(xy).reshape(b*t*k,h,w,2),align_corners=True,padding_mode='zeros').reshape(b,t,k,base.shape[2],h,w)


def future_xy(last_xy,last_pose,pose):
    """Blend two rigid displacements by distances to last observed body centers.

    This is a forecasting chart transport, not a proof of positive Jacobian.
    Geometry failures must be reported at evaluation.
    """
    b,k,h,w,_=last_xy.shape;m=pose.shape[1]
    old=last_pose.reshape(b,2,3);new=pose.reshape(b,m,2,3)
    relative=last_xy[:,None,:,None]-old[:,None,None,:,None,None,:2]
    angle=new[:,:,:,2]-old[:,None,:,2]
    co=angle.cos()[:,:,None,:,None,None];si=angle.sin()[:,:,None,:,None,None]
    x=relative[...,0];y=relative[...,1]
    transformed=torch.stack([co*x-si*y,si*x+co*y],-1)+new[:,:,None,:,None,None,:2]
    distance=relative.square().sum(-1).clamp_min(1e-3)
    weights=distance.reciprocal();weights=weights/weights.sum(3,keepdim=True)
    if k==4:
        weights=weights.expand(-1,m,-1,-1,-1,-1).clone()
        weights[:,:,0]=0;weights[:,:,0,0]=1
        weights[:,:,1]=0;weights[:,:,1,1]=1
    result=(transformed*weights[...,None]).sum(3)
    if k==4:result[:,:,2]=last_xy[:,None,2]  # fixed outer chart
    return result


def solid_mask(xy,pose,templates):
    """Ray crossing using predicted rigid polygons; bounded edge loop memory."""
    b,m,k,h,w,_=xy.shape;inside=torch.zeros((b,m,k,h,w),device=xy.device,dtype=torch.bool)
    for body in range(2):
        local=templates[:,body];angle=pose[:,:,body*3+2];co=angle.cos();si=angle.sin()
        px=co[:,:,None]*local[:,None,:,0]-si[:,:,None]*local[:,None,:,1]+pose[:,:,body*3,None]
        py=si[:,:,None]*local[:,None,:,0]+co[:,:,None]*local[:,None,:,1]+pose[:,:,body*3+1,None]
        flag=torch.zeros_like(inside);x=xy[...,0];y=xy[...,1]
        for edge in range(local.shape[1]):
            ax=px[:,:,edge,None,None,None];ay=py[:,:,edge,None,None,None]
            bx=px[:,:,(edge+1)%local.shape[1],None,None,None];by=py[:,:,(edge+1)%local.shape[1],None,None,None]
            denominator=by-ay;denominator=torch.where(denominator.abs()<1e-12,torch.ones_like(denominator),denominator)
            flag^=((ay>y)!=(by>y))&(x<(bx-ax)*(y-ay)/denominator+ax)
        inside|=flag
    return inside


class JointPredictor(nn.Module):
    def __init__(self,**config):
        super().__init__();self.flow=Predictor(**config)
        # Shared learned correction to a history-only constant velocity baseline.
        self.motion=nn.GRU(7,64,batch_first=True)
        self.flow_motion=nn.Linear(5,64)
        self.motion_head=nn.Sequential(nn.Linear(65,64),nn.GELU(),nn.Linear(64,6))
        nn.init.zeros_(self.motion_head[-1].weight);nn.init.zeros_(self.motion_head[-1].bias)

    def forward(self,history,mach,pose_history,dt,last_xy,templates,horizon,route):
        b,t=pose_history.shape[:2];scale=pose_history.new_tensor([110,70,1,110,70,1])
        delta=(pose_history-pose_history[:,-1:])/scale
        z,_=self.motion(torch.cat([delta,(mach/7)[:,None,None].expand(b,t,1)],-1))
        steps=torch.arange(1,horizon+1,device=history.device,dtype=history.dtype)[None,:,None]
        velocity=(pose_history[:,-1]-pose_history[:,-2])/dt[:,-1,None]
        future_dt=dt[:,-1,None,None]*steps
        mask=history[:,-1,:,5:6];pooled=(history[:,-1,:,:5]*mask).sum((1,3,4))/mask.sum((1,3,4)).clamp_min(1)
        feature=(z[:,-1]+self.flow_motion(pooled))[:,None].expand(-1,horizon,-1)
        correction=self.motion_head(torch.cat([feature,steps.expand(b,-1,-1)/horizon],-1))
        pose=pose_history[:,-1,None]+velocity[:,None]*future_dt+correction*scale*future_dt.square()
        xy=future_xy(last_xy,pose_history[:,-1],pose) if route=='taoca' else last_xy[:,None].expand(-1,horizon,-1,-1,-1,-1)
        # Predicted geometry only; support is determined by known fixed domain.
        normalized=normalize_xy(xy)
        valid=((normalized.abs()<=1).all(-1)&~solid_mask(xy,pose,templates)).to(history.dtype)
        geometry=torch.cat([valid[:,:,:,None],normalized.permute(0,1,2,5,3,4)],3)
        fields=self.flow(history,mach,geometry)
        return dict(fields=fields,pose=pose,xy=xy,geometry=geometry)


class JointWindows(torch.utils.data.Dataset):
    def __init__(self,root,paired,route,split,history=20,horizon=10):
        self.root=Path(root);self.paired=Path(paired);self.route=route;self.history=history;self.horizon=horizon
        self.meta=json.loads((self.root/'manifest.json').read_text());self.samples=[]
        for case,r in self.meta['cases'].items():
            n=len(r['times']);a=round(.7*n);v=a+round(.1*n)
            lo,hi={'train':(history,a),'val':(a,v),'test':(v,n)}[split]
            for s in range(max(history,lo),hi-horizon+1):self.samples.append((case,s))
        self.mean=np.array(self.meta['mean'],dtype='f4')[None,:,None,None];self.std=np.array(self.meta['std'],dtype='f4')[None,:,None,None]

    def __len__(self):return len(self.samples)

    def __getitem__(self,index):
        case,s=self.samples[index];n=self.history;m=self.horizon
        with h5py.File(self.root/(case+'.h5')) as f:
            raw=f['fields'][s-n:s+m];valid=f['valid'][s-n:s+m]
            pose=f['pose'][s-n:s+m];times=f['time'][s-n:s+m];templates=f['templates'][:]
        normalized=(raw-self.mean)/self.std;normalized*=valid[:,None]
        # A preserves the original complete Cartesian grid; B uses four charts.
        if self.route=='cartesian':
            yy,xx=np.meshgrid(np.linspace(-20,50,280),np.linspace(-10,100,440),indexing='ij')
            xy=np.stack([xx,yy],-1)[None].astype('f4')
            h=normalized[:n,None]
            mask=valid[:n,None]
            coords=np.broadcast_to(xy,(n,*xy.shape))
        else:
            coords=[]
            # Only historical geometry files are opened for model inputs.
            for i in range(s-n,s):
                compact=self.root/(case+'_geometry.h5')
                with h5py.File(compact if compact.exists() else self.paired/case/f'{i:04d}.h5') as f:
                    c=f['xy'][i] if compact.exists() else np.stack([f['taoca'][name]['xy'][:] for name in ['smj','xmj','outer','gap']])
                    c=F.interpolate(torch.from_numpy(c),size=(70,440),mode='bilinear',align_corners=True).permute(0,2,3,1).numpy()
                    coords.append(c)
            coords=np.stack(coords)
            base=torch.from_numpy(np.concatenate([normalized[:n],valid[:n,None]],1))[None]
            sampled=sample_grid(base,torch.from_numpy(coords)[None])[0].numpy()
            h=sampled[:,:,:5];mask=(sampled[:,:,5]>.999).astype('f4');h*=mask[:,:,None]
        coords=torch.from_numpy(np.array(coords,copy=True));nc=normalize_xy(coords).permute(0,1,4,2,3).numpy()
        history=np.concatenate([h,mask[:,:,None],nc],2).astype('f4')
        return dict(history=torch.from_numpy(history),mach=torch.tensor(float(case[2:])),pose_history=torch.from_numpy(pose[:n].astype('f4')),dt=torch.from_numpy(np.diff(times[:n]).astype('f4')),last_xy=coords[-1],templates=torch.from_numpy(templates.astype('f4')),target=torch.from_numpy(normalized[n:].astype('f4')),target_valid=torch.from_numpy(valid[n:].astype('f4')),target_pose=torch.from_numpy(pose[n:].astype('f4')),case=case,start=s)


def joint_loss(output,batch,route):
    pred=output['fields'];target=sample_grid(batch['target'],output['xy'].detach())
    valid=sample_grid(batch['target_valid'][:,:,None],output['xy'].detach())>.999
    valid=valid*output['geometry'][:,:,:,:1].bool()
    flow=((pred-target).square()*valid).sum()/(5*valid.sum()).clamp_min(1)
    scale=pred.new_tensor([110,70,1,110,70,1])
    pose=((output['pose']-batch['target_pose'])/scale).square().mean()
    # Target support in the loss is allowed; it is never a forward input.
    return flow+pose,dict(flow=flow.detach().item(),pose=pose.detach().item())
