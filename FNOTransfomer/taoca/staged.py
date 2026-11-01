"""Frozen history-only kinematic surrogate shared by both flow representations."""
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .model import Predictor
from .joint import future_xy,normalize_xy,solid_mask


class FrozenTrajectory(nn.Module):
    def __init__(self,path):
        super().__init__();m=json.loads(Path(path).read_text())
        self.degree=m['degree'];self.history=m['history'];self.gain=m['gain']

    def forward(self,pose,dt,horizon):
        # Fit a local time polynomial using observed history only. No Mach lookup,
        # absolute test time, future labels, or flow-dependent gradients.
        with torch.no_grad():
            if pose.shape[1]!=self.history:raise ValueError('Trajectory history mismatch')
            times=torch.cat([torch.zeros_like(dt[:,:1]),dt.cumsum(1)],1)
            times=times-times[:,-1:];span=(-times[:,0]).clamp_min(1e-8)
            u=times/span[:,None]
            design=torch.stack([u**i for i in range(self.degree+1)],-1).double()
            y=(pose-pose[:,-1:]).double()
            coef=torch.linalg.lstsq(design,y).solution
            future=torch.arange(1,horizon+1,device=pose.device)[None]*dt[:,-1:]/span[:,None]
            powers=torch.stack([future**i for i in range(self.degree+1)],-1).double()
            local=pose[:,-1:]+(powers@coef).to(pose.dtype)
            steps=torch.arange(1,horizon+1,device=pose.device)[None,:,None]
            cv=pose[:,-1:]+steps*(pose[:,-1:]-pose[:,-2:-1])
            return (cv+self.gain*(local-cv)).detach()


class StagedPredictor(nn.Module):
    def __init__(self,trajectory_path,**config):
        super().__init__();self.trajectory=FrozenTrajectory(trajectory_path);self.flow=Predictor(**config)

    def forward(self,history,mach,pose_history,dt,last_xy,templates,horizon,route):
        pose=self.trajectory(pose_history,dt,horizon)
        with torch.no_grad():
            xy=future_xy(last_xy,pose_history[:,-1],pose) if route=='taoca' else last_xy[:,None].expand(-1,horizon,-1,-1,-1,-1)
            normalized=normalize_xy(xy)
            valid=((normalized.abs()<=1).all(-1)&~solid_mask(xy,pose,templates)).to(history.dtype)
            geometry=torch.cat([valid[:,:,:,None],normalized.permute(0,1,2,5,3,4)],3)
        return dict(fields=self.flow(history,mach,geometry),pose=pose,xy=xy,geometry=geometry)
