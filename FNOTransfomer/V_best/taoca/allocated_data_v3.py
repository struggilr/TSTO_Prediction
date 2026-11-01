"""Fixed number of sampled points; per-chart height may differ, no label-driven geometry."""
import numpy as np
import torch
import torch.nn.functional as F
from .physics_data import PhysicsWindows
from .allocation_v3 import allocate,pad_coordinates
from .joint import sample_grid,normalize_xy

class AllocatedWindows(PhysicsWindows):
    def __init__(self,*args,heights,**kwargs):super().__init__(*args,**kwargs);self.heights=list(heights)
    def __getitem__(self,index):
        if self.route=='cartesian':return super().__getitem__(index)
        case,s=self.samples[index];n=self.history;m=self.horizon;r=self.record(case);key='allocation_'+str(self.heights)
        if key not in r:r[key]=allocate(r['coords'],self.heights)
        base=torch.cat([r['fields'][s-n:s],r['valid'][s-n:s,None]],1)[None]
        histories=[];coordinates=[];hmax=max(self.heights)
        for p in r[key]:
            past=p[s-n:s];h=past.shape[1];sampled=sample_grid(base,past[None,:,None])[0,:,0]
            mask=(sampled[:,5]>.999).float();fields=sampled[:,:5]*mask[:,None]
            xy=pad_coordinates(past,hmax);coordinates.append(xy)
            histories.append(torch.cat([F.pad(fields,(0,0,0,hmax-h)),F.pad(mask[:,None],(0,0,0,hmax-h)),normalize_xy(xy).permute(0,3,1,2)],1))
        return dict(history=torch.stack(histories,1),mach=torch.tensor(float(case[2:])),pose_history=r['pose'][s-n:s],dt=torch.from_numpy(np.diff(r['time'][s-n:s]).astype('f4')),last_xy=torch.stack(coordinates,1)[-1],templates=r['templates'],target=r['fields'][s:s+m],target_valid=r['valid'][s:s+m],target_pose=r['pose'][s:s+m],last_base=r['fields'][s-1],last_valid=r['valid'][s-1],case=case,start=s)
