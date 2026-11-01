"""Same chronological splits and fields as JointWindows, shared in-process cache."""
import json
from pathlib import Path
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from .joint import JointWindows,normalize_xy,sample_grid
from .domain_charts import repair_charts

class PhysicsWindows(JointWindows):
    _records={}
    def record(self,case):
        key=(str(self.root.resolve()),case)
        if key not in self._records:
            with h5py.File(self.root/(case+'.h5')) as f:
                fields=f['fields'][:];valid=f['valid'][:].astype('f4');pose=f['pose'][:].astype('f4');times=f['time'][:];templates=f['templates'][:].astype('f4')
            fields=(fields-self.mean)/self.std;fields*=valid[:,None]
            self._records[key]=dict(fields=torch.from_numpy(fields),valid=torch.from_numpy(valid),pose=torch.from_numpy(pose),time=times,templates=torch.from_numpy(templates))
        record=self._records[key]
        if self.route=='taoca' and 'coords' not in record:
            with h5py.File(self.root/(case+'_geometry.h5')) as f:xy=torch.from_numpy(f['xy'][:])
            n,k,c,h,w=xy.shape
            coords=F.interpolate(xy.reshape(n*k,c,h,w),size=(70,440),mode='bilinear',align_corners=True).reshape(n,k,2,70,440).permute(0,1,3,4,2).contiguous()
            record['coords']=repair_charts(coords)
        return record
    def __getitem__(self,index):
        case,s=self.samples[index];n=self.history;m=self.horizon;r=self.record(case)
        if self.route=='cartesian':
            yy,xx=torch.meshgrid(torch.linspace(-20,50,280),torch.linspace(-10,100,440),indexing='ij')
            coords=torch.stack([xx,yy],-1)[None,None].expand(n,1,-1,-1,-1)
            h=r['fields'][s-n:s,None];mask=r['valid'][s-n:s,None]
        else:
            coords=r['coords'][s-n:s]
            base=torch.cat([r['fields'][s-n:s],r['valid'][s-n:s,None]],1)[None]
            sampled=sample_grid(base,coords[None])[0]
            mask=(sampled[:,:,5]>.999).float();h=sampled[:,:,:5]*mask[:,:,None]
        history=torch.cat([h,mask[:,:,None],normalize_xy(coords).permute(0,1,4,2,3)],2)
        return dict(history=history,mach=torch.tensor(float(case[2:])),pose_history=r['pose'][s-n:s],dt=torch.from_numpy(np.diff(r['time'][s-n:s]).astype('f4')),last_xy=coords[-1],templates=r['templates'],target=r['fields'][s:s+m],target_valid=r['valid'][s:s+m],target_pose=r['pose'][s:s+m],last_base=r['fields'][s-1],last_valid=r['valid'][s-1],case=case,start=s)
