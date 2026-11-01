import json
from pathlib import Path
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

NAMES={'taoca':['smj','xmj','outer','gap'],'cartesian':['cartesian']}


class Windows(Dataset):
    def __init__(self, root, route, split, history=20, horizon=10):
        self.root=Path(root);self.route=route;self.history=history;self.horizon=horizon
        self.meta=json.loads((self.root/'manifest.json').read_text());self.samples=[]
        norm=self.meta['normalization'];self.mean=np.array(norm['mean'],dtype='f4')[:,None,None];self.std=np.array(norm['std'],dtype='f4')[:,None,None]
        for case,records in self.meta['cases'].items():
            lo,hi={'train':(history,60),'val':(60,80),'test':(80,len(records))}[split]
            for start in range(max(lo,history),hi-horizon+1):self.samples.append((case,start))

    def __len__(self):return len(self.samples)

    def frame(self,case,index):
        with h5py.File(self.root/self.meta['cases'][case][index]['file'],'r') as f:
            fields=[];masks=[];coords=[]
            for name in NAMES[self.route]:
                c=f[self.route][name];valid=c['mask'][:]
                fields.append(np.where(valid[None],(c['fields'][:]-self.mean)/self.std,0))
                masks.append(valid[None].astype('f4'))
                xy=c['xy'][:];lo=np.array(self.meta['bounds'][0],dtype='f4')[:,None,None];hi=np.array(self.meta['bounds'][1],dtype='f4')[:,None,None]
                coords.append(2*(xy-lo)/(hi-lo)-1)
        return np.stack(fields),np.stack(masks),np.stack(coords)

    def __getitem__(self,item):
        case,start=self.samples[item];frames=[self.frame(case,i) for i in range(start-self.history,start+self.horizon)]
        fields,masks,coords=[np.stack(x) for x in zip(*frames)]
        inputs=np.concatenate([fields[:self.history],masks[:self.history],coords[:self.history]],axis=2)
        target_geometry=np.concatenate([masks[self.history:],coords[self.history:]],axis=2)
        return dict(history=torch.from_numpy(inputs),target=torch.from_numpy(fields[self.history:]),geometry=torch.from_numpy(target_geometry),mach=torch.tensor(float(case[2:]),dtype=torch.float32),case=case,start=start)


def finalize(root):
    root=Path(root);cases={};total=np.zeros(5);sq=np.zeros(5);count=0;lo=np.full(2,np.inf);hi=-lo
    for case in sorted(root.glob('Ma*')):
        records=json.loads((case/'frames.json').read_text());cases[case.name]=records
        if len(records) not in (96,101):raise ValueError(f'Incomplete trajectory {case.name}: {len(records)}')
        for i,r in enumerate(records):
            if i and r['time']<=records[i-1]['time']:raise ValueError('Time order is not strictly increasing')
            with h5py.File(root/r['file'],'r+') as f:
                if abs(f.attrs['time']-r['time'])>1e-8:raise ValueError('Time mismatch')
                # Audited inverse fusion: geometric partition weighted by donor resolution.
                # Applied after all workers finish so resume and parallel shards use one version.
                for name in NAMES['taoca']:
                    c=f['taoca'][name]
                    if c.attrs.get('fusion','')=='geometric_over_h2':continue
                    ids=c['inv_ids'][:];valid=ids[:,0]>=0
                    xy=c['xy'][:].reshape(2,-1).T.astype('f8');p=xy[ids[valid]]
                    h2=np.max(np.sum((p-np.roll(p,1,axis=1))**2,axis=2),axis=1)
                    weights=c['inv_weight'][:];weights[valid]/=np.maximum(h2,1e-24)
                    c['inv_weight'][:]=weights;c.attrs['fusion']='geometric_over_h2'
                if i<60:
                    v=f['source/fields'][:].astype('f8');total+=v.sum(axis=1);sq+=(v*v).sum(axis=1);count+=v.shape[1]
                    xy=np.stack([f['source/x'][:],f['source/y'][:]])
                    lo=np.minimum(lo,xy.min(axis=1));hi=np.maximum(hi,xy.max(axis=1))
    if len(cases)!=5 or sum(map(len,cases.values()))!=500:raise ValueError('Expected five trajectories / 500 frames')
    mean=total/count;std=np.sqrt(np.maximum(sq/count-mean**2,1e-12))
    meta=dict(cases=cases,fields=['Pressure','Density','Temperature','VelocityX','VelocityY'],bounds=[lo.tolist(),hi.tolist()],normalization=dict(mean=mean.tolist(),std=std.tolist(),source='unweighted source nodes, frames [0,60) only; identical for both routes'),split=dict(train=[0,60],val=[60,80],test='[80,N)'),history=20,horizon=10,protocol='known target geometry conditional flow forecasting; first frame anchor; no future flow inputs; overlapping windows within each target split',time_source='filename: exported SOLUTIONTIME is zero',inverse_fusion='geometric_over_h2',sampling_budget=dict(taoca=[4,48,256],cartesian=[1,192,256],nodes_per_route=49152))
    (root/'manifest.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    return meta
