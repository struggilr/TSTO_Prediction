"""Representation gate on all validation windows, before any candidate training."""
import os,sys,json,time
os.environ['MKL_THREADING_LAYER']='GNU'
from pathlib import Path
import numpy as np
import h5py
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from taoca.allocation_v3 import allocate
from taoca.joint import normalize_xy,solid_mask
from taoca.physics_evaluate import polygons
from taoca.physics_losses import regional_maps
from taoca.chart_transfer_v3 import build_maps,apply_maps,fuse
OUT=ROOT/'outputs/atlas_v3'

def main():
    torch.set_num_threads(4);plan=json.loads((OUT/'selection_contract.json').read_text());allocations=plan['representation_candidates'];fusion=json.loads((OUT/'fusion_diagnosis.json').read_text())['fusion_selected'];names=['_'.join(map(str,h)) for h in allocations]
    groups={name:{region:dict(n=0,sse=np.zeros(5)) for region in ['all','wall','shock']} for name in names};cover={name:dict(legal=0,truth_support=0,reference=0,nonpositive=0) for name in names};begin=time.perf_counter()
    frames=0
    for window,path in enumerate(sorted((OUT/'validation_native').glob('*.h5'))):
        with h5py.File(path) as f:xy=torch.from_numpy(f['xy'][:]);pose=torch.from_numpy(f['pose'][:]);templates=torch.from_numpy(f['templates'][:]);target=torch.from_numpy(f['reference'][:]);valid=torch.from_numpy(f['valid'][:]);case=f.attrs['case']
        near,shock=regional_maps(target[None],valid[None]);near=near[0,:,0].numpy()>.5;shock=shock[0,:,0].numpy()>.5
        prepared=[]
        for heights in allocations:
            coords=allocate(xy,heights);sampled=[];masks=[];truth_masks=[]
            for p in coords:
                grid=normalize_xy(p);sampled.append(F.grid_sample(target,grid,align_corners=True).numpy())
                tv=F.grid_sample(valid[:,None],grid,align_corners=True)[:,0]>.999
                geom=(grid.abs()<=1+2e-6).all(-1)&~solid_mask(p[None,:,None],pose[None],templates[None])[0,:,0]
                masks.append(geom.numpy());truth_masks.append(tv.numpy())
            prepared.append((coords,sampled,masks,truth_masks))
        for t in range(10):
            outputs=[];supports=[];fluid=valid[t].numpy()>.5
            for name,(coords,sampled,masks,truth_masks) in zip(names,prepared):
                p=[c[t].numpy() for c in coords];maps=build_maps(p,[m[t] for m in masks],polygons(templates.numpy(),pose[t].numpy()))
                values,q=apply_maps([a[t] for a in sampled],maps,[m[t] for m in truth_masks]);rec,support=fuse(values,q,fusion)
                legal=np.zeros((280*440,),bool)
                for mm in maps:legal[mm['q']]=True
                cover[name]['legal']+=int((legal.reshape(280,440)&fluid).sum());cover[name]['truth_support']+=int((support&fluid).sum());cover[name]['reference']+=int(fluid.sum());cover[name]['nonpositive']+=sum(m['stats']['nonpositive'] for m in maps)
                outputs.append(rec);supports.append(support)
            common=fluid&np.logical_and.reduce(supports)
            for name,rec in zip(names,outputs):
                for region,mask in [('all',common),('wall',common&near[t]),('shock',common&shock[t])]:
                    g=groups[name][region];g['n']+=int(mask.sum());g['sse']+=((rec[:,mask]-target[t].numpy()[:,mask])**2).sum(1)
            frames+=1
        print(json.dumps(dict(validation_windows=window+1,case=case,frames=frames)),flush=True)
    metrics={name:{region:dict(n=g['n'],nrmse=np.sqrt(g['sse']/max(1,g['n'])).tolist()) for region,g in regions.items()} for name,regions in groups.items()}
    for name,c in cover.items():c['legal_coverage']=c['legal']/c['reference'];c['truth_supported_coverage']=c['truth_support']/c['reference']
    base=metrics[names[0]];candidates=[]
    for name,heights in zip(names,allocations):
        m=metrics[name];gain=1-np.mean(m['all']['nrmse'])/np.mean(base['all']['nrmse']);shockgain=1-np.mean(m['shock']['nrmse'][:2])/np.mean(base['shock']['nrmse'][:2]);wallratio=np.mean(m['wall']['nrmse'])/np.mean(base['wall']['nrmse'])
        passed=bool(gain>=.05 and shockgain>=.10 and wallratio<=1.10 and cover[name]['truth_supported_coverage']>=cover[names[0]]['truth_supported_coverage']-.002 and cover[name]['legal_coverage']>=.995 and cover[name]['nonpositive']==0)
        candidates.append(dict(name=name,heights=heights,mean_roundtrip_gain=float(gain),shock_p_rho_gain=float(shockgain),wall_ratio=float(wallratio),passed=passed))
    allowed=[c for c in candidates if c['passed']];selected=max(allowed,key=lambda c:c['mean_roundtrip_gain']) if allowed else None
    result=dict(split='validation',forecast_instances=frames,fusion=fusion,common_support_comparison=True,metrics=metrics,coverage=cover,candidates=candidates,selected=selected,training_authorized_by_gate=selected is not None,seconds=time.perf_counter()-begin)
    (OUT/'allocation_selection.json').write_text(json.dumps(result,indent=2));print(json.dumps(dict(candidates=candidates,selected=selected),indent=2),flush=True)
if __name__=='__main__':main()
