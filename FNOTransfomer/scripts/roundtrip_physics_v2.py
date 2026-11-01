"""Fixed representative subset: truth -> predicted atlas -> physical grid diagnostics."""
import os,sys,json
os.environ['MKL_THREADING_LAYER']='GNU'
from pathlib import Path
import numpy as np
import torch
import h5py
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from taoca.physics_data import PhysicsWindows
from taoca.joint import future_xy,normalize_xy,solid_mask,sample_grid
from taoca.physics_losses import regional_maps
from taoca.physics_evaluate import confidence,polygons
from taoca.inverse_validated import validated_inverse
def main():
    torch.set_num_threads(4);ds=PhysicsWindows(ROOT/'datasets/joint_nearest',ROOT/'datasets/fenli_paired','taoca','test');groups={};details=[];root=ROOT/'outputs/physics_v2'
    for case in sorted(set(c for c,s in ds.samples)):
        index=next(i for i,(c,s) in enumerate(ds.samples) if c==case);item=ds[index];start=item['start']
        with h5py.File(root/'taoca/predictions'/f'{case}_s{start:03d}.h5') as f:pose=torch.from_numpy(f['pose'][:])[None];pred=f['predicted'][:]
        coords=future_xy(item['last_xy'][None],item['pose_history'][-1][None],pose)
        valid=(normalize_xy(coords).abs()<=1+2e-6).all(-1)&~solid_mask(coords,pose,item['templates'][None])
        truth=sample_grid(item['target'][None],coords)[0].numpy();targetmask=sample_grid(item['target_valid'][None,:,None],coords)[0,:, :,0].numpy()>.999
        near,shock=regional_maps(item['target'][None],item['target_valid'][None]);near=near[0,:,0].numpy()>.5;shock=shock[0,:,0].numpy()>.5
        for t in [0,9]:
            mask=valid[0,t].numpy()&targetmask[t];xy=coords[0,t].numpy();obs=polygons(item['templates'].numpy(),pose[0,t].numpy())
            rec,support,_=validated_inverse(truth[t],xy,mask,obs,confidence(mask,xy));common=support&(item['target_valid'][t].numpy()>.5)&np.isfinite(pred[t]).all(0)
            target=item['target'][t].numpy()
            for key,m in [('all',common),('wall',common&near[t]),('shock',common&shock[t])]:
                g=groups.setdefault(key,dict(n=0,roundtrip=np.zeros(5),forecast=np.zeros(5)));g['n']+=int(m.sum());g['roundtrip']+=((rec[:,m]-target[:,m])**2).sum(1);g['forecast']+=((pred[t][:,m]-target[:,m])**2).sum(1)
            details.append(dict(case=case,start=start,lead=t+1,coverage=float(common.sum()/(item['target_valid'][t].numpy()>.5).sum())))
    result=dict(scope='First test window of each Mach at leads 1 and 10; 10 overlapping forecast instances. Roundtrip is a diagnostic, not an irreducible lower bound or additive RMSE component.',regions={k:dict(n=v['n'],roundtrip_nrmse=np.sqrt(v['roundtrip']/v['n']).tolist(),forecast_nrmse_on_same_subset=np.sqrt(v['forecast']/v['n']).tolist()) for k,v in groups.items()},frames=details)
    (root/'roundtrip_diagnostic.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
