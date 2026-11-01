"""Locked test evaluation of fixed Cartesian, old TAOCA, reallocated TAOCA."""
import os,sys,json,hashlib
os.environ['MKL_THREADING_LAYER']='GNU'
from pathlib import Path
import numpy as np
import h5py
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from taoca.physics_losses import regional_maps
def main():
    torch.set_num_threads(4);root=ROOT/'outputs/local_v9';v2=ROOT/'outputs/physics_v2';run=json.loads((root/'taoca/run.json').read_text());a=json.loads((v2/'cartesian/run.json').read_text())
    if hashlib.sha256((v2/'cartesian/best.pt').read_bytes()).hexdigest()!=run['cartesian_checkpoint_sha256']:raise RuntimeError('Cartesian checkpoint changed')
    assert run['parameters']==a['parameters']+6405
    for key in ['seed','batch_size']:assert run['arguments'][key]==a['arguments'][key]
    paths=sorted((root/'taoca/predictions').glob('*.h5'));assert len(paths)==54
    totals={};rows=[];reference_count=0;v7_sse=np.zeros(5);v7_count=0
    for path in paths:
        with h5py.File(path) as b,h5py.File(v2/'cartesian/predictions'/path.name) as fa,h5py.File(ROOT/'outputs/atlas_v3/taoca/predictions'/path.name) as fb:
            ref=b['reference'][:];valid=b['valid'][:].astype(bool)
            assert np.array_equal(ref,fa['reference'][:]) and np.array_equal(ref,fb['reference'][:])
            assert np.array_equal(b['pose'][:],fa['pose'][:])
            predictions={'FNO_Cartesian':fa['predicted'][:],'FNO_TAOCA_V3':fb['predicted'][:],'FNO_TAOCA_V9':b['predicted'][:],'Persistence':np.broadcast_to(b['persistence'][:],ref.shape)}
            common=valid&b['support'][:]&fa['support'][:]&fb['support'][:]
            with h5py.File(ROOT/'outputs/physical_v7/taoca/predictions'/path.name) as fv7:
                assert np.array_equal(ref,fv7['reference'][:]);previous=fv7['predicted'][:];common&=fv7['support'][:]&np.isfinite(previous).all(1)
            for values in predictions.values():common&=np.isfinite(values).all(1)
            wall,shock=regional_maps(torch.from_numpy(ref)[None],torch.from_numpy(valid.astype('f4'))[None]);wall=wall[0,:,0].numpy()>.5;shock=shock[0,:,0].numpy()>.5
            reference_count+=int(valid.sum());v7_sse+=np.square(previous.transpose(1,0,2,3)[:,common].astype('f8')-ref.transpose(1,0,2,3)[:,common]).sum(1);v7_count+=int(common.sum())
            for t in range(10):
                row=dict(window=path.stem,lead=t+1,n=int(common[t].sum()))
                for region,mask in [('all',common[t]),('wall',common[t]&wall[t]),('shock',common[t]&shock[t])]:
                    g=totals.setdefault(region,dict(n=0,models={name:np.zeros(5) for name in predictions}));g['n']+=int(mask.sum())
                    for name,values in predictions.items():
                        error=((values[t][:,mask]-ref[t][:,mask])**2).sum(1);g['models'][name]+=error
                        if region=='all':row[name]=np.sqrt(error/max(1,int(mask.sum()))).tolist()
                rows.append(row)
    result=dict(forecast_instances=len(rows),common_coverage=totals['all']['n']/reference_count,heights=run['heights'],sampled_points=run['sampled_points'],parameters=run['parameters'],regions={region:dict(n=g['n'],nrmse={name:np.sqrt(value/max(1,g['n'])).tolist() for name,value in g['models'].items()}) for region,g in totals.items()},shared_trajectory_exact=True,cartesian_checkpoint_unchanged=True,scope='Single seed; TAOCA V9 has 6405 additional parameters and cumulative extra fine-tuning budget from V3 through V8; not equal-budget from a V3 checkpoint. This is an enhanced-system comparison, not a pure representation/equal-budget ablation. All overlapping instances on common legal support; repeatedly viewed test set remains exploratory.')
    result['v7_same_support_all_nrmse']=np.sqrt(v7_sse/max(1,v7_count)).tolist();result['all_five_global_channels_better']=bool(all(x<y for x,y in zip(result['regions']['all']['nrmse']['FNO_TAOCA_V9'],result['regions']['all']['nrmse']['FNO_Cartesian'])));result['pressure_density_shock_better']=bool(all(x<y for x,y in zip(result['regions']['shock']['nrmse']['FNO_TAOCA_V9'][:2],result['regions']['shock']['nrmse']['FNO_Cartesian'][:2])));(root/'comparison.json').write_text(json.dumps(result,indent=2));(root/'test_frame_metrics.json').write_text(json.dumps(rows));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
