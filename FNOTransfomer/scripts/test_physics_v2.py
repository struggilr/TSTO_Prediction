"""Meaningful regression gates: valid donors, obstacle exclusion, spatial derivatives, labels."""
import os,sys,json
os.environ['MKL_THREADING_LAYER']='GNU'
from pathlib import Path
import numpy as np
import torch
import h5py
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from taoca.inverse_validated import validated_inverse
from taoca.physics_losses import physical_gradient,PhysicsLoss
from taoca.physics_data import PhysicsWindows
from taoca.spatial_physics import SpatialPredictor
from taoca.physics_evaluate import polygons,confidence

def main():
    torch.set_num_threads(4);results={}
    y,x=np.meshgrid(np.linspace(0,4,9),np.linspace(0,4,9),indexing='ij');xy=np.stack([x,y],-1)[None].repeat(4,0);mask=np.ones(xy.shape[:-1],bool)
    values=np.ones((4,1,9,9),'f4');values[:,:,4,4]=10000;mask[:,4,4]=False
    query=np.array([[.2,.2],[1.2,.3],[2.,2.],[3.4,3.3]])
    rec,support,_=validated_inverse(values,xy,mask,query=query)
    assert np.max(abs(rec[:,support]-1))<1e-6 and not support[2]
    obstacle=[np.array([[0,0],[1,0],[1,1],[0,1]])]
    _,s,_=validated_inverse(np.ones_like(values),xy,np.ones_like(mask),obstacle,query=query)
    assert not s[0]
    results['invalid_donor_and_solid_rejection']=True
    coords=torch.tensor(xy[:1][None,None],dtype=torch.float32);q=(3*coords[...,0]+2*coords[...,1])[:,:,:,None]
    dx,dy,ok=physical_gradient(q,coords);assert torch.allclose(dx,torch.full_like(dx,3)) and torch.allclose(dy,torch.full_like(dy,2));results['physical_gradient_affine']=True
    meta=json.loads((ROOT/'datasets/joint_nearest/manifest.json').read_text());thermo=json.loads((ROOT/'outputs/physics_v2/thermodynamics.json').read_text())
    lossfn=PhysicsLoss(meta,thermo);base=torch.zeros(1,3,1,5,9,9);physical=base+torch.tensor(meta['mean'])[None,None,None,:,None,None]
    moving=coords.expand(-1,3,-1,-1,-1,-1).clone()+torch.arange(1,4)[None,:,None,None,None,None]*.1
    hist=torch.zeros(1,20,1,8,9,9);hist[:,:,:,5]=1
    cont=lossfn.continuity(dict(fields=base,xy=moving),dict(history=hist,last_xy=coords[:,0],dt=torch.full((1,19),.01)),physical,torch.ones(1,3,1,1,9,9,dtype=torch.bool),torch.zeros(1,3,1,1,9,9),torch.zeros(1,3,1,1,9,9),torch.ones(1,3,1,1,9,9))
    assert cont.item()<1e-10;results['uniform_state_moving_geometry_continuity']=float(cont)
    device=torch.device('cuda');batch={}
    for route in ['cartesian','taoca']:
        ds=PhysicsWindows(ROOT/'datasets/joint_nearest',ROOT/'datasets/fenli_paired',route,'train');item=ds[0]
        b={k:v[None].to(device) if torch.is_tensor(v) else v for k,v in item.items()};batch[route]=b
        model=SpatialPredictor(ROOT/'outputs/shared_trajectory/model.json').to(device).eval()
        args=[b[k] for k in ['history','mach','pose_history','dt','last_xy','templates']]
        with torch.no_grad():out=model(*args,10,route,b['last_base'],b['last_valid'])
        assert out['fields'].shape[1]==10 and torch.isfinite(out['fields']).all()
        # Future labels are absent from the forward signature; they cannot alter the result.
        b['target']=torch.randn_like(b['target'])*100
        with torch.no_grad():again=model(*args,10,route,b['last_base'],b['last_valid'])
        assert torch.equal(out['fields'],again['fields'])
        results[route+'_no_future_label_leakage']=True
        if route=='taoca':
            mask=out['geometry'][0,0,:,0].cpu().numpy()>.5;p=out['xy'][0,0].cpu().numpy();obs=polygons(item['templates'].numpy(),out['pose'][0,0].cpu().numpy())
            v=np.ones((4,1,*mask.shape[1:]),'f4');v[~mask[:,None]]=10000
            z,s,stats=validated_inverse(v,p,mask,obs,confidence(mask,p));valid=item['target_valid'][0].numpy()>.5
            assert np.max(abs(z[0,s&valid]-1))<2e-6
            results['real_geometry_masked_constant_max_error']=float(np.max(abs(z[0,s&valid]-1)));results['real_geometry_valid_coverage']=float((s&valid).sum()/valid.sum());results['donor_stats']=stats
        del model,out,again
        torch.cuda.empty_cache()
    assert torch.equal(batch['cartesian']['pose_history'],batch['taoca']['pose_history'])
    (ROOT/'outputs/physics_v2/verification.json').write_text(json.dumps(results,indent=2));print(json.dumps(results),flush=True)
if __name__=='__main__':main()
