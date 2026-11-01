"""Parameter parity, unchanged Cartesian, no encoded padding, finite learning."""
import os,sys,json
os.environ['MKL_THREADING_LAYER']='GNU'
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from taoca.spatial_physics import SpatialPredictor
from taoca.allocated_model_v3 import AllocatedPredictor
from taoca.allocated_data_v3 import AllocatedWindows
from taoca.physics_losses import PhysicsLoss

def main():
    torch.set_num_threads(4);torch.manual_seed(42);selection=ROOT/'outputs/atlas_v3/allocation_selection.json'
    chosen=json.loads(selection.read_text())['selected'] if selection.exists() else None
    heights=chosen['heights'] if chosen else [42,42,126,70];trajectory=ROOT/'outputs/shared_trajectory/model.json';device='cuda';report={'tested_heights':heights}
    old=SpatialPredictor(trajectory).cuda().eval();new=AllocatedPredictor(trajectory,heights).cuda().eval();new.load_state_dict(old.state_dict())
    for route in ['cartesian','taoca']:
        ds=AllocatedWindows(ROOT/'datasets/joint_nearest',ROOT/'datasets/fenli_paired',route,'train',heights=heights);item=ds[0];b={k:v[None].cuda() if torch.is_tensor(v) else v for k,v in item.items()}
        args=[b[k] for k in ['history','mach','pose_history','dt','last_xy','templates']]
        if route=='cartesian':
            with torch.no_grad():a=old(*args,10,route,b['last_base'],b['last_valid']);c=new(*args,10,route,b['last_base'],b['last_valid'])
            assert torch.equal(a['fields'],c['fields']);report['unchanged_cartesian_bit_exact']=True
        else:
            inputs=[];handle=new.fine.register_forward_pre_hook(lambda module,args:inputs.append(tuple(args[0].shape)))
            with torch.no_grad():a=new(*args,10,route,b['last_base'],b['last_valid'])
            handle.remove();points=sum(x[0]*x[2]*x[3] for x in inputs)//20
            assert points==123200;report['encoded_points_per_history_frame']=points;report['encoder_inputs']=inputs
            noisy=b['history'].clone()
            for k,h in enumerate(heights):noisy[:,:,k,:,h:]=torch.randn_like(noisy[:,:,k,:,h:])*100
            with torch.no_grad():c=new(noisy,*args[1:],10,route,b['last_base'],b['last_valid'])
            assert torch.equal(a['fields'],c['fields']);report['padding_does_not_enter_encoder']=True
            meta=json.loads((ROOT/'datasets/joint_nearest/manifest.json').read_text());thermo=json.loads((ROOT/'outputs/physics_v2/thermodynamics.json').read_text());lossfn=PhysicsLoss(meta,thermo)
            optimizer=torch.optim.AdamW(new.parameters(),lr=3e-4);losses=[]
            for step in range(3):
                new.train();optimizer.zero_grad(set_to_none=True);out=new(*args,10,route,b['last_base'],b['last_valid']);loss,parts=lossfn(out,b,1);loss.backward();assert torch.isfinite(loss)
                if step==2:
                    grad=float(new.spectral[0].positive.grad[:,:,1:,1:].abs().sum());assert grad>0;report['nonzero_fourier_gradient_l1']=grad
                torch.nn.utils.clip_grad_norm_(new.parameters(),1);optimizer.step();losses.append(float(loss.detach()))
            report['three_real_optimizer_steps']=losses
        del a,c,b
    report['parameters']=sum(p.numel() for p in new.parameters());assert report['parameters']==sum(p.numel() for p in old.parameters())==340349
    report['verified']=True;out=ROOT/'outputs/atlas_v3';out.mkdir(exist_ok=True);(out/'implementation_tests.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
if __name__=='__main__':main()
