"""Metric identities, history-only inputs and warm-start parity on a real window."""
import os
os.environ['MKL_THREADING_LAYER']='GNU'
import sys,json
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from taoca.metric_features_v8 import metric_features,MetricEncoder
from taoca.metric_model_v8 import MetricPredictor
from taoca.allocated_model_v3 import AllocatedPredictor
from taoca.allocated_data_v3 import AllocatedWindows
def main():
    torch.set_num_threads(2);torch.manual_seed(42);yy,xx=torch.meshgrid(torch.arange(15,dtype=torch.double),torch.arange(19,dtype=torch.double),indexing='ij');x=torch.zeros(1,8,15,19,dtype=torch.double);x[:,6]=(2*xx+.5*yy)/55;x[:,7]=(3*yy)/35;m=metric_features(x)
    assert torch.allclose(m[:,0],torch.ones_like(xx),atol=1e-12) and torch.allclose(m[:,1],torch.zeros_like(xx),atol=1e-12)
    scale=(110/439*70/279)**.5;assert torch.allclose(m[:,7],torch.full_like(xx,np.log(6/(scale*scale))),atol=1e-12)
    shifted=x.clone();shifted[:,6]+=10;shifted[:,7]-=20;assert torch.allclose(metric_features(shifted),m,atol=1e-11)
    rng=torch.get_rng_state();layer=MetricEncoder(24).double();assert torch.equal(rng,torch.get_rng_state());bias=layer(x,(8,10));assert not bias.count_nonzero();bias.sum().backward();assert layer.net[-1].weight.grad.abs().sum()>0
    heights=[35,35,140,70];ds=AllocatedWindows(ROOT/'datasets/joint_nearest',ROOT/'datasets/fenli_paired','taoca','val',heights=heights);item=ds[0];b={k:v[None].cuda() if torch.is_tensor(v) else v for k,v in item.items()}
    base=AllocatedPredictor(ROOT/'outputs/shared_trajectory/model.json',heights).cuda().eval();new=MetricPredictor(ROOT/'outputs/shared_trajectory/model.json',heights).cuda().eval();state=torch.load(ROOT/'outputs/atlas_v3/taoca/best.pt',map_location='cuda',weights_only=True);base.load_state_dict(state);missing,extra=new.load_state_dict(state,strict=False);assert not extra and all(k.startswith('metric_encoder.') for k in missing)
    keys=['history','mach','pose_history','dt','last_xy','templates']
    with torch.inference_mode():
        a=base(*[b[k] for k in keys],10,'taoca',b['last_base'],b['last_valid']);c=new(*[b[k] for k in keys],10,'taoca',b['last_base'],b['last_valid'])
    assert torch.equal(a['fields'],c['fields']) and torch.equal(a['pose'],c['pose'])
    p0=sum(p.numel() for p in base.parameters());p1=sum(p.numel() for p in new.parameters());out=ROOT/'outputs/diagnosis_v8';out.mkdir(exist_ok=True);report=dict(verified=True,affine_metric_and_jacobian=True,translation_invariance=True,zero_initialized_branch_has_gradient=True,real_window_warm_start_bit_exact=True,base_parameters=p0,candidate_parameters=p1,additional_parameters=p1-p0,scope='Candidate implementation checks only, not prediction-performance evidence. Branch uses observed history coordinates; no future truth inputs.')
    (out/'metric_candidate_tests.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
if __name__=='__main__':main()
