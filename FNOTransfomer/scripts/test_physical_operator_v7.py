"""Exact evaluation-fusion agreement plus gradients, constants and missing support."""
import os
os.environ['MKL_THREADING_LAYER']='GNU'
import sys,json
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from taoca.physical_supervision_v7 import pack_maps,project,load_window
from taoca.chart_transfer_v3 import apply_maps,fuse
def main():
    rng=np.random.default_rng(42);maps=[];q=np.array([0,4,1760,1764,3520,3524])
    for k in range(4):
        qq=q[k%2:];b=rng.random((len(qq),3));b/=b.sum(1,keepdims=True)
        maps.append(dict(q=qq,ids=rng.integers(0,12,(len(qq),3)),bary=b,quality=rng.random(len(qq))+.1))
    pred=rng.normal(size=(4,5,2,6));charts,quality=apply_maps(pred,maps);reference,support=fuse(charts,quality);op=pack_maps(maps);x=torch.tensor(pred,dtype=torch.float64,requires_grad=True);y=project(x,op);err=float(np.max(abs(y.detach().numpy()-reference.reshape(5,-1)[:,op['q']])))
    assert err<1e-6,err
    assert torch.autograd.gradcheck(lambda a:project(a,op),(x,),fast_mode=True,atol=1e-5,rtol=1e-3)
    ones=project(torch.ones_like(x),op);assert float((ones-1).abs().max())<1e-6
    assert all(support.ravel()[op['q']]);assert len(op['q'])==6
    out=ROOT/'outputs/physical_v7';out.mkdir(exist_ok=True);result=dict(verified=True,numpy_evaluation_fusion_max_error=err,double_precision_gradcheck=True,constant_preservation=True,unsupported_points_excluded=True)
    paths=sorted((out/'operators').glob('*.npz'))
    if paths:
        with np.load(paths[0]) as f:
            for t in range(10):
                op={k.split('_',1)[1]:f[k] for k in f.files if k.startswith(f't{t}_')};z=project(torch.ones(4,5,140,440),op)
                assert float((z-1).abs().max())<1e-5
        result['actual_window_constant_test']=paths[0].stem
    (out/'operator_tests.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
