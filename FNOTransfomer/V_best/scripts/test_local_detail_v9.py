import os
os.environ['MKL_THREADING_LAYER']='GNU'
import sys,json
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from taoca.local_detail_v9 import LocalDetail
from taoca.metric_features_v8 import metric_features

def main():
    torch.set_num_threads(2);torch.manual_seed(42)
    module=LocalDetail();x=torch.randn(2,22,13,32,requires_grad=True)
    for periodic in [False,True]:
        out=module(x,periodic);assert out.shape==(2,5,13,32) and torch.equal(out,torch.zeros_like(out))
        (out-torch.randn_like(out)).square().mean().backward();assert module.out.weight.grad.abs().sum()>0;module.zero_grad()
    y,z=torch.meshgrid(torch.linspace(-1,1,13),torch.linspace(-1,1,32),indexing='ij')
    h=torch.zeros(1,8,13,32);h[:,6]=z;h[:,7]=y;features=metric_features(h);assert features.shape==(1,8,13,32) and torch.isfinite(features).all()
    report=dict(verified=True,scope='CPU module-level checks only; full predictor and real-data tests still required before formal training',zero_initial_output=True,finite_gradients=True,local_parameters=sum(v.numel() for v in module.parameters()))
    out=ROOT/'outputs/shock_diagnosis_v9';out.mkdir(exist_ok=True);(out/'local_module_tests.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
if __name__=='__main__':main()
