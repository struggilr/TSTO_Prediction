"""Check temporal alignment, chart roles, loss masking and parameter equality."""
import sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from taoca.model import Predictor
from train_compare import loss_fn


def main():
    torch.set_num_threads(2);torch.manual_seed(42)
    model=Predictor(width=8,modes=4,latent=32,layers=1,temporal_layers=1).eval()
    for k,h,w in [(1,32,48),(4,16,48)]:
        history=torch.randn(1,20,k,8,h,w);geometry=torch.randn(1,10,k,3,h,w);geometry[:,:,:,0]=1
        with torch.no_grad():
            y=model(history,torch.tensor([5.]),geometry)
            changed=history.clone();changed[:,0,:,:5]+=5
            y_changed=model(changed,torch.tensor([5.]),geometry)
        assert y.shape==(1,10,k,5,h,w)
        assert (y-y_changed).abs().max()>1e-7,'Model ignored earlier history'
        assert torch.isfinite(y).all()
        target=y.clone();target[:,:,:,:,0,0]+=100
        geometry[:,:,:,:,0,0]=0
        assert loss_fn(y,target,geometry)==0,'Invalid point contributed to loss'
        # Preserve output ordering: every chart/time block gets its own supplied mask.
        geometry[:,:,0,0]=0
        with torch.no_grad():out=model(history,torch.tensor([5.]),geometry)
        assert torch.count_nonzero(out[:,:,0])==0,'Future geometry mask misaligned'
    print('PASS: 20-frame history dependence, dimensions, masks, finite outputs, both chart counts')


if __name__=='__main__':main()
