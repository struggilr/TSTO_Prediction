import sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from taoca.joint import JointPredictor,joint_loss,solid_mask


def main():
    torch.set_num_threads(2);torch.manual_seed(42)
    model=JointPredictor(width=8,modes=4,latent=32,layers=1,temporal_layers=1).eval()
    template=torch.tensor([[[[-1.,-1.],[1.,-1.],[1.,1.],[-1.,1.]]]*2])
    for route,k in [('cartesian',1),('taoca',4)]:
        history=torch.randn(1,20,k,8,16,32);history[:,:,:,5]=1
        pose=torch.zeros(1,20,6);pose[:,:,0]=20;pose[:,:,3]=30
        args=(history,torch.tensor([5.]),pose,torch.ones(1,19)*.01,torch.rand(1,k,16,32,2)*10,template,10,route)
        out=model(*args)
        targets=dict(target=torch.randn(1,10,5,280,440),target_valid=torch.ones(1,10,280,440),target_pose=torch.randn(1,10,6))
        loss,_=joint_loss(out,targets,route);loss.backward()
        targets['target'].fill_(1234);targets['target_pose'].fill_(-999);targets['target_valid'].zero_()
        again=model(*args)
        assert torch.equal(out['fields'],again['fields']) and torch.equal(out['pose'],again['pose'])
        assert torch.isfinite(loss)
        print(route,'PASS: finite backward, future target mutation does not affect forecast')
    xy=torch.tensor([[[[[[20.,0.],[24.,0.]]]]]])
    mask=solid_mask(xy,pose[:,-1,None],template)
    assert mask.flatten().tolist()==[True,False]
    print('PASS: predicted solid mask')


if __name__=='__main__':main()
