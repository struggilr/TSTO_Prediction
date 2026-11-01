"""Check preserved targets/geometry and exclusion of future raw-field inputs."""
import os
os.environ['MKL_THREADING_LAYER']='GNU'
import sys,json
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from taoca.direct_history_v6 import DirectHistoryWindows
from taoca.allocated_data_v3 import AllocatedWindows
def main():
    torch.set_num_threads(4);heights=[35,35,140,70];reports=[]
    for split in ['train','val','test']:
        args=[ROOT/'datasets/joint_nearest',ROOT/'datasets/fenli_paired','taoca',split]
        old=AllocatedWindows(*args,heights=heights);new=DirectHistoryWindows(*args,heights=heights);a=old[0];b=new[0];case,s=new.samples[0]
        for key in ['target','target_valid','target_pose','last_base','last_valid','pose_history','dt','last_xy','templates','mach']:assert torch.equal(a[key],b[key]),key
        assert torch.equal(a['history'][:,:,6:],b['history'][:,:,6:])
        assert torch.isfinite(b['history']).all()
        assert torch.all(b['history'][:,:,5]<=a['history'][:,:,5])
        changed=float((a['history'][:,:,:5]-b['history'][:,:,:5]).abs().mean());assert changed>1e-6
        key=(str(new.direct_root.resolve()),case);saved=[]
        for field,valid in new._direct[key]:
            saved.append(field[s:s+new.horizon].clone());field[s:s+new.horizon]+=123
        c=new[0];assert torch.equal(b['history'],c['history']), 'Future field leakage'
        for (field,valid),backup in zip(new._direct[key],saved):field[s:s+new.horizon]=backup
        reports.append(dict(split=split,windows=len(new),history_field_mean_abs_change=changed,target_and_geometry_exact=True,future_field_poisoning_passed=True))
    out=ROOT/'outputs/direct_history_v6/implementation_tests.json';out.write_text(json.dumps(dict(verified=True,results=reports),indent=2));print(out.read_text(),flush=True)
if __name__=='__main__':main()
