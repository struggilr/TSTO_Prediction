"""Cache geometry-only maps for train/validation; no test-field selection."""
import os
os.environ['MKL_THREADING_LAYER']='GNU'
import sys,json,time,hashlib,argparse,gc
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from taoca.allocated_data_v3 import AllocatedWindows
from taoca.staged import FrozenTrajectory
from taoca.joint import future_xy,normalize_xy,solid_mask
from taoca.chart_transfer_v3 import build_maps
from taoca.physical_supervision_v7 import pack_maps
from taoca.physics_evaluate import polygons
OUT=ROOT/'outputs/physical_v7'
def main():
    p=argparse.ArgumentParser();p.add_argument('--limit',type=int);p.add_argument('--split',choices=['train','val','all'],default='all');a=p.parse_args();torch.set_num_threads(2);OUT.mkdir(exist_ok=True);folder=OUT/'operators';folder.mkdir(exist_ok=True);heights=[35,35,140,70];motion=FrozenTrajectory(ROOT/'outputs/shared_trajectory/model.json').cuda();count=0
    contract=dict(heights=heights,stride=4,phase='(lead_index%4,(lead_index//4)%4)',scope='Only train and validation geometry. Operator uses history-only predicted pose, valid triangles and V3 quality fusion; no flow-label-selected queries',source_sha256={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in ['taoca/joint.py','taoca/staged.py','taoca/chart_transfer_v3.py','taoca/physical_supervision_v7.py','taoca/allocation_v3.py','taoca/domain_charts.py','datasets/joint_nearest/manifest.json','outputs/shared_trajectory/model.json']})
    cp=OUT/'operator_contract.json'
    if cp.exists() and json.loads(cp.read_text())!=contract:raise RuntimeError('Cached operator contract changed')
    cp.write_text(json.dumps(contract,indent=2))
    for split in ['train','val']:
        if a.split!='all' and split!=a.split:continue
        ds=AllocatedWindows(ROOT/'datasets/joint_nearest',ROOT/'datasets/fenli_paired','taoca',split,heights=heights)
        for index in range(len(ds)):
            item=ds[index];case=item['case'];start=int(item['start']);path=folder/f'{case}_s{start:03d}.npz'
            if not path.exists():
                with torch.no_grad():
                    pose=motion(item['pose_history'][None].cuda(),item['dt'][None].cuda(),10);xy=future_xy(item['last_xy'][None].cuda(),item['pose_history'][-1: ].cuda(),pose);norm=normalize_xy(xy);valid=(norm.abs()<=1+2e-6).all(-1)&~solid_mask(xy,pose,item['templates'][None].cuda())
                arrays={}
                for t in range(10):
                    points=[xy[0,t,k,:h].cpu().numpy() for k,h in enumerate(heights)];masks=[valid[0,t,k,:h].cpu().numpy() for k,h in enumerate(heights)];maps=build_maps(points,masks,polygons(item['templates'].numpy(),pose[0,t].cpu().numpy()));op=pack_maps(maps,4,(t%4,(t//4)%4))
                    if len(op['q'])<7000:raise RuntimeError('Unexpected low physical support')
                    arrays.update({f't{t}_{k}':v for k,v in op.items()})
                tmp=path.with_suffix('.partial.npz');np.savez_compressed(tmp,**arrays);tmp.replace(path)
            count+=1;(OUT/'prepare_status.json').write_text(json.dumps(dict(state='preparing',split=split,window=index+1,total_windows=len(ds),completed=count,pid=os.getpid())))
            print(json.dumps(dict(split=split,window=index+1,total=len(ds),completed=count)),flush=True)
            if a.limit and count>=a.limit:return
            del item;gc.collect()
    (OUT/'prepare_status.json').write_text(json.dumps(dict(state='complete',windows=count,pid=os.getpid())));print('Operators ready',flush=True)
if __name__=='__main__':main()
