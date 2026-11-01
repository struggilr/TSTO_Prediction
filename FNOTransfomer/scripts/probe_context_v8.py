"""Train-geometry alignment audit and validation-only frozen context ablation."""
import os
os.environ['MKL_THREADING_LAYER']='GNU'
import sys,json,io,hashlib,time,importlib
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from taoca.allocated_data_v3 import AllocatedWindows
from taoca.allocated_model_v3 import AllocatedPredictor
from taoca.physical_supervision_v7 import project,load_window
from taoca.physics_losses import regional_maps
OUT=ROOT/'outputs/diagnosis_v8'
def main():
    OUT.mkdir(exist_ok=True);torch.set_num_threads(2)
    text=(ROOT/'taoca/allocated_model_v3.py').read_text().replace('class AllocatedPredictor(SpatialPredictor):','class ContextProbe(SpatialPredictor):').replace('self.heights=list(heights)','self.heights=list(heights);self.context_scale=.25').replace('z=maps[j]+.25*context','z=maps[j]+self.context_scale*context')
    module=ROOT/'taoca/context_ablation_v8.py';module.write_text(text,encoding='utf-8');ContextProbe=importlib.import_module('taoca.context_ablation_v8').ContextProbe
    heights=[35,35,140,70];args=[ROOT/'datasets/joint_nearest',ROOT/'datasets/fenli_paired','taoca'];train=AllocatedWindows(*args,'train',heights=heights);item=train[0];record=train.record(item['case']);from taoca.allocation_v3 import allocate
    coords=allocate(record['coords'],heights);s=int(item['start']);points=[p[s-20:s,::4,::4] for p in coords];distances=[]
    for i,target in enumerate(points):
        for j,source in enumerate(points):
            if i==j:continue
            warped=F.interpolate(source.permute(0,3,1,2),size=target.shape[1:3],mode='bilinear',align_corners=False).permute(0,2,3,1);d=(warped-target).norm(dim=-1).numpy().ravel();distances.append(dict(recipient=i,donor=j,median_distance_m=float(np.median(d)),p95_distance_m=float(np.quantile(d,.95)),fraction_above_half_m=float(np.mean(d>.5))))
    (OUT/'context_coordinate_mismatch.json').write_text(json.dumps(dict(training_window=f'{item["case"]}_s{s:03d}',scope='Centers of convolutional receptive fields, approximate spatial meaning of same-logical-index feature mixing; large distances do not alone prove bad predictions',pairs=distances),indent=2))
    ds=AllocatedWindows(*args,'val',heights=heights);model=ContextProbe(ROOT/'outputs/shared_trajectory/model.json',heights).cuda().eval();old=AllocatedPredictor(ROOT/'outputs/shared_trajectory/model.json',heights).cuda().eval();path=ROOT/'outputs/atlas_v3/taoca/best.pt';raw=path.read_bytes();state=torch.load(io.BytesIO(raw),map_location='cuda',weights_only=True);model.load_state_dict(state);old.load_state_dict(state)
    item=ds[0];b={k:v[None].cuda() if torch.is_tensor(v) else v for k,v in item.items()};keys=['history','mach','pose_history','dt','last_xy','templates']
    with torch.inference_mode():
        a=old(*[b[k] for k in keys],10,'taoca',b['last_base'],b['last_valid']);b0=model(*[b[k] for k in keys],10,'taoca',b['last_base'],b['last_valid'])
    assert torch.equal(a['fields'],b0['fields']),'Default context probe changed baseline';del old,a,b0
    totals={}
    for i in range(len(ds)):
        item=ds[i];b={k:v[None].cuda() if torch.is_tensor(v) else v for k,v in item.items()};ops=load_window(ROOT/'outputs/physical_v7/operators',item['case'],int(item['start']));near,shock=regional_maps(b['target'],b['target_valid'])
        for scale in [.25,.125,0.]:
            model.context_scale=scale
            with torch.inference_mode():out=model(*[b[k] for k in keys],10,'taoca',b['last_base'],b['last_valid'])
            for t,op in enumerate(ops):
                q=torch.as_tensor(op['q'],device='cuda',dtype=torch.long);pred=project(out['fields'][0,t],op);ref=b['target'][0,t].reshape(5,-1)[:,q];valid=b['target_valid'][0,t].reshape(-1)[q]>.5;err=(pred-ref).square()
                for region,mask in [('all',valid),('wall',valid&(near[0,t,0].reshape(-1)[q]>.5)),('shock',valid&(shock[0,t,0].reshape(-1)[q]>.5))]:
                    g=totals.setdefault(f'{scale}/{region}',dict(n=0,sse=np.zeros(5)));g['n']+=int(mask.sum());g['sse']+=err[:,mask].double().sum(1).cpu().numpy()
        print(json.dumps(dict(validation_window=i+1,total=len(ds))),flush=True)
    metrics={k:dict(n=v['n'],nrmse=np.sqrt(v['sse']/max(1,v['n'])).tolist()) for k,v in totals.items()};result=dict(verified_default_bit_exact=True,checkpoint_sha256=hashlib.sha256(raw).hexdigest(),metrics=metrics,scope='Frozen V3 checkpoint, validation-only sparse physical common points; changing context inference without retraining is a diagnostic, not evidence of a trained V8 advantage.')
    (OUT/'context_validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
