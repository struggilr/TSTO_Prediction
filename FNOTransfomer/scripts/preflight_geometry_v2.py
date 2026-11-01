import os,sys,json
os.environ['MKL_THREADING_LAYER']='GNU'
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from taoca.physics_data import PhysicsWindows
from taoca.spatial_physics import SpatialPredictor
from taoca.physics_evaluate import polygons,confidence
from taoca.inverse_validated import validated_inverse
def main():
    torch.set_num_threads(4);ds=PhysicsWindows(ROOT/'datasets/joint_nearest',ROOT/'datasets/fenli_paired','taoca','test');model=SpatialPredictor(ROOT/'outputs/shared_trajectory/model.json').cuda().eval();rows=[]
    for case in sorted(set(c for c,s in ds.samples)):
        ids=[i for i,(c,s) in enumerate(ds.samples) if c==case]
        for index in sorted(set([ids[0],ids[-1]])):
            item=ds[index];b={k:v[None].cuda() if torch.is_tensor(v) else v for k,v in item.items()}
            with torch.no_grad():out=model(*[b[k] for k in ['history','mach','pose_history','dt','last_xy','templates']],10,'taoca',b['last_base'],b['last_valid'])
            for t in [0,9]:
                mask=out['geometry'][0,t,:,0].cpu().numpy()>.5;xy=out['xy'][0,t].cpu().numpy();obs=polygons(item['templates'].numpy(),out['pose'][0,t].cpu().numpy())
                rec,support,stats=validated_inverse(np.ones((4,1,*mask.shape[1:]),'f4'),xy,mask,obs,confidence(mask,xy));valid=item['target_valid'][t].numpy()>.5;common=valid&support
                coverage=float(common.sum()/valid.sum());error=float(np.max(abs(rec[0,common]-1)))
                if coverage<.99 or error>2e-6:raise RuntimeError(f'Geometry preflight failed {case} {coverage} {error}')
                if any(r['nonpositive'] for r in stats):raise RuntimeError('Nonpositive chart triangles')
                rows.append(dict(case=case,start=item['start'],lead=t+1,coverage=coverage,constant_max_error=error))
            print(case,item['start'],flush=True)
    report=dict(verified=True,scope='First and last test window per Mach, first and last lead; not full-trajectory proof',frames=len(rows),minimum_coverage=min(r['coverage'] for r in rows),rows=rows)
    (ROOT/'outputs/physics_v2/geometry_preflight.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
if __name__=='__main__':main()
