"""Physical evaluation after trimming all masked bookkeeping padding."""
import json,time
import numpy as np
import h5py
import torch
from .chart_transfer_v3 import build_maps,apply_maps,fuse
from .physics_evaluate import polygons
from .physics_losses import regional_maps

def evaluate(model,loader,device,output,fusion='quality',smoke=False):
    model.eval();folder=output/'predictions';folder.mkdir(exist_ok=True);rows=[];sse=np.zeros(5);count=truth=0;begin=time.perf_counter();poses=[]
    with torch.no_grad():
        for index,batch in enumerate(loader):
            b={k:v.to(device) if torch.is_tensor(v) else v for k,v in batch.items()}
            out=model(*[b[k] for k in ['history','mach','pose_history','dt','last_xy','templates']],10,'taoca',b['last_base'],b['last_valid'])
            near,shock=regional_maps(batch['target'],batch['target_valid'])
            for i in range(len(b['history'])):
                start=int(batch['start'][i]);case=batch['case'][i];fields=[];supports=[];poses.append((out['pose'][i]-b['target_pose'][i]).cpu().numpy())
                for t in range(10):
                    native=[out['fields'][i,t,k,:,:h].cpu().numpy() for k,h in enumerate(model.heights)]
                    xy=[out['xy'][i,t,k,:h].cpu().numpy() for k,h in enumerate(model.heights)]
                    masks=[out['geometry'][i,t,k,0,:h].cpu().numpy()>.5 for k,h in enumerate(model.heights)]
                    obs=polygons(batch['templates'][i].numpy(),out['pose'][i,t].cpu().numpy())
                    maps=build_maps(xy,masks,obs);charts,q=apply_maps(native,maps);rec,support=fuse(charts,q,fusion)
                    target=batch['target'][i,t].numpy();valid=batch['target_valid'][i,t].numpy()>.5;common=valid&support&np.isfinite(rec).all(0)
                    error=((rec[:,common]-target[:,common])**2).sum(1);sse+=error;count+=int(common.sum());truth+=int(valid.sum());fields.append(rec);supports.append(support)
                    row=dict(case=case,start=start,lead=t+1,n=int(common.sum()),reference_count=int(valid.sum()),sse=error.tolist(),donor_audit=[m['stats'] for m in maps])
                    for key,mask in [('wall',near),('shock',shock)]:
                        use=common&(mask[i,t,0].numpy()>.5);row[key]=dict(n=int(use.sum()),sse=((rec[:,use]-target[:,use])**2).sum(1).tolist())
                    rows.append(row)
                path=folder/f'{case}_s{start:03d}.h5';tmp=path.with_suffix('.partial.h5')
                with h5py.File(tmp,'w') as f:
                    for key,value in dict(predicted=np.array(fields),support=np.array(supports),reference=batch['target'][i].numpy(),valid=batch['target_valid'][i].numpy(),pose=out['pose'][i].cpu().numpy(),target_pose=batch['target_pose'][i].numpy(),persistence=batch['last_base'][i].numpy()).items():f.create_dataset(key,data=value,compression='lzf')
                    f.attrs['case']=case;f.attrs['start']=start
                tmp.replace(path)
            print(json.dumps(dict(evaluation_windows=index+1)),flush=True)
    pose=np.array(poses);result=dict(smoke_only=smoke,forecast_instances=len(rows),coverage=count/max(truth,1),nrmse=np.sqrt(sse/max(count,1)).tolist(),pose_rmse=np.sqrt((pose**2).mean((0,1))).tolist(),seconds=time.perf_counter()-begin,heights=model.heights,fusion=fusion,scope='All overlapping instances on native legal support; final Cartesian comparison uses intersection.')
    (output/'evaluation.json').write_text(json.dumps(result,indent=2));(output/'frame_metrics.json').write_text(json.dumps(rows));print(json.dumps(result),flush=True)
    return result
