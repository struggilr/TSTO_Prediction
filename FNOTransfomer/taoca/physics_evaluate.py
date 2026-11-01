"""All-window physical evaluation with explicit valid-donor coverage and no hole imputation."""
import json,time
from pathlib import Path
import numpy as np
import h5py
import torch
from scipy.ndimage import distance_transform_edt
from .inverse_validated import validated_inverse
from .physics_losses import regional_maps

def polygons(templates,pose):
    result=[]
    for i,p in enumerate(templates):
        q=pose[3*i:3*i+3];c,s=np.cos(q[2]),np.sin(q[2]);result.append(p@np.array([[c,s],[-s,c]])+q[:2])
    return result

def confidence(mask,xy):
    result=[]
    for k,(m,p) in enumerate(zip(mask,xy)):
        d=distance_transform_edt(np.tile(m,(1,3)))[:,m.shape[1]:2*m.shape[1]] if k<3 else distance_transform_edt(m)
        di=np.gradient(p,axis=0);dj=np.gradient(p,axis=1);edge2=np.maximum((di*di).sum(-1),(dj*dj).sum(-1))
        result.append((d/(d+2))**2/np.maximum(edge2,1e-5))
    return np.array(result)

def evaluate(model,loader,device,route,output,smoke=False):
    model.eval();folder=output/'predictions';folder.mkdir(exist_ok=True)
    total=np.zeros(5);persist=np.zeros(5);count=truth=0;pose=[];rows=[];begin=time.perf_counter()
    with torch.no_grad():
        for window,batch in enumerate(loader):
            b={k:v.to(device) if torch.is_tensor(v) else v for k,v in batch.items()}
            out=model(b['history'],b['mach'],b['pose_history'],b['dt'],b['last_xy'],b['templates'],10,route,b['last_base'],b['last_valid'])
            near,shock=regional_maps(batch['target'],batch['target_valid'])
            for i in range(len(b['history'])):
                start=int(batch['start'][i]);case=batch['case'][i];rec=[];supports=[];pose.append((out['pose'][i]-b['target_pose'][i]).cpu().numpy())
                for t in range(10):
                    f=out['fields'][i,t].cpu().numpy();xy=out['xy'][i,t].cpu().numpy();mask=out['geometry'][i,t,:,0].cpu().numpy()>.5
                    if route=='taoca':
                        obs=polygons(batch['templates'][i].numpy(),out['pose'][i,t].cpu().numpy())
                        field,support,stats=validated_inverse(f,xy,mask,obs,confidence(mask,xy))
                    else:field=f[0];support=mask[0];stats=[]
                    target=batch['target'][i,t].numpy();valid=batch['target_valid'][i,t].numpy()>.5;common=valid&support&np.isfinite(field).all(0)
                    error=((field[:,common]-target[:,common])**2).sum(1);baseline=((batch['last_base'][i].numpy()[:,common]-target[:,common])**2).sum(1)
                    total+=error;persist+=baseline;count+=int(common.sum());truth+=int(valid.sum());rec.append(field);supports.append(support)
                    row=dict(case=case,start=start,lead=t+1,n=int(common.sum()),reference_count=int(valid.sum()),sse=error.tolist(),persistence_sse=baseline.tolist(),donor_audit=stats)
                    for name,mask2 in [('wall',near),('shock',shock)]:
                        region=common&(mask2[i,t,0].numpy()>.5);row[name]=dict(n=int(region.sum()),sse=((field[:,region]-target[:,region])**2).sum(1).tolist())
                    rows.append(row)
                path=folder/f'{case}_s{start:03d}.h5';tmp=path.with_suffix('.partial.h5')
                with h5py.File(tmp,'w') as h:
                    for key,val in dict(predicted=np.array(rec),support=np.array(supports),reference=batch['target'][i].numpy(),valid=batch['target_valid'][i].numpy(),pose=out['pose'][i].cpu().numpy(),target_pose=batch['target_pose'][i].numpy(),persistence=batch['last_base'][i].numpy()).items():h.create_dataset(key,data=val,compression='lzf')
                    h.attrs['case']=case;h.attrs['start']=start
                tmp.replace(path)
            print(json.dumps(dict(evaluation_windows=window+1,route=route)),flush=True)
    pose=np.array(pose)
    result=dict(smoke_only=smoke,forecast_instances=len(rows),coverage=count/max(truth,1),nrmse=np.sqrt(total/max(count,1)).tolist(),persistence_nrmse_on_same_support=np.sqrt(persist/max(count,1)).tolist(),pose_rmse=np.sqrt((pose**2).mean((0,1))).tolist(),end_to_end_evaluation_seconds=time.perf_counter()-begin,scope='All overlapping forecast instances; native valid-donor support. Cross-route comparison must use support intersection. Missing predictions remain NaN, no imputation.')
    (output/'evaluation.json').write_text(json.dumps(result,indent=2));(output/'frame_metrics.json').write_text(json.dumps(rows));print(json.dumps(result),flush=True)
    return result
