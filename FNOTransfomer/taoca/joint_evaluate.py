"""Evaluate predicted geometry on the common fixed physical grid."""
import numpy as np
import torch
from scipy.spatial import cKDTree
from .geometry import structured_triangles


def reconstruct(fields,xy):
    from .inverse_complete import complete_inverse
    result,support,stats=complete_inverse(fields,xy,periodic=True)
    return result,support,sum(c['nonpositive'] for c in stats),sum(c['triangles'] for c in stats)


def evaluate(model,loader,device,horizon,route,output):
    pose_errors=[];sse=np.zeros(5);full_sse=np.zeros(5);count=0;truth=0;invalid=0;cells=0;seen=set();saved=False;per_lead={}
    model.eval()
    with torch.no_grad():
        for batch in loader:
            b={k:v.to(device) if torch.is_tensor(v) else v for k,v in batch.items()}
            out=model(b['history'],b['mach'],b['pose_history'],b['dt'],b['last_xy'],b['templates'],horizon,route)
            pose_errors.append((out['pose']-b['target_pose']).cpu().numpy())
            for i in range(len(b['history'])):
                for t in range(horizon):
                    key=(batch['case'][i],int(batch['start'][i])+t)
                    if key in seen:continue
                    seen.add(key)
                    values=out['fields'][i,t].cpu().numpy();xy=out['xy'][i,t].cpu().numpy()
                    if route=='cartesian':rec=values[0];support=np.ones((280,440),bool);bad=0;number=1
                    else:rec,support,bad,number=reconstruct(values,xy)
                    target=batch['target'][i,t].numpy();valid=batch['target_valid'][i,t].numpy().astype(bool);common=valid&support
                    err=((rec[:,common]-target[:,common])**2).sum(1);sse+=err;full_sse+=((rec[:,valid]-target[:,valid])**2).sum(1);count+=int(common.sum());truth+=int(valid.sum());invalid+=bad;cells+=number
                    lead=per_lead.setdefault(t+1,[np.zeros(5),0]);lead[0]+=err;lead[1]+=int(common.sum())
                    if not saved:
                        np.savez_compressed(output/'joint_snapshot.npz',predicted=rec,target=target,valid=common,xy=xy,pose=out['pose'][i].cpu().numpy(),target_pose=batch['target_pose'][i].numpy());saved=True
    e=np.concatenate(pose_errors)
    return dict(pose_rmse=np.sqrt((e**2).mean((0,1))).tolist(),pose_units=['m','m','rad']*2,flow_nrmse_training_std=np.sqrt(sse/max(count,1)).tolist(),full_domain_nrmse_missing_as_training_mean=np.sqrt(full_sse/max(truth,1)).tolist(),coverage=count/max(truth,1),nonpositive_triangle_fraction=invalid/max(cells,1),unique_flow_frames=len(seen),per_lead_nrmse={str(k):np.sqrt(v[0]/max(v[1],1)).tolist() for k,v in per_lead.items()},limitations='Geometry-dependent native support; full-domain metric imputes missing predictions with training mean. Candidate containment search may miss cells; holes reported. Pose RMSE counts overlapping forecast windows; flow frames use first forecast occurrence.')
