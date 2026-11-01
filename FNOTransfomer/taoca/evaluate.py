import time
import h5py
import numpy as np
import torch
from .data import NAMES


def inverse(frame,route,fields):
    n=frame['source/x'].shape[0];result=np.zeros((5,n));den=np.zeros(n)
    for values,name in zip(fields,NAMES[route]):
        c=frame[route][name];ids=c['inv_ids'][:];bary=c['inv_bary'][:];weight=c['inv_weight'][:]
        interpolated=np.einsum('cnk,nk->cn',values.reshape(5,-1)[:,np.maximum(ids,0)],bary)
        result+=interpolated*weight;den+=weight
    valid=den>0;result[:,valid]/=den[valid];result[:,~valid]=np.nan
    return result,valid


def coverage(frame,route):
    return np.sum([frame[route][name]['inv_weight'][:] for name in NAMES[route]],axis=0)>0


def evaluate(model,dataset,device,limit=None,output=None):
    model.eval();sse=np.zeros(5);floor=np.zeros(5);chart=np.zeros(5);persistence=np.zeros(5);count=0;nodes=0;covered=0;native=0;seen=set();frames=0;seconds=0.;windows=0
    mean=dataset.mean[:,0,0];std=dataset.std[:,0,0]
    selection=[i for i,(case,s) in enumerate(dataset.samples) if (s-80)%10==0 or s==len(dataset.meta['cases'][case])-dataset.horizon]
    for index in selection[:limit]:
        item=dataset[index];case=item['case'];start=item['start']
        history=item['history'][None].to(device);geometry=item['geometry'][None].to(device)
        if device.type=='cuda':torch.cuda.synchronize(device)
        begin=time.perf_counter()
        with torch.no_grad():pred=model(history,item['mach'][None].to(device),geometry)
        if device.type=='cuda':torch.cuda.synchronize(device)
        seconds+=time.perf_counter()-begin;windows+=1;pred=pred[0].cpu().numpy()
        for t in range(dataset.horizon):
            key=(case,start+t)
            if key in seen:continue
            seen.add(key);frames+=1
            path=dataset.root/dataset.meta['cases'][case][start+t]['file']
            with h5py.File(path,'r') as f:
                reference=f['source/fields'][:];raw=pred[t]*std[None,:,None,None]+mean[None,:,None,None]
                reconstructed,valid=inverse(f,dataset.route,raw)
                true_chart=np.stack([f[dataset.route][name]['fields'][:] for name in NAMES[dataset.route]])
                rt,_=inverse(f,dataset.route,true_chart)
                last=item['history'][-1,:,:5].numpy()*std[None,:,None,None]+mean[None,:,None,None]
                held,_=inverse(f,dataset.route,last)
                common=coverage(f,'cartesian')&coverage(f,'taoca');covered+=int(common.sum());native+=int(valid.sum());nodes+=len(valid)
                if not common.any():raise ValueError('Empty common physical domain')
                sse+=np.sum((reconstructed[:,common]-reference[:,common])**2,axis=1)
                floor+=np.sum((rt[:,common]-reference[:,common])**2,axis=1)
                chart+=np.sum((reconstructed[:,common]-rt[:,common])**2,axis=1)
                persistence+=np.sum((held[:,common]-reference[:,common])**2,axis=1)
                count+=int(common.sum())
                if output is not None and t==0:
                    from pathlib import Path
                    folder=Path(output)/'snapshots';folder.mkdir(parents=True,exist_ok=True)
                    np.savez_compressed(folder/f'{case}_{start+t:04d}.npz',x=f['source/x'][:],y=f['source/y'][:],triangles=f['source/triangles'][:],reference=reference,predicted=reconstructed,roundtrip=rt,native_valid=valid,common_valid=common,chart_prediction=raw,fields=np.array(dataset.meta['fields']),time=float(f.attrs['time']))
    if not count:raise ValueError('No evaluation samples')
    return dict(fields=dataset.meta['fields'],frames=frames,common_source_nodes=count,common_coverage=covered/nodes,native_coverage=native/nodes,physical_rmse=np.sqrt(sse/count).tolist(),physical_normalized_rmse=(np.sqrt(sse/count)/std).tolist(),roundtrip_floor_rmse=np.sqrt(floor/count).tolist(),forecast_vs_mapped_truth_rmse=np.sqrt(chart/count).tolist(),logical_last_frame_persistence_rmse=np.sqrt(persistence/count).tolist(),inference_seconds=seconds,inference_windows=windows,seconds_per_10frame_window=seconds/windows,normalizer='shared source-node training std',metric_domain='intersection of both physical inverse supports; native coverage also reported',decomposition='RMSE terms are diagnostics, not an additive identity',persistence_scope='last logical field held fixed then mapped using target geometry; history-invalid nodes retain normalized zero')
