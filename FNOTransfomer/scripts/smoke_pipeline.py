"""One real-data optimizer step + inverse per route; NOT a trained accuracy result."""
import sys,json,time
from pathlib import Path
import h5py,numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from taoca.model import Predictor
from taoca.data import NAMES
from taoca.evaluate import inverse,coverage


def main():
    torch.set_num_threads(2);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');rows=[]
    paths=[ROOT/'datasets/fenli_paired/Ma5.0'/f'{i:04d}.h5' for i in range(30)]
    if not all(p.exists() for p in paths):raise ValueError('Need first 30 contiguous paired frames')
    with h5py.File(paths[0]) as f:
        ref=f['source/fields'][:];mean=ref.mean(axis=1);std=ref.std(axis=1).clip(1e-8);lo=np.array([f['source/x'][:].min(),f['source/y'][:].min()]);hi=np.array([f['source/x'][:].max(),f['source/y'][:].max()])
    for route,names in NAMES.items():
        fields=[];geometries=[]
        for path in paths:
            with h5py.File(path) as f:
                fields.append(np.stack([np.where(f[route][n]['mask'][:][None],(f[route][n]['fields'][:]-mean[:,None,None])/std[:,None,None],0) for n in names]).astype('f4'))
                geometries.append(np.stack([np.concatenate([f[route][n]['mask'][:][None],2*(f[route][n]['xy'][:]-lo[:,None,None])/(hi-lo)[:,None,None]-1]) for n in names]).astype('f4'))
        fields=np.stack(fields);geometries=np.stack(geometries)
        x=torch.from_numpy(np.concatenate([fields[:20],geometries[:20]],axis=2))[None].to(device);geo=torch.from_numpy(geometries[20:])[None].to(device);target=torch.from_numpy(fields[20:])[None].to(device)
        model=Predictor().to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)
        if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
        begin=time.perf_counter();pred=model(x,torch.tensor([5.],device=device),geo)
        mask=geo[:,:,:,:1];loss=((pred-target).square()*mask).sum()/(mask.sum()*5);loss.backward()
        finite=bool(torch.isfinite(loss).item()) and all(torch.isfinite(p.grad).all().item() for p in model.parameters() if p.grad is not None)
        if not finite:raise ValueError('Nonfinite loss or gradient')
        optimizer.step()
        if device.type=='cuda':torch.cuda.synchronize()
        elapsed=time.perf_counter()-begin
        prediction=pred[0,0].detach().cpu().numpy()*std[None,:,None,None]+mean[None,:,None,None]
        with h5py.File(paths[20]) as f:
            back,valid=inverse(f,route,prediction);common=coverage(f,'cartesian')&coverage(f,'taoca')
            if not np.isfinite(back[:,common]).all():raise ValueError('Nonfinite common inverse')
        row=dict(route=route,optimizer_step=True,loss=float(loss.item()),finite_gradients=finite,shape=list(pred.shape),physical_inverse_finite=True,common_coverage=float(common.mean()),parameters=sum(p.numel() for p in model.parameters()),seconds=elapsed,peak_gpu_bytes=torch.cuda.max_memory_allocated() if device.type=='cuda' else 0)
        rows.append(row);print(json.dumps(row),flush=True)
        del model,optimizer,pred,x,geo,target,loss;torch.cuda.empty_cache()
    report=dict(smoke_only=True,scope='First 30 real paired frames, 20->10; first-frame normalization for this isolated smoke only; not formal model training',gpu=torch.cuda.get_device_name(0) if device.type=='cuda' else None,results=rows)
    out=ROOT/'outputs/smoke_pipeline.json';out.parent.mkdir(exist_ok=True);out.write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':main()
