"""Paired, geometry-conditioned FNO + Transformer experiment."""
import argparse, hashlib, json, random, time, platform
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from taoca.data import Windows
from taoca.model import Predictor

ROOT=Path(__file__).resolve().parent


def sync(device):
    if device.type=='cuda':torch.cuda.synchronize(device)


def loss_fn(pred,target,geometry):
    mask=geometry[:,:,:,:1]
    return ((pred-target).square()*mask).sum()/(5*mask.sum()).clamp_min(1)


def move(batch,device):return {k:v.to(device) if torch.is_tensor(v) else v for k,v in batch.items()}


def main():
    p=argparse.ArgumentParser();p.add_argument('--route',choices=['cartesian','taoca'],required=True);p.add_argument('--data',type=Path,default=ROOT/'datasets/fenli_paired');p.add_argument('--output',type=Path);p.add_argument('--epochs',type=int,default=100);p.add_argument('--batch-size',type=int,default=1);p.add_argument('--workers',type=int,default=2);p.add_argument('--seed',type=int,default=42);p.add_argument('--lr',type=float,default=1e-3);p.add_argument('--width',type=int,default=32);p.add_argument('--modes',type=int,default=12);p.add_argument('--latent',type=int,default=128);p.add_argument('--layers',type=int,default=3);p.add_argument('--temporal-layers',type=int,default=2);p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu');p.add_argument('--smoke',action='store_true');p.add_argument('--evaluate-only',action='store_true')
    a=p.parse_args();a.output=a.output or ROOT/'outputs'/a.route;a.output.mkdir(parents=True,exist_ok=True)
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(a.seed)
    torch.set_num_threads(4);device=torch.device(a.device)
    config={k:getattr(a,k) for k in ['width','modes','latent','layers','temporal_layers']};model=Predictor(**config).to(device)
    train=Windows(a.data,a.route,'train');val=Windows(a.data,a.route,'val')
    if not len(train) or not len(val):raise ValueError('Empty split')
    val_ids=[i for i,(_,s) in enumerate(val.samples) if (s-60)%10==0]
    if a.batch_size<1 or a.workers<0:raise ValueError('Invalid batch size or worker count')
    smoke_ids=list(range(min(a.batch_size,len(train))))
    train_loader=DataLoader(Subset(train,smoke_ids) if a.smoke else train,batch_size=a.batch_size,shuffle=True,num_workers=a.workers,pin_memory=device.type=='cuda')
    val_loader=DataLoader(Subset(val,val_ids[:a.batch_size] if a.smoke else val_ids),batch_size=a.batch_size,num_workers=a.workers,pin_memory=device.type=='cuda')
    meta=dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},model_config=config,parameters=sum(p.numel() for p in model.parameters()),device=str(device),gpu=torch.cuda.get_device_name(device) if device.type=='cuda' else None,torch=torch.__version__,python=platform.python_version(),dataset_manifest_sha256=hashlib.sha256((a.data/'manifest.json').read_bytes()).hexdigest(),train_windows=len(train),val_windows=len(val_ids),protocol='known future geometry; no target flow input; common training normalization; FNOEncoder + temporal Transformer reused from V2.0; portable shared decoder; no chart-coordinate PDE loss')
    (a.output/'run.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    if a.evaluate_only:
        model.load_state_dict(torch.load(a.output/'best.pt',map_location=device,weights_only=False)['model'])
    else:
        optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-4)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,max(1,a.epochs));best=float('inf');history=[];stale=0
        for epoch in range(1,(1 if a.smoke else a.epochs)+1):
            model.train();sync(device);start=time.perf_counter();total=0.;steps=0
            for batch in train_loader:
                batch=move(batch,device);optimizer.zero_grad(set_to_none=True)
                pred=model(batch['history'],batch['mach'],batch['geometry']);loss=loss_fn(pred,batch['target'],batch['geometry'])
                if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();total+=loss.item();steps+=1
            sync(device);seconds=time.perf_counter()-start;model.eval();val_loss=0.;n=0
            with torch.no_grad():
                for batch in val_loader:
                    batch=move(batch,device);pred=model(batch['history'],batch['mach'],batch['geometry']);val_loss+=loss_fn(pred,batch['target'],batch['geometry']).item();n+=1
            row=dict(epoch=epoch,train_mse=total/steps,val_mse=val_loss/n,train_seconds=seconds,train_windows_per_second=(len(smoke_ids) if a.smoke else len(train))/seconds,peak_gpu_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else 0)
            history.append(row);print(json.dumps(row),flush=True)
            if row['val_mse']<best:
                best=row['val_mse'];stale=0;torch.save(dict(model=model.state_dict(),config=config,epoch=epoch,meta=meta),a.output/'best.pt')
            else:stale+=1
            (a.output/'history.json').write_text(json.dumps(history,indent=2),encoding='utf-8');scheduler.step()
            if stale>=20:break
        model.load_state_dict(torch.load(a.output/'best.pt',map_location=device,weights_only=False)['model'])
    from taoca.evaluate import evaluate
    report=evaluate(model,Windows(a.data,a.route,'test'),device,limit=1 if a.smoke else None,output=a.output);report['smoke_only']=a.smoke
    (a.output/'evaluation.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
