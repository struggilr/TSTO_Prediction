"""Joint flow/trajectory training; no future tensors in model forward."""
import argparse,json,time,random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
from taoca.joint import JointPredictor,JointWindows,joint_loss
ROOT=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser();p.add_argument('--trajectory-model',type=Path);p.add_argument('--route',required=True,choices=['cartesian','taoca']);p.add_argument('--data',type=Path,default=ROOT/'datasets/joint_nearest');p.add_argument('--paired',type=Path,default=ROOT/'datasets/fenli_paired');p.add_argument('--epochs',type=int,default=100);p.add_argument('--batch-size',type=int,default=2);p.add_argument('--workers',type=int,default=0);p.add_argument('--history',type=int,default=20);p.add_argument('--horizon',type=int,default=10);p.add_argument('--device',choices=['cuda','cpu'],default='cuda');p.add_argument('--smoke',action='store_true');a=p.parse_args()
    torch.set_num_threads(4);random.seed(42);np.random.seed(42);torch.manual_seed(42);device=torch.device(a.device)
    if device.type=='cuda':
        if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable: run scripts/check_cuda.py --devices 1 in this exact Python environment; CPU fallback is disabled.')
        x=torch.randn(16,16,device=device);(x@x).sum().item();torch.fft.rfft2(x).abs().sum().item()
    print(json.dumps(dict(python=__import__('sys').executable,torch=torch.__version__,cuda_build=torch.version.cuda,device=str(device),gpu=torch.cuda.get_device_name(0) if device.type=='cuda' else None)),flush=True)
    if a.trajectory_model:
        from taoca.staged import StagedPredictor
        model=StagedPredictor(a.trajectory_model).to(device)
    else:model=JointPredictor().to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    output=ROOT/'outputs'/(('staged_' if a.trajectory_model else 'joint_')+a.route+('_smoke' if a.smoke else ''));output.mkdir(parents=True,exist_ok=True)
    datasets={s:JointWindows(a.data,a.paired,a.route,s,a.history,a.horizon) for s in ['train','val','test']}
    if any(not len(d) for d in datasets.values()):raise ValueError('Empty time split')
    loaders={s:DataLoader(Subset(d,list(range(min(a.batch_size,len(d))))) if a.smoke else d,batch_size=a.batch_size,shuffle=s=='train',num_workers=a.workers) for s,d in datasets.items()}
    metadata=dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},parameters=sum(x.numel() for x in model.parameters()),windows={s:len(d) for s,d in datasets.items()},protocol='history-only joint prediction; transported predicted charts; nearest-grid common source; no future masks/coordinates/flow inputs')
    (output/'run.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8');best=float('inf');history=[]
    for epoch in range(1,(1 if a.smoke else a.epochs)+1):
        row=dict(epoch=epoch);begin=time.perf_counter()
        for split in ['train','val']:
            model.train(split=='train');total=0;count=0
            with torch.set_grad_enabled(split=='train'):
                for batch in loaders[split]:
                    b={k:v.to(device) if torch.is_tensor(v) else v for k,v in batch.items()}
                    pred=model(b['history'],b['mach'],b['pose_history'],b['dt'],b['last_xy'],b['templates'],a.horizon,a.route)
                    loss,parts=joint_loss(pred,b,a.route)
                    if a.trajectory_model:loss=loss-pred['fields'].new_tensor(parts['pose'])
                    if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
                    if split=='train':opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
                    total+=loss.item()*len(b['history']);count+=len(b['history'])
            if device.type=='cuda':torch.cuda.synchronize()
            row[split+'_loss']=total/count
            if split=='train':row['train_seconds']=time.perf_counter()-begin
        history.append(row);print(json.dumps(row),flush=True)
        if row['val_loss']<best:best=row['val_loss'];torch.save(model.state_dict(),output/'best.pt')
        (output/'history.json').write_text(json.dumps(history,indent=2))
    model.load_state_dict(torch.load(output/'best.pt',map_location=device,weights_only=True))
    from taoca.joint_evaluate import evaluate
    import time as clock
    begin=clock.perf_counter();report=evaluate(model,loaders['test'],device,a.horizon,a.route,output);report['test_end_to_end_seconds']=clock.perf_counter()-begin;report['smoke_only']=a.smoke
    (output/'evaluation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':main()
