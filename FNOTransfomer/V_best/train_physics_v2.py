"""Controlled spatial/physics V2 training; old runs and models remain reproducible."""
import os
os.environ.setdefault('MKL_THREADING_LAYER','GNU')
import argparse,json,random,time,hashlib
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
from taoca.physics_data import PhysicsWindows
from taoca.spatial_physics import SpatialPredictor
from taoca.physics_losses import PhysicsLoss
ROOT=Path(__file__).resolve().parent

def arguments():
    p=argparse.ArgumentParser();p.add_argument('--route',choices=['cartesian','taoca'],required=True);p.add_argument('--epochs',type=int,default=100);p.add_argument('--batch-size',type=int,default=1);p.add_argument('--lr',type=float,default=3e-4);p.add_argument('--seed',type=int,default=42);p.add_argument('--smoke',action='store_true');p.add_argument('--smoke-steps',type=int,default=3);p.add_argument('--resume',action='store_true');p.add_argument('--device',default='cuda');p.add_argument('--skip-evaluation',action='store_true');return p.parse_args()
def forward(model,b,route):return model(b['history'],b['mach'],b['pose_history'],b['dt'],b['last_xy'],b['templates'],10,route,b['last_base'],b['last_valid'])
def atomic_json(path,data):
    tmp=path.with_suffix('.partial.json');tmp.write_text(json.dumps(data,indent=2),encoding='utf8');os.replace(tmp,path)
def main():
    a=arguments();torch.set_num_threads(4);random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    device=torch.device(a.device)
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA required; no CPU fallback')
    output=ROOT/'outputs/physics_v2'/(a.route+('_smoke' if a.smoke else ''));output.mkdir(parents=True,exist_ok=True)
    thermo=json.loads((ROOT/'outputs/physics_v2/thermodynamics.json').read_text());meta=json.loads((ROOT/'datasets/joint_nearest/manifest.json').read_text())
    datasets={s:PhysicsWindows(ROOT/'datasets/joint_nearest',ROOT/'datasets/fenli_paired',a.route,s) for s in ['train','val','test']}
    loaders={s:DataLoader(Subset(d,list(range(min(1,len(d))))) if a.smoke else d,batch_size=a.batch_size,shuffle=s=='train',num_workers=0) for s,d in datasets.items()}
    model=SpatialPredictor(ROOT/'outputs/shared_trajectory/model.json').to(device);loss_fn=PhysicsLoss(meta,thermo)
    optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-4);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,a.epochs,eta_min=a.lr*.05)
    metadata=dict(arguments=vars(a),parameters=sum(p.numel() for p in model.parameters()),windows={s:len(d) for s,d in datasets.items()},hardware=torch.cuda.get_device_name() if device.type=='cuda' else 'cpu',thermodynamics=thermo,protocol='Same chronological targets, shared frozen history-only trajectory, common Eulerian persistence prior; spatial FNO/Transformer; mask-aware inverse; physical-bin quadrature; wall/shock weights 5/4; gradient matching, EOS, positivity, smooth-region ALE continuity.',loss_weights=dict(wall_extra=4,shock_extra=3,gradient=.03,eos=.01,positivity=.01,continuity=.001),source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'taoca/spatial_physics.py',ROOT/'taoca/physics_losses.py',ROOT/'taoca/physics_data.py',ROOT/'taoca/inverse_validated.py',Path(__file__)]})
    metadata['loss_weights']['overlap_consistency']=.01
    metadata['source_sha256'].update({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'taoca/domain_charts.py',ROOT/'taoca/physics_evaluate.py',ROOT/'taoca/staged.py',ROOT/'taoca/joint.py',ROOT/'outputs/shared_trajectory/model.json',ROOT/'datasets/joint_nearest/manifest.json']})
    metadata['geometry']='V2 1.5m near-body offsets with orientation gate; fixed domain-aware annular background and masked solids; original gap transport; no future truth geometry.'
    if (output/'run.json').exists() and a.resume:
        previous=json.loads((output/'run.json').read_text())
        if previous['source_sha256']!=metadata['source_sha256']:raise RuntimeError('Cannot resume with changed source or manifest; use a separate versioned output')
    atomic_json(output/'run.json',metadata);history=[];best=float('inf');start=1
    if a.resume and (output/'last.pt').exists():
        state=torch.load(output/'last.pt',map_location=device,weights_only=False);model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler']);history=state['history'];best=state['best'];start=state['epoch']+1
    atomic_json(output/'status.json',dict(state='training',epoch=start-1,total_epochs=a.epochs,smoke=a.smoke));print(json.dumps(metadata),flush=True)
    epochs=a.smoke_steps if a.smoke else a.epochs
    try:
        for epoch in range(start,epochs+1):
            row=dict(epoch=epoch,lr=optimizer.param_groups[0]['lr']);begin=time.perf_counter()
            for split in ['train','val']:
                model.train(split=='train');totals={};count=0
                with torch.set_grad_enabled(split=='train'):
                    for batch in loaders[split]:
                        b={k:v.to(device) if torch.is_tensor(v) else v for k,v in batch.items()}
                        out=forward(model,b,a.route);loss,parts=loss_fn(out,b,epoch if split=='train' else 10)
                        if not torch.isfinite(loss):raise ValueError('Non-finite V2 loss')
                        if split=='train':
                            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                            if a.smoke and epoch==3:
                                grad=model.spectral[0].positive.grad
                                nonzero=float(grad[:,:,1:,1:].abs().sum()) if grad is not None else 0.
                                if nonzero<=0:raise RuntimeError('Nonzero Fourier modes do not receive gradients')
                                row['nonzero_fourier_gradient_l1']=nonzero
                            optimizer.step()
                        bs=len(b['history']);count+=bs
                        for k,v in dict(loss=float(loss.detach()),**parts).items():totals[k]=totals.get(k,0)+v*bs
                if device.type=='cuda':torch.cuda.synchronize()
                row[split]={k:v/count for k,v in totals.items()}
                if split=='train':row['train_seconds']=time.perf_counter()-begin
            scheduler.step();history.append(row)
            if row['val']['loss']<best:best=row['val']['loss'];torch.save(model.state_dict(),output/'best.pt')
            torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),history=history,best=best,epoch=epoch),output/'last.pt')
            atomic_json(output/'history.json',history);atomic_json(output/'status.json',dict(state='training',epoch=epoch,total_epochs=epochs,best_validation_loss=best,elapsed_epoch_seconds=time.perf_counter()-begin,peak_gpu_bytes=torch.cuda.max_memory_allocated() if device.type=='cuda' else 0));print(json.dumps(row),flush=True)
        model.load_state_dict(torch.load(output/'best.pt',map_location=device,weights_only=True))
        if not a.skip_evaluation:
            atomic_json(output/'status.json',dict(state='evaluating',epoch=epochs,total_epochs=epochs))
            from taoca.physics_evaluate import evaluate
            evaluate(model,loaders['test'],device,a.route,output,smoke=a.smoke)
        atomic_json(output/'status.json',dict(state='complete',epochs=epochs,smoke=a.smoke,evaluated=not a.skip_evaluation))
    except Exception as e:
        atomic_json(output/'status.json',dict(state='failed',error=repr(e),completed_epochs=len(history)));raise
if __name__=='__main__':main()
