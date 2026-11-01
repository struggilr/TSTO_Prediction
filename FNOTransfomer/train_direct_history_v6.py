"""Direct original-CFD history ablation; frozen V3 architecture and Cartesian baseline."""
import os
os.environ.setdefault('MKL_THREADING_LAYER','GNU')
import argparse,json,time,random,hashlib
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
from taoca.direct_history_v6 import DirectHistoryWindows
from taoca.allocated_model_v3 import AllocatedPredictor
from taoca.physics_losses import PhysicsLoss
from train_physics_v2 import atomic_json
ROOT=Path(__file__).resolve().parent
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');parser.add_argument('--resume',action='store_true');a=parser.parse_args()
    root=ROOT/'outputs/direct_history_v6';selection=json.loads((ROOT/'outputs/atlas_v3/allocation_selection.json').read_text())
    if not selection['training_authorized_by_gate']:raise RuntimeError('Representation gate failed; training is not launched')
    heights=selection['selected']['heights'];fusion=selection['fusion'];torch.set_num_threads(4);random.seed(42);np.random.seed(42);torch.manual_seed(42);device='cuda'
    output=root/('taoca_smoke' if a.smoke else 'taoca');output.mkdir(exist_ok=True)
    meta=json.loads((ROOT/'datasets/joint_nearest/manifest.json').read_text());thermo=json.loads((ROOT/'outputs/physics_v2/thermodynamics.json').read_text());baseline=json.loads((ROOT/'outputs/physics_v2/cartesian/run.json').read_text())
    shared=['taoca/spatial_physics.py','taoca/physics_losses.py','taoca/staged.py','taoca/joint.py','taoca/physics_data.py','taoca/domain_charts.py','datasets/joint_nearest/manifest.json','outputs/shared_trajectory/model.json']
    for name in shared:
        p=ROOT/name;key=str(p.relative_to(ROOT))
        if hashlib.sha256(p.read_bytes()).hexdigest()!=baseline['source_sha256'][key]:raise RuntimeError('Frozen baseline shared source changed: '+name)
    data={s:DirectHistoryWindows(ROOT/'datasets/joint_nearest',ROOT/'datasets/fenli_paired','taoca',s,heights=heights) for s in ['train','val','test']}
    loaders={s:DataLoader(Subset(d,[0]) if a.smoke else d,batch_size=1,shuffle=s=='train',num_workers=0) for s,d in data.items()}
    model=AllocatedPredictor(ROOT/'outputs/shared_trajectory/model.json',heights).cuda();opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,100,eta_min=1.5e-5);lossfn=PhysicsLoss(meta,thermo)
    files=shared+['taoca/allocated_data_v3.py','taoca/allocated_model_v3.py','taoca/allocation_v3.py','taoca/allocated_evaluate_v3.py','taoca/chart_transfer_v3.py','train_direct_history_v6.py','taoca/direct_history_v6.py','datasets/taoca_direct_history_v6/manifest.json','outputs/atlas_v3/allocation_selection.json','outputs/atlas_v3/selection_contract.json']
    metadata=dict(heights=heights,sampled_points=sum(heights)*440,parameters=sum(p.numel() for p in model.parameters()),arguments=dict(route='taoca',seed=42,epochs=100,batch_size=1,lr=3e-4,smoke=a.smoke),windows={s:len(d) for s,d in data.items()},fusion=fusion,cartesian_checkpoint_sha256=hashlib.sha256((ROOT/'outputs/physics_v2/cartesian/best.pt').read_bytes()).hexdigest(),source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in files},protocol='V2 shared parameters/loss/optimizer/chronological data; V6 changes historical chart-field sampling only to original CFD P1; target labels, physical prior, architecture, geometry and losses unchanged; no encoded padding; cross-chart context interpolates feature maps to the recipient logical size; representation and fusion choices use validation only.')
    assert metadata['parameters']==baseline['parameters']==340349
    if a.resume and (output/'run.json').exists() and json.loads((output/'run.json').read_text())['source_sha256']!=metadata['source_sha256']:raise RuntimeError('Source changed; cannot resume')
    atomic_json(output/'run.json',metadata);history=[];best=float('inf');start=1
    if a.resume and (output/'last.pt').exists():
        state=torch.load(output/'last.pt',map_location=device,weights_only=False);model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler']);history=state['history'];best=state['best'];start=state['epoch']+1
    print(json.dumps(metadata),flush=True)
    try:
        for epoch in range(start,(3 if a.smoke else 100)+1):
            row=dict(epoch=epoch,lr=opt.param_groups[0]['lr']);begin=time.perf_counter()
            for split in ['train','val']:
                model.train(split=='train');total={};count=0
                with torch.set_grad_enabled(split=='train'):
                    for batch in loaders[split]:
                        b={k:v.cuda() if torch.is_tensor(v) else v for k,v in batch.items()};out=model(*[b[k] for k in ['history','mach','pose_history','dt','last_xy','templates']],10,'taoca',b['last_base'],b['last_valid']);loss,parts=lossfn(out,b,epoch if split=='train' else 10)
                        if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
                        if split=='train':opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
                        count+=1
                        for k,v in dict(loss=float(loss.detach()),**parts).items():total[k]=total.get(k,0)+v
                torch.cuda.synchronize();row[split]={k:v/count for k,v in total.items()}
                if split=='train':row['train_seconds']=time.perf_counter()-begin
            scheduler.step();history.append(row)
            if row['val']['loss']<best:best=row['val']['loss'];torch.save(model.state_dict(),output/'best.pt')
            torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),history=history,best=best,epoch=epoch),output/'last.pt')
            atomic_json(output/'history.json',history);atomic_json(output/'status.json',dict(state='training',epoch=epoch,total_epochs=3 if a.smoke else 100,best_validation=best,epoch_seconds=time.perf_counter()-begin,peak_gpu_bytes=torch.cuda.max_memory_allocated()));print(json.dumps(row),flush=True)
        model.load_state_dict(torch.load(output/'best.pt',map_location=device,weights_only=True));atomic_json(output/'status.json',dict(state='evaluating',epoch=len(history)))
        from taoca.allocated_evaluate_v3 import evaluate
        evaluate(model,loaders['test'],device,output,fusion,a.smoke);atomic_json(output/'status.json',dict(state='complete',epochs=len(history),smoke=a.smoke))
    except Exception as e:atomic_json(output/'status.json',dict(state='failed',error=repr(e),completed_epochs=len(history)));raise
if __name__=='__main__':main()
