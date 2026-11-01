"""Fine-tune TAOCA through its evaluation inverse; select by validation minimax."""
import os
os.environ['MKL_THREADING_LAYER']='GNU'
import argparse,json,time,random,hashlib,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'outputs/physical_v7'
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
from taoca.allocated_data_v3 import AllocatedWindows
from taoca.allocated_model_v3 import AllocatedPredictor
from taoca.spatial_physics import SpatialPredictor
from taoca.physics_losses import PhysicsLoss,regional_maps
from taoca.physical_supervision_v7 import physical_loss,project,load_window
def atomic_json(path,data):
    tmp=path.with_suffix('.partial.json');tmp.write_text(json.dumps(data,indent=2),encoding='utf8');os.replace(tmp,path)
def forward(model,b):return model(*[b[k] for k in ['history','mach','pose_history','dt','last_xy','templates']],10,'taoca',b['last_base'],b['last_valid'])
def main():
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--resume',action='store_true');p.add_argument('--epochs',type=int,default=60);a=p.parse_args();torch.set_num_threads(4);random.seed(42);np.random.seed(42);torch.manual_seed(42)
    assert json.loads((BASE/'operator_tests.json').read_text())['verified']
    if not a.smoke:assert json.loads((BASE/'prepare_status.json').read_text())['state']=='complete'
    heights=[35,35,140,70];folder=BASE/('taoca_smoke' if a.smoke else 'taoca');folder.mkdir(exist_ok=True)
    ds={s:AllocatedWindows(ROOT/'datasets/joint_nearest',ROOT/'datasets/fenli_paired','taoca',s,heights=heights) for s in ['train','val','test']}
    loaders={s:DataLoader(Subset(d,[0]) if a.smoke else d,batch_size=1,shuffle=s=='train',num_workers=0) for s,d in ds.items()}
    meta=json.loads((ROOT/'datasets/joint_nearest/manifest.json').read_text());thermo=json.loads((ROOT/'configs/thermodynamics.json').read_text());native_loss=PhysicsLoss(meta,thermo)
    model=AllocatedPredictor(ROOT/'configs/trajectory_model.json',heights).cuda();model.load_state_dict(torch.load(ROOT/'outputs/atlas_v3/taoca/best.pt',map_location='cuda',weights_only=True));opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,a.epochs,eta_min=5e-6)
    # Cartesian is read-only and used only for fixed validation normalization.
    cart=SpatialPredictor(ROOT/'configs/trajectory_model.json').cuda().eval();cart.load_state_dict(torch.load(ROOT/'outputs/physics_v2/cartesian/best.pt',map_location='cuda',weights_only=True));cds=AllocatedWindows(ROOT/'datasets/joint_nearest',ROOT/'datasets/fenli_paired','cartesian','val',heights=heights);baseline_sse=np.zeros(5);baseline_shock=np.zeros(5);baseline_n=baseline_sn=0
    for i in range(1 if a.smoke else len(cds)):
        item=cds[i];b={k:v[None].cuda() if torch.is_tensor(v) else v for k,v in item.items()};ops=load_window(BASE/'operators',item['case'],item['start']);near,shock=regional_maps(b['target'],b['target_valid'])
        with torch.no_grad():out=cart(*[b[k] for k in ['history','mach','pose_history','dt','last_xy','templates']],10,'cartesian',b['last_base'],b['last_valid'])
        for t,op in enumerate(ops):
            q=torch.as_tensor(op['q'],device='cuda',dtype=torch.long);valid=b['target_valid'][0,t].reshape(-1)[q]>.5;edge=shock[0,t,0].reshape(-1)[q]>.5;err=(out['fields'][0,t,0].reshape(5,-1)[:,q]-b['target'][0,t].reshape(5,-1)[:,q]).square()
            baseline_sse+=err[:,valid].double().sum(1).cpu().numpy();baseline_n+=int(valid.sum());baseline_shock+=err[:,valid&edge].double().sum(1).cpu().numpy();baseline_sn+=int((valid&edge).sum())
    baseline=np.sqrt(baseline_sse/max(1,baseline_n));shock_base=np.sqrt(baseline_shock/max(1,baseline_sn));del cart,out,b
    baseline_run=json.loads((ROOT/'outputs/physics_v2/cartesian/run.json').read_text());shared=['taoca/spatial_physics.py','taoca/physics_losses.py','taoca/staged.py','taoca/joint.py','taoca/physics_data.py','taoca/domain_charts.py','datasets/joint_nearest/manifest.json']
    baseline_hashes={k.replace('\\','/'):v for k,v in baseline_run['source_sha256'].items()}
    for f in shared:assert hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==baseline_hashes[f]
    tracked=shared+['training/train_taoca_v7.py','taoca/physical_supervision_v7.py','taoca/allocated_model_v3.py','outputs/physical_v7/operator_contract.json']
    run=dict(heights=heights,parameters=sum(p.numel() for p in model.parameters()),sampled_points=123200,arguments=dict(epochs=a.epochs,seed=42,batch_size=1,lr=1e-4,smoke=a.smoke),initialization='atlas_v3/taoca/best.pt',initialization_sha256=hashlib.sha256((ROOT/'outputs/atlas_v3/taoca/best.pt').read_bytes()).hexdigest(),cartesian_checkpoint_sha256=hashlib.sha256((ROOT/'outputs/physics_v2/cartesian/best.pt').read_bytes()).hexdigest(),validation_cartesian_nrmse=baseline.tolist(),validation_cartesian_shock_nrmse=shock_base.tolist(),source_sha256={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in tracked},protocol='Exploratory TAOCA-specific inverse-consistent fine tuning; changed objective and additional budget, not a pure representation ablation. No Cartesian teacher outputs in train loss. No future geometry/field inputs. Exact legal evaluation inverse at fixed sparse points, 1/16 lattice with lead phase fixed before training.')
    if a.resume and (folder/'run.json').exists():assert json.loads((folder/'run.json').read_text())['source_sha256']==run['source_sha256']
    atomic_json(folder/'run.json',run);history=[];best=float('inf');start=1
    if a.resume and (folder/'last.pt').exists():
        saved=torch.load(folder/'last.pt',map_location='cuda',weights_only=False);model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler']);history=saved['history'];best=saved['best'];start=saved['epoch']+1;torch.set_rng_state(saved['rng'].cpu());torch.cuda.set_rng_state(saved['cuda_rng'].cpu());random.setstate(saved['python_rng']);np.random.set_state(saved['numpy_rng'])
    try:
        for epoch in range(start,(2 if a.smoke else a.epochs)+1):
            row=dict(epoch=epoch,lr=opt.param_groups[0]['lr']);begin=time.perf_counter()
            for split in ['train','val']:
                model.train(split=='train');parts={};count=0;sse=np.zeros(5);shock_sse=np.zeros(5);n=sn=0
                for batch in loaders[split]:
                    b={k:v.cuda() if torch.is_tensor(v) else v for k,v in batch.items()};ops=load_window(BASE/'operators',batch['case'][0],int(batch['start'][0]));near,shock=regional_maps(b['target'],b['target_valid'])
                    with torch.set_grad_enabled(split=='train'):
                        out=forward(model,b);loss,values=physical_loss(out,b,ops,near,shock);old,unused=native_loss(out,b,10);loss=loss+.05*old
                        if not torch.isfinite(loss):raise RuntimeError('Nonfinite V7 loss')
                        if split=='train':opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
                    count+=1
                    for k,v in dict(loss=float(loss.detach()),native=float(old.detach()),**values).items():parts[k]=parts.get(k,0)+v
                    if split=='val':
                        with torch.no_grad():
                            for t,op in enumerate(ops):
                                q=torch.as_tensor(op['q'],device='cuda',dtype=torch.long);rec=project(out['fields'][0,t],op);valid=b['target_valid'][0,t].reshape(-1)[q]>.5;edge=shock[0,t,0].reshape(-1)[q]>.5;err=(rec-b['target'][0,t].reshape(5,-1)[:,q]).square();sse+=err[:,valid].double().sum(1).cpu().numpy();n+=int(valid.sum());shock_sse+=err[:,valid&edge].double().sum(1).cpu().numpy();sn+=int((valid&edge).sum())
                row[split]={k:v/count for k,v in parts.items()}
                if split=='train':torch.cuda.synchronize();row['train_seconds']=time.perf_counter()-begin
                else:
                    nr=np.sqrt(sse/max(n,1));sr=np.sqrt(shock_sse/max(sn,1));ratios=nr/np.maximum(baseline,1e-8);shock_ratios=sr[:2]/np.maximum(shock_base[:2],1e-8);score=float(max(ratios.max(),shock_ratios.max())+.05*ratios.mean());row['val'].update(nrmse=nr.tolist(),shock_nrmse=sr.tolist(),ratios=ratios.tolist(),score=score)
            scheduler.step();history.append(row)
            if score<best:best=score;torch.save(model.state_dict(),folder/'best.pt')
            torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),history=history,best=best,epoch=epoch,rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),python_rng=random.getstate(),numpy_rng=np.random.get_state()),folder/'last.pt');atomic_json(folder/'history.json',history);atomic_json(folder/'status.json',dict(state='training',epoch=epoch,total_epochs=2 if a.smoke else a.epochs,best_validation_score=best,epoch_seconds=time.perf_counter()-begin));print(json.dumps(row),flush=True)
        if a.smoke:atomic_json(folder/'status.json',dict(state='smoke_passed',epochs=len(history)));return
        model.load_state_dict(torch.load(folder/'best.pt',map_location='cuda',weights_only=True));atomic_json(folder/'status.json',dict(state='evaluating',epoch=len(history)))
        from taoca.allocated_evaluate_v3 import evaluate
        evaluate(model,loaders['test'],'cuda',folder,'quality',False);atomic_json(folder/'status.json',dict(state='complete',epochs=len(history)))
    except Exception as e:atomic_json(folder/'status.json',dict(state='failed',error=repr(e),completed_epochs=len(history)));raise
if __name__=='__main__':main()
