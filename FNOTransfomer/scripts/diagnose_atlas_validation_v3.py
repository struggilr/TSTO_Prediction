"""Frozen V2 model, validation-only chart/fusion diagnosis; no optimization."""
import os,sys,json,hashlib,time
os.environ['MKL_THREADING_LAYER']='GNU'
from pathlib import Path
import numpy as np
import torch
import h5py
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from taoca.physics_data import PhysicsWindows
from taoca.spatial_physics import SpatialPredictor
from taoca.physics_losses import regional_maps
from taoca.joint import sample_grid
from taoca.physics_evaluate import polygons
from taoca.chart_transfer_v3 import build_maps,apply_maps,fuse
OUT=ROOT/'outputs/atlas_v3';MODES=['quality','equal','background_x4','background_x16','finest']

def add(acc,key,pred,ref,mask):
    g=acc.setdefault(key,dict(n=0,sse=np.zeros(5)));g['n']+=int(mask.sum());g['sse']+=((pred[:,mask]-ref[:,mask])**2).sum(1)
def finish(acc):return {k:dict(n=v['n'],nrmse=np.sqrt(v['sse']/max(1,v['n'])).tolist()) for k,v in acc.items()}

def main():
    torch.set_num_threads(4);OUT.mkdir(parents=True,exist_ok=True)
    contract=dict(split='validation only',fusion_candidates=MODES,selection='minimum mean five-channel pooled NRMSE, with mean near-wall NRMSE <= 1.03 times quality baseline',representation_candidates=[[70,70,70,70],[49,49,112,70],[42,42,126,70],[35,35,140,70]],representation_gate='>=5% reduction in mean channel roundtrip NRMSE, >=10% reduction in mean p/rho high-gradient roundtrip NRMSE; <=10% mean wall roundtrip regression; truth-supported coverage no worse by >0.2 percentage point; legal predicted-mask coverage >=99.5%',notes='Geometry selection uses validation target fields as evaluation labels, never as model inputs or online geometry. Test set is not accessed by this script.')
    (OUT/'selection_contract.json').write_text(json.dumps(contract,indent=2))
    ds=PhysicsWindows(ROOT/'datasets/joint_nearest',ROOT/'datasets/fenli_paired','taoca','val')
    model=SpatialPredictor(ROOT/'outputs/shared_trajectory/model.json').cuda().eval();checkpoint=ROOT/'outputs/physics_v2/taoca/best.pt';model.load_state_dict(torch.load(checkpoint,map_location='cuda',weights_only=True))
    totals={};begin=time.perf_counter();native=OUT/'validation_native';native.mkdir(exist_ok=True)
    for index in range(len(ds)):
        item=ds[index];b={k:v[None].cuda() if torch.is_tensor(v) else v for k,v in item.items()}
        with torch.no_grad():
            out=model(*[b[k] for k in ['history','mach','pose_history','dt','last_xy','templates']],10,'taoca',b['last_base'],b['last_valid'])
            sampled=sample_grid(b['target'],out['xy']);tv=sample_grid(b['target_valid'][:,:,None],out['xy'])>.999
        pred=out['fields'][0].cpu().numpy();xy=out['xy'][0].cpu().numpy();masks=out['geometry'][0,:, :,0].cpu().numpy()>.5;pose=out['pose'][0].cpu().numpy()
        with h5py.File(native/f'{item["case"]}_s{item["start"]:03d}.h5','w') as h:
            for key,value in dict(fields=pred,xy=xy,mask=masks,pose=pose,reference=item['target'].numpy(),valid=item['target_valid'].numpy(),last_xy=item['last_xy'].numpy(),last_pose=item['pose_history'][-1].numpy(),templates=item['templates'].numpy()).items():h.create_dataset(key,data=value,compression='lzf')
            h.attrs['case']=item['case'];h.attrs['start']=item['start']
        near,shock=regional_maps(item['target'][None],item['target_valid'][None]);near=near[0,:,0].numpy()>.5;shock=shock[0,:,0].numpy()>.5
        for t in range(10):
            maps=build_maps(xy[t],masks[t],polygons(item['templates'].numpy(),pose[t]));rec,quality=apply_maps(pred[t],maps)
            truth_rec,truth_q=apply_maps(sampled[0,t].cpu().numpy(),maps,tv[0,t,:,0].cpu().numpy())
            ref=item['target'][t].numpy();fluid=item['target_valid'][t].numpy()>.5
            for mode in MODES:
                field,support=fuse(rec,quality,mode);rt,rs=fuse(truth_rec,truth_q,mode)
                for region,mask in [('all',fluid),('wall',fluid&near[t]),('shock',fluid&shock[t])]:
                    add(totals,f'{mode}/{region}',field,ref,mask&support)
                    add(totals,f'roundtrip/{mode}/{region}',rt,ref,mask&rs)
            for k in range(4):
                support=np.isfinite(rec[k]).all(0)
                for region,mask in [('all',fluid),('wall',fluid&near[t]),('shock',fluid&shock[t])]:add(totals,f'chart{k}/{region}',rec[k],ref,mask&support)
        print(json.dumps(dict(validation_window=index+1,total=len(ds),case=item['case'])),flush=True)
    metrics=finish(totals);basewall=np.mean(metrics['quality/wall']['nrmse'])
    allowed=[m for m in MODES if np.mean(metrics[m+'/wall']['nrmse'])<=1.03*basewall]
    selected=min(allowed,key=lambda m:np.mean(metrics[m+'/all']['nrmse']))
    report=dict(split='validation',windows=len(ds),forecast_instances=len(ds)*10,fusion_selected=selected,metrics=metrics,checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),seconds=time.perf_counter()-begin,scope='Chart-specific errors have different supports; only fusion candidates share union support. No test data or per-pixel truth-oracle selection.')
    (OUT/'fusion_diagnosis.json').write_text(json.dumps(report,indent=2));print(json.dumps(dict(selected=selected,metrics={m:metrics[m+'/all'] for m in MODES}),indent=2),flush=True)
if __name__=='__main__':main()
