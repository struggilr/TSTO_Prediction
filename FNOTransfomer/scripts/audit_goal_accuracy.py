"""Independently audit five-channel superiority on fixed, baseline-defined support."""
import os
os.environ['MKL_THREADING_LAYER']='GNU'
import sys,argparse,json,hashlib,datetime
from pathlib import Path
import numpy as np
import h5py
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from taoca.physics_losses import regional_maps
def calculate(folder):
    cart=ROOT/'outputs/physics_v2/cartesian';v3=ROOT/'outputs/atlas_v3/taoca';names={p.name for p in (cart/'predictions').glob('*.h5')};assert len(names)==54 and names=={p.name for p in folder.glob('*.h5')}
    sse={r:{m:np.zeros(5) for m in ['Cartesian','candidate']} for r in ['all','wall','shock']};counts={r:0 for r in sse};missing=0;reference_count=0
    for name in sorted(names):
        with h5py.File(cart/'predictions'/name) as a,h5py.File(v3/'predictions'/name) as old,h5py.File(folder/name) as b:
            ref=a['reference'][:];valid=a['valid'][:].astype(bool);pose=a['pose'][:]
            assert np.array_equal(ref,b['reference'][:]) and np.array_equal(ref,old['reference'][:])
            assert np.array_equal(pose,b['pose'][:]) and np.array_equal(pose,old['pose'][:])
            assert np.array_equal(valid,b['valid'][:].astype(bool))
            pred=a['predicted'][:];candidate=b['predicted'][:]
            fixed=valid&a['support'][:].astype(bool)&old['support'][:].astype(bool)&np.isfinite(pred).all(1)&np.isfinite(old['predicted'][:]).all(1)
            supported=b['support'][:].astype(bool)&np.isfinite(candidate).all(1);missing+=int((fixed&~supported).sum());use=fixed&supported;reference_count+=int(valid.sum())
            near,shock=regional_maps(torch.from_numpy(ref)[None],torch.from_numpy(valid.astype('f4'))[None]);masks={'all':use,'wall':use&(near[0,:,0].numpy()>.5),'shock':use&(shock[0,:,0].numpy()>.5)}
            for region,mask in masks.items():
                counts[region]+=int(mask.sum())
                for key,value in [('Cartesian',pred),('candidate',candidate)]:sse[region][key]+=np.square(value.transpose(1,0,2,3)[:,mask].astype('f8')-ref.transpose(1,0,2,3)[:,mask]).sum(1)
    metrics={r:{m:np.sqrt(sse[r][m]/max(1,counts[r])).tolist() for m in sse[r]} for r in sse}
    global_wins=[x<y for x,y in zip(metrics['all']['candidate'],metrics['all']['Cartesian'])];shock_wins=[x<y for x,y in zip(metrics['shock']['candidate'][:2],metrics['shock']['Cartesian'][:2])]
    return dict(forecast_instances=540,baseline_defined_support=True,missing_from_fixed_support=missing,common_coverage=counts['all']/reference_count,metrics=metrics,global_channel_wins=global_wins,shock_pressure_density_wins=shock_wins,accuracy_criterion_met=missing==0 and all(global_wins) and all(shock_wins))
def main():
    p=argparse.ArgumentParser();p.add_argument('--trial',choices=['physical_v7','metric_v8'],required=True);p.add_argument('--verify-reinference',action='store_true');a=p.parse_args();torch.set_num_threads(4);out=ROOT/'outputs'/a.trial;folder=out/'taoca';run=json.loads((folder/'run.json').read_text());history=json.loads((folder/'history.json').read_text());status=json.loads((folder/'status.json').read_text())
    assert status['state']=='complete' and len(history)==run['arguments']['epochs']==60
    for name,wanted in run['source_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==wanted,name
    assert hashlib.sha256((ROOT/'outputs/physics_v2/cartesian/best.pt').read_bytes()).hexdigest()==run['cartesian_checkpoint_sha256']
    result=calculate(folder/'predictions');result.update(trial=a.trial,checked_at=datetime.datetime.now().isoformat(),checkpoint_sha256=hashlib.sha256((folder/'best.pt').read_bytes()).hexdigest(),source_hashes_verified=True,reinference_verified=False,scope='Fixed original Cartesian/V3 common support; candidate cannot improve scores by dropping points. Float64 SSE. Single seed and repeatedly viewed test; not proof of independent generalization or equal-budget representation superiority.')
    if a.verify_reinference and result['accuracy_criterion_met']:
        from torch.utils.data import DataLoader
        from taoca.allocated_data_v3 import AllocatedWindows
        from taoca.allocated_evaluate_v3 import evaluate
        if a.trial=='physical_v7':from taoca.allocated_model_v3 import AllocatedPredictor as Predictor
        else:from taoca.metric_model_v8 import MetricPredictor as Predictor
        model=Predictor(ROOT/'outputs/shared_trajectory/model.json',run['heights']).cuda().eval();model.load_state_dict(torch.load(folder/'best.pt',map_location='cuda',weights_only=True));ds=AllocatedWindows(ROOT/'datasets/joint_nearest',ROOT/'datasets/fenli_paired','taoca','test',heights=run['heights']);verified=out/'independent_reinference';verified.mkdir(exist_ok=True);evaluate(model,DataLoader(ds,batch_size=1,num_workers=0),'cuda',verified,'quality',False)
        delta=0.
        for path in (verified/'predictions').glob('*.h5'):
            with h5py.File(path) as b,h5py.File(folder/'predictions'/path.name) as old:
                assert np.array_equal(b['support'][:],old['support'][:]);mask=b['support'][:].astype(bool);delta=max(delta,float(np.max(np.abs(b['predicted'][:].transpose(1,0,2,3)[:,mask]-old['predicted'][:].transpose(1,0,2,3)[:,mask]))))
        fresh=calculate(verified/'predictions');result.update(independent_metrics=fresh,reinference_max_normalized_difference=delta,reinference_verified=delta<1e-5 and fresh['accuracy_criterion_met'])
    analysis=json.loads((out/'analysis_status.json').read_text()) if (out/'analysis_status.json').exists() else {};clouds=list((out/'analysis/figures/clouds').glob('*.png'));result['all_2700_cloud_figures_present']=len(clouds)==2700;result['analysis_complete']=analysis.get('state')=='complete' and (out/'analysis/分析报告.md').exists();result['ready_for_completion_review']=result['accuracy_criterion_met'] and result['reinference_verified'] and result['analysis_complete'] and result['all_2700_cloud_figures_present']
    (out/'goal_accuracy_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
