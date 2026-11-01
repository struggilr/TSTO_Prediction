"""Compare all forecast occurrences on identical legal physical support."""
import json
from pathlib import Path
import numpy as np
import h5py
ROOT=Path(__file__).resolve().parents[1]/'outputs/physics_v2'
def main():
    a=ROOT/'cartesian';b=ROOT/'taoca'
    runs=[json.loads((p/'run.json').read_text()) for p in [a,b]]
    for key in ['seed','epochs','batch_size','lr']:
        if runs[0]['arguments'][key]!=runs[1]['arguments'][key]:raise ValueError('Unmatched '+key)
    if runs[0]['parameters']!=runs[1]['parameters'] or runs[0]['source_sha256']!=runs[1]['source_sha256']:raise ValueError('Unmatched model parameters or source versions')
    paths=sorted((a/'predictions').glob('*.h5'));other=sorted((b/'predictions').glob('*.h5'))
    if len(paths)!=54 or [p.name for p in paths]!=[p.name for p in other]:raise ValueError('Expected matched 54 full test windows')
    totals={name:np.zeros(5) for name in ['FNO_Cartesian','FNO_TAOCA','Persistence']};count=truth=0;rows=[]
    for path in paths:
        with h5py.File(path) as fa,h5py.File(b/'predictions'/path.name) as fb:
            x=fa['predicted'][:];y=fb['predicted'][:];ref=fa['reference'][:]
            if not np.array_equal(ref,fb['reference'][:]):raise ValueError('Different reference fields')
            if not np.array_equal(fa['pose'][:],fb['pose'][:]):raise ValueError('Trajectory outputs differ')
            valid=fa['valid'][:].astype(bool);common=valid&fa['support'][:]&fb['support'][:]&np.isfinite(x).all(1)&np.isfinite(y).all(1)
            for t in range(10):
                n=int(common[t].sum());count+=n;truth+=int(valid[t].sum());row=dict(window=path.stem,lead=t+1,n=n)
                for name,field in [('FNO_Cartesian',x[t]),('FNO_TAOCA',y[t]),('Persistence',fa['persistence'][:])]:
                    error=((field[:,common[t]]-ref[t][:,common[t]])**2).sum(1);totals[name]+=error;row[name]=np.sqrt(error/max(n,1)).tolist()
                rows.append(row)
    result=dict(forecast_instances=len(rows),common_coverage=count/max(truth,1),nrmse={name:np.sqrt(val/max(count,1)).tolist() for name,val in totals.items()},shared_trajectories_verified=True,scope='Single-seed corrected spatial/physics V2 comparison; all overlapping 54x10 instances, no missing-value imputation.')
    (ROOT/'comparison.json').write_text(json.dumps(result,indent=2));(ROOT/'common_frame_metrics.json').write_text(json.dumps(rows));print(json.dumps(result))
if __name__=='__main__':main()
