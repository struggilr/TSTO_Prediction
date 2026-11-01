"""Select local history surrogate on validation only; test is final evaluation."""
import sys,json
from pathlib import Path
import numpy as np
import h5py
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    root=ROOT/'datasets/joint_nearest';meta=json.loads((root/'manifest.json').read_text());samples={s:[] for s in ['train','val','test']}
    for case,r in meta['cases'].items():
        with h5py.File(root/(case+'.h5')) as f:q=f['pose'][:];time=f['time'][:]
        n=len(q);a=round(.7*n);v=a+round(.1*n)
        for split,(lo,hi) in dict(train=(20,a),val=(a,v),test=(v,n)).items():
            for start in range(max(20,lo),hi-10+1):samples[split].append((q[start-20:start],time[start-20:start],q[start:start+10]))
    # Scale displacement errors by observed training increments, not domain size.
    scale=np.maximum(np.std(np.concatenate([y-h[-1] for h,t,y in samples['train']]),axis=0),1e-5)
    def score(degree,gain,split):
        errors=[]
        for h,t,y in samples[split]:
            u=(t-t[-1])/(t[-1]-t[0]);future=np.arange(1,11)*(t[-1]-t[-2])/(t[-1]-t[0])
            coef=np.polynomial.polynomial.polyfit(u,h-h[-1],degree)
            local=h[-1]+np.stack([np.polynomial.polynomial.polyval(future,coef[:,i]) for i in range(6)],1)
            cv=h[-1]+np.arange(1,11)[:,None]*(h[-1]-h[-2])
            errors.append(cv+gain*(local-cv)-y)
        e=np.array(errors);return float(np.mean((e/scale)**2)),np.sqrt((e*e).mean((0,1))).tolist()
    candidates=[]
    for degree in [1,2,3]:
        for gain in [0.,.25,.5,.75,1.]:
            value,rmse=score(degree,gain,'val');candidates.append(dict(degree=degree,gain=gain,validation_score=value,validation_rmse=rmse))
    best=min(candidates,key=lambda x:x['validation_score']);out=ROOT/'outputs/shared_trajectory';out.mkdir(exist_ok=True)
    report=dict(history=20,horizon=10,**best,training_increment_scale=scale.tolist(),candidates=candidates,test_rmse=score(best['degree'],best['gain'],'test')[1],constant_velocity_test_rmse=score(1,0,'test')[1],protocol='local polynomial kinematic surrogate; no fitted future trajectory; hyperparameters selected on validation only; fixed for both A/B; not 3DOF force integration')
    (out/'model.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':main()
