"""Chronology/source availability audit; does not change splits or normalization."""
import json,sys,datetime
from pathlib import Path
import numpy as np
import h5py
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/shock_diagnosis_v9'
def main():
    meta=json.loads((ROOT/'datasets/joint_nearest/manifest.json').read_text());records=[];count=0;features=[]
    for case,row in meta['cases'].items():
        times=np.array(row['times']);assert np.all(np.diff(times)>0)
        with h5py.File(ROOT/'datasets/joint_nearest'/f'{case}.h5') as f:
            assert np.allclose(f['time'][:],times,rtol=0,atol=1e-7)
            assert f['fields'].shape==(len(times),5,280,440)
            assert f['pose'].shape==(len(times),6)
            a=round(.7*len(times));v=a+round(.1*len(times))
            for i,t in enumerate(times):
                fields=f['fields'][i];valid=f['valid'][i].astype(bool);assert np.isfinite(fields[:,valid]).all()
                path=ROOT/'datasets/fenli_paired'/case/f'{i:04d}.h5'
                with h5py.File(path) as raw:
                    assert abs(float(raw.attrs['time'])-t)<1e-7
                    q=raw['source/fields'];assert q.shape[0]==5
                    ids=np.unique(np.linspace(0,q.shape[1]-1,512,dtype=int));sample=q[:,ids];assert np.isfinite(sample).all()
                    if i<a:features.append(np.stack([sample[1]*sample[2],sample[0]],1))
                count+=1
            records.append(dict(case=case,frames=len(times),train_target_end=a,val_target_end=v,test_target_end=len(times)))
            print(json.dumps(records[-1]),flush=True)
    fit=np.concatenate(features,axis=0);xx=fit[:,0];yy=fit[:,1];design=np.stack([xx,np.ones_like(xx)],1);R,offset=np.linalg.lstsq(design,yy,rcond=None)[0];res=design@np.array([R,offset])-yy
    result=dict(verified=True,frames=count,cases=records,all_joint_source_times_match=True,valid_joint_fields_finite=True,raw_sample_fields_finite=True,training_source_eos_fit=dict(R=float(R),offset=float(offset),pressure_std_normalized_rmse=float(np.sqrt(np.mean(res**2))/meta['std'][0])),limitations=['Sampled source fields only; not all source nodes checked','EOS fit does not identify gamma or caloric gas model','This audit does not establish conservation or raw-source reconstruction equivalence'],checked_at=datetime.datetime.now().isoformat())
    (OUT/'data_consistency_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
