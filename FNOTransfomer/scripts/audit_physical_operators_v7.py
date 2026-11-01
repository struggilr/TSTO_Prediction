"""Read-only audit of every cached physical inverse operator used in training."""
import json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'outputs/physical_v7'
def main():
    contract=json.loads((BASE/'operator_contract.json').read_text());meta=json.loads((ROOT/'datasets/joint_nearest/manifest.json').read_text());heights=contract['heights'];expected=set()
    for name,r in meta['cases'].items():
        n=len(r['times']);a=round(.7*n);v=a+round(.1*n)
        for lo,hi in [(20,a),(a,v)]:
            for start in range(max(20,lo),hi-10+1):expected.add(f'{name}_s{start:03d}')
    paths=sorted((BASE/'operators').glob('*.npz'));assert {p.stem for p in paths}==expected
    for name,wanted in contract['source_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==wanted,name
    maximum=0.;minqueries=123200;querycounts=[];hashes={}
    for path in paths:
        with np.load(path) as f:
            assert len(f.files)==10*13
            for t in range(10):
                q=f[f't{t}_q'];assert q.dtype.kind in 'iu' and len(q)>7000;assert (np.diff(q)>0).all() and q[0]>=0 and q[-1]<280*440
                assert ((q//440%4)==t%4).all() and ((q%440%4)==(t//4)%4).all()
                denominator=np.zeros(len(q),'f8')
                for k,h in enumerate(heights):
                    ids=f[f't{t}_ids{k}'];index=f[f't{t}_index{k}'];weight=f[f't{t}_weight{k}']
                    assert ids.shape==(len(index),3) and weight.shape==ids.shape
                    assert ((ids>=0)&(ids<h*440)).all() and ((index>=0)&(index<len(q))).all()
                    assert np.isfinite(weight).all() and (weight>=0).all()
                    np.add.at(denominator,index,weight.astype('f8').sum(1))
                error=float(abs(denominator-1).max());assert error<1e-5
                maximum=max(maximum,error);minqueries=min(minqueries,len(q));querycounts.append(len(q))
        hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    report=dict(verified=True,windows=len(paths),forecast_operators=len(querycounts),all_train_validation_windows_present=True,no_test_windows=True,source_hashes_verified=True,all_indices_and_coefficients_valid=True,maximum_partition_of_unity_error=maximum,minimum_queries=minqueries,mean_queries=float(np.mean(querycounts)),operator_sha256=hashes)
    (BASE/'operator_full_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='operator_sha256'}),flush=True)
if __name__=='__main__':main()
