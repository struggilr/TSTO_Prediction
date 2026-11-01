"""Audit roundtrip errors and target splits on every paired source frame."""
import json,sys
from pathlib import Path
import h5py,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from taoca.data import Windows,NAMES
from taoca.evaluate import inverse,coverage


def main():
    root=ROOT/'datasets/fenli_paired';meta=json.loads((root/'manifest.json').read_text());rows=[]
    for case,records in meta['cases'].items():
        for r in records:
            with h5py.File(root/r['file'],'r') as f:
                row=dict(case=case,index=r['index'],time=r['time'],preprocess_seconds=float(f.attrs['preprocess_seconds']),geometry=json.loads(f.attrs['audit']))
                common=coverage(f,'cartesian')&coverage(f,'taoca');row['common_coverage']=float(common.mean());reference=f['source/fields'][:]
                if not np.isfinite(reference).all():raise ValueError('Nonfinite source')
                for route,names in NAMES.items():
                    fields=[]
                    for name in names:
                        c=f[route][name];v=c['fields'][:];ids=c['inv_ids'][:];bary=c['inv_bary'][:];weights=c['inv_weight'][:];valid=ids[:,0]>=0
                        if not np.isfinite(v).all() or not np.isfinite(weights).all():raise ValueError('Nonfinite data')
                        if (c['J'][:]<=0).any():raise ValueError('Nonpositive sampled Jacobian')
                        if not np.allclose(bary[valid].sum(axis=1),1,atol=2e-6):raise ValueError('Invalid barycentric sum')
                        if (bary[valid]<-1e-5).any():raise ValueError(f'Negative barycentric weight: {case} {r["index"]} {route} {name}')
                        if np.any(weights[~valid]!=0):raise ValueError('Invalid donor weight')
                        if ids[valid].max()>=v.shape[1]*v.shape[2]:raise ValueError('Donor out of range')
                        fields.append(v)
                    back,valid=inverse(f,route,np.stack(fields));error=back[:,common]-reference[:,common]
                    row[route]=dict(coverage=float(valid.mean()),rmse=np.sqrt(np.mean(error**2,axis=1)).tolist(),normalized_rmse=(np.sqrt(np.mean(error**2,axis=1))/np.array(meta['normalization']['std'])).tolist())
                rows.append(row)
        print(case,len(records),'audited',flush=True)
    counts={}
    for split in ['train','val','test']:
        d=Windows(root,'taoca',split);counts[split]=len(d)
        for case,s in d.samples:
            lo,hi={'train':(0,60),'val':(60,80),'test':(80,len(meta['cases'][case]))}[split]
            if s<lo or s+d.horizon>hi:raise ValueError('Target split leakage')
    summary=dict(frames=len(rows),windows=counts,common_coverage_range=[min(r['common_coverage'] for r in rows),max(r['common_coverage'] for r in rows)],preprocess_seconds_sum=sum(r['preprocess_seconds'] for r in rows),roundtrip={route:dict(mean_normalized_rmse=np.mean([r[route]['normalized_rmse'] for r in rows],axis=0).tolist(),coverage_range=[min(r[route]['coverage'] for r in rows),max(r[route]['coverage'] for r in rows)]) for route in NAMES},scope='P1 nodal mapping, nonconservative; node errors are not area-weighted; no neural training evidence')
    (root/'audit.json').write_text(json.dumps(dict(summary=summary,frames=rows),indent=2),encoding='utf-8');print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
