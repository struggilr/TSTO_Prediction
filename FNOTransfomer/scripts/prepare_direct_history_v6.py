"""Direct original-triangle P1 chart sampling for history-only V6 ablation."""
import os
os.environ['MKL_THREADING_LAYER']='GNU'
import sys,json,time,hashlib,gc
from pathlib import Path
import numpy as np
import h5py
import torch
import matplotlib.tri as mtri
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from taoca.allocated_data_v3 import AllocatedWindows
from taoca.allocation_v3 import allocate
OUT=ROOT/'datasets/taoca_direct_history_v6'
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    torch.set_num_threads(4);OUT.mkdir(exist_ok=True);meta=json.loads((ROOT/'datasets/joint_nearest/manifest.json').read_text());heights=json.loads((ROOT/'outputs/atlas_v3/taoca/run.json').read_text())['heights']
    ds=AllocatedWindows(ROOT/'datasets/joint_nearest',ROOT/'datasets/fenli_paired','taoca','train',heights=heights);mean=np.array(meta['mean']);std=np.array(meta['std']);total=0;audits=[]
    contract=dict(heights=heights,normalization_sha256=digest(ROOT/'datasets/joint_nearest/manifest.json'),allocation_sha256=digest(ROOT/'taoca/allocation_v3.py'),domain_sha256=digest(ROOT/'taoca/domain_charts.py'),sampling='Original stored CFD source triangles, linear P1; normalized with unchanged train-only statistics; no extrapolation')
    for case,info in meta['cases'].items():
        record=ds.record(case);coords=allocate(record['coords'],heights);path=OUT/(case+'.h5');n=len(info['times']);start=time.perf_counter()
        if not path.exists():
            tmp=path.with_suffix('.partial.h5')
            with h5py.File(tmp,'w') as out:
                out.attrs['contract']=json.dumps(contract,sort_keys=True)
                for k,h in enumerate(heights):
                    g=out.create_group(f'chart{k}');g.create_dataset('fields',shape=(n,5,h,440),dtype='f4',compression='lzf',chunks=(1,5,h,440));g.create_dataset('valid',shape=(n,h,440),dtype='?',compression='lzf')
                for frame in range(n):
                    with h5py.File(ROOT/'datasets/fenli_paired'/case/f'{frame:04d}.h5') as f:
                        assert abs(float(f.attrs['time'])-info['times'][frame])<1e-7
                        x=f['source/x'][:];y=f['source/y'][:];triangles=f['source/triangles'][:];values=f['source/fields'][:]
                    tri=mtri.Triangulation(x,y,triangles);finder=tri.get_trifinder();points=np.stack([x,y],1)
                    for k,p in enumerate(coords):
                        query=p[frame].numpy();shape=query.shape[:-1];q=query.reshape(-1,2);idx=finder(q[:,0],q[:,1]);valid=idx>=0;ids=triangles[np.maximum(idx,0)];v=points[ids];a=v[:,1]-v[:,0];b=v[:,2]-v[:,0];r=q-v[:,0];det=a[:,0]*b[:,1]-a[:,1]*b[:,0];u=(r[:,0]*b[:,1]-r[:,1]*b[:,0])/det;w=(a[:,0]*r[:,1]-a[:,1]*r[:,0])/det;bary=np.stack([1-u-w,u,w],1)
                        if not np.isfinite(bary[valid]).all() or bary[valid].min() < -1e-6:raise RuntimeError('Invalid source barycentric coordinates')
                        field=np.einsum('cnk,nk->cn',values[:,ids],bary);field=(field-mean[:,None])/std[:,None];field[:,~valid]=0
                        if not np.isfinite(field).all():raise RuntimeError('Nonfinite sampled fields')
                        out[f'chart{k}/fields'][frame]=field.reshape(5,*shape).astype('f4');out[f'chart{k}/valid'][frame]=valid.reshape(shape)
                    del finder,tri;gc.collect()
                    print(json.dumps(dict(case=case,frame=frame+1,total=n)),flush=True)
            tmp.replace(path)
        with h5py.File(path) as f:
            if json.loads(f.attrs['contract'])!=contract:raise RuntimeError('Existing cache contract changed')
            for k,h in enumerate(heights):
                assert f[f'chart{k}/fields'].shape==(n,5,h,440)
                for i in range(n):
                    a=f[f'chart{k}/fields'][i];v=f[f'chart{k}/valid'][i]
                    if not np.isfinite(a).all() or np.any(a[:,~v]!=0):raise RuntimeError('Direct cache audit failed')
        audits.append(dict(case=case,frames=n,sha256=digest(path),seconds=time.perf_counter()-start));total+=n
        del coords;gc.collect()
    manifest=dict(**contract,verified=total==500,frames=total,cases=audits,scope='All frames prepared offline, but each dataset window reads only its past history; future labels and physical prior remain unchanged.')
    tmp=OUT/'manifest.tmp';tmp.write_text(json.dumps(manifest,indent=2));tmp.replace(OUT/'manifest.json');print(json.dumps(manifest),flush=True)
if __name__=='__main__':main()
