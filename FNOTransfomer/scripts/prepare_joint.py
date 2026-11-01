"""Legacy nearest-grid layout with corrected named walls and actual time order."""
import sys,json,re,argparse
from pathlib import Path
import numpy as np
import h5py
import shapely
from scipy.spatial import cKDTree
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from taoca.tecplot import read,FIELDS
from taoca.tecplot import ordered


def cached_frame(path,case,index):
    cache=ROOT/'datasets/fenli_paired'/case/f'{index:04d}.h5'
    if not cache.exists():
        d=read(path);points=np.c_[d['x'],d['y']]
        return points,np.stack([d['fields'][name] for name in FIELDS]),[points[d['wall_loops'][name]] for name in ['smj','xmj']]
    with h5py.File(cache) as f:points=np.c_[f['source/x'][:],f['source/y'][:]];values=f['source/fields'][:]
    # Read only the two small named line-zone payloads; reuse audited fluid nodes.
    text=path.read_text(encoding='utf-8-sig');walls={}
    for zone in re.split(r'(?=^ZONE\b)',text,flags=re.M)[1:]:
        title=re.search(r'ZONE\s+T="([^"]+)"',zone)
        if title is None or title[1].split('->')[-1] not in ['smj','xmj']:continue
        name=title[1].split('->')[-1];end=re.search(r'^\s*DT\s*=\s*\(.*?\)\s*$',zone,re.M).end()
        n=int(re.search(r'Nodes\s*=\s*(\d+)',zone[:end],re.I)[1]);nv=len(re.findall(r'"(.*?)"',re.search(r'VARIABLES\s*=\s*(.*?)(?=DATASETAUXDATA|ZONE)',text,re.S|re.I)[1]))
        payload=np.fromstring(zone[end:],sep=' ');matrix=payload[:n*nv].reshape(n,nv) if 'DATAPACKING=POINT' in zone[:end].replace(' ','') else payload[:n*nv].reshape(nv,n).T
        ids=ordered(payload[n*nv:].astype('i8').reshape(-1,2)-1);walls[name]=matrix[ids,:2]
    if set(walls)!=set(['smj','xmj']):raise ValueError('Missing named wall payloads')
    return points,values,[walls[name] for name in ['smj','xmj']]


def pose_of(points):
    center=points.mean(0);_,v=np.linalg.eigh(np.cov((points-center).T))
    axis=v[:,-1];return np.r_[center,np.arctan2(axis[1],axis[0])]


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=Path('E:/fenli'));p.add_argument('--output',type=Path,default=ROOT/'datasets/joint_nearest');p.add_argument('--case',default='all');p.add_argument('--limit',type=int);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    yy,xx=np.meshgrid(np.linspace(-20,50,280),np.linspace(-10,100,440),indexing='ij');query=np.c_[xx.ravel(),yy.ravel()]
    cases=sorted(a.source.glob('Ma*')) if a.case=='all' else [a.source/a.case]
    for case in cases:
        stamp=lambda f:float(re.search(r'-(\d+\.\d{6})\.5',f.name)[1])
        files=sorted(case.glob('*.dat'),key=stamp)[:a.limit];destination=a.output/(case.name+'.h5')
        if destination.exists():continue
        temporary=destination.with_suffix('.partial.h5');previous=None
        with h5py.File(temporary,'a') as f:
            if 'fields' not in f:
                f.create_dataset('fields',(len(files),5,280,440),dtype='f4',compression='lzf');f.create_dataset('valid',(len(files),280,440),dtype='u1',compression='lzf');f.create_dataset('pose',(len(files),6),dtype='f8');f.create_dataset('time',data=[stamp(x) for x in files]);f.attrs['completed']=0
            fields=f['fields'];valid=f['valid'];poses=f['pose'];completed=int(f.attrs.get('completed',0))
            if completed:previous=poses[completed-1].reshape(2,3)
            for i,path in enumerate(files):
                if i<completed and (i!=0 or 'templates' in f):continue
                points,values,walls=cached_frame(path,case.name,i);_,ids=cKDTree(points).query(query)
                solid=shapely.union_all([shapely.Polygon(w) for w in walls]);mask=~shapely.intersects_xy(solid,query[:,0],query[:,1])
                v=values[:,ids];v[:,~mask]=0
                fields[i]=v.reshape(5,280,440);valid[i]=mask.reshape(280,440)
                pose=np.array([pose_of(w) for w in walls])
                if previous is not None:pose[:,2]+=np.pi*np.round((previous[:,2]-pose[:,2])/np.pi)
                poses[i]=pose.ravel();previous=pose
                if i==0:
                    templates=[]
                    for wall,state in zip(walls,pose):
                        # Arc-length sample a fixed known shape (256 vertices).
                        edge=np.roll(wall,-1,axis=0)-wall;length=np.linalg.norm(edge,axis=1);arc=np.r_[0,np.cumsum(length)]
                        q=np.arange(256)*arc[-1]/256;j=np.minimum(np.searchsorted(arc,q,side='right')-1,len(wall)-1)
                        vertices=wall[j]+((q-arc[j])/length[j])[:,None]*edge[j]-state[:2]
                        co=np.cos(state[2]);si=np.sin(state[2]);templates.append(vertices@np.array([[co,-si],[si,co]]))
                    f.create_dataset('templates',data=np.array(templates))
                f.attrs['completed']=max(completed,i+1);f.flush()
                print(json.dumps(dict(case=case.name,index=i,frames=len(files))),flush=True)
        temporary.replace(destination)
    if a.limit:return
    records={};total=np.zeros(5);sq=np.zeros(5);count=0
    for path in sorted(a.output.glob('Ma*.h5')):
        if '.partial.' in path.name or '_geometry' in path.stem:continue
        with h5py.File(path) as f:
            times=f['time'][:];assert np.all(np.diff(times)>0);records[path.stem]=dict(times=times.tolist(),frames=len(times))
            for i in range(round(.7*len(times))):
                v=f['fields'][i][:,f['valid'][i].astype(bool)].astype('f8');total+=v.sum(1);sq+=(v*v).sum(1);count+=v.shape[1]
    if len(records)!=5:
        print(json.dumps(dict(status='waiting_for_remaining_cases',complete=len(records))),flush=True);return
    mean=total/count;std=np.sqrt(np.maximum(sq/count-mean**2,1e-12))
    (a.output/'manifest.json').write_text(json.dumps(dict(cases=records,mean=mean.tolist(),std=std.tolist(),fields=FIELDS,split='cgns2npz 70/10/remaining percent by time; target-contained windows',grid=[280,440],interpolation='nearest; named wall polygons; no future geometry input',pose='boundary-node centroid and continuous principal-axis angle; not mass center'),indent=2),encoding='utf-8')


if __name__=='__main__':main()
