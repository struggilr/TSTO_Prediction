"""Strict nodal Tecplot FE reader; uses named boundaries and actual connectivity."""
import re
import numpy as np
from scipy.spatial import cKDTree

FIELDS = ['Pressure', 'Density', 'Temperature', 'VelocityX', 'VelocityY']


def ordered(edges):
    adjacency = {}
    for a, b in edges:
        adjacency.setdefault(int(a), []).append(int(b))
        adjacency.setdefault(int(b), []).append(int(a))
    if any(len(v) != 2 for v in adjacency.values()):
        raise ValueError('Boundary is not a simple closed degree-two loop')
    loop = [min(adjacency)]; previous = -1
    while True:
        nxt = next(v for v in adjacency[loop[-1]] if v != previous)
        if nxt == loop[0]: break
        if nxt in loop: raise ValueError('Boundary cycle repeats')
        previous = loop[-1]; loop.append(nxt)
    if len(loop) != len(adjacency): raise ValueError('Disconnected boundary')
    return np.array(loop)


def read(path):
    text = path.read_text(encoding='utf-8-sig')
    variables = re.findall(r'"(.*?)"', re.search(r'VARIABLES\s*=\s*(.*?)(?=DATASETAUXDATA|ZONE)', text, re.S|re.I)[1])
    zones = re.split(r'(?=^ZONE\b)', text, flags=re.M)[1:]
    nodes=[]; triangles=[]; boundaries={}; offset=0; times=[]
    for zone in zones:
        dt=re.search(r'^\s*DT\s*=\s*\(.*?\)\s*$', zone, re.M)
        if dt is None: raise ValueError('Missing DT')
        header=zone[:dt.end()]
        if re.search('VARLOCATION|VARSHARELIST|CONNECTIVITYSHARE',header,re.I):
            raise ValueError('Shared or cell-centered variables are unsupported')
        name=re.search(r'ZONE\s+T="([^"]+)"',header)[1].split('->')[-1]
        n=int(re.search(r'Nodes\s*=\s*(\d+)',header,re.I)[1]); e=int(re.search(r'Elements\s*=\s*(\d+)',header,re.I)[1])
        typ=re.search(r'ZONETYPE\s*=\s*(\w+)',header,re.I)[1].upper()
        packing=re.search(r'DATAPACKING\s*=\s*(\w+)',header,re.I)[1].upper()
        values=np.fromstring(zone[dt.end():],sep=' ')
        nv=len(variables); width={'FEQUADRILATERAL':4,'FETRIANGLE':3,'FELINESEG':2}[typ]
        if len(values)!=n*nv+e*width: raise ValueError(f'Payload mismatch: {name}')
        a=values[:n*nv].reshape((n,nv)) if packing=='POINT' else values[:n*nv].reshape(nv,n).T
        conn=values[n*nv:].astype(np.int64).reshape(e,width)-1
        if conn.min()<0 or conn.max()>=n: raise ValueError('Connectivity out of range')
        if typ=='FELINESEG': boundaries[name]=(a[:,:2],conn)
        else:
            nodes.append(a)
            triangles.append(conn[:,:3]+offset)
            if width==4: triangles.append(conn[:,[0,2,3]]+offset)
            offset+=n
        times.append(float(re.search(r'SOLUTIONTIME\s*=\s*([\d.eE+-]+)',header)[1]))
    a=np.concatenate(nodes); tri=np.concatenate(triangles)
    # Merge exactly matching interface nodes; retain the first zone's field value.
    xy, first, remap=np.unique(a[:,:2],axis=0,return_index=True,return_inverse=True)
    tri=remap[tri]; p=xy[tri]
    area=np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0])
    tri=tri[np.abs(area)>1e-16]; area=area[np.abs(area)>1e-16]
    tri[area<0]=tri[area<0][:,[0,2,1]]
    _,unique=np.unique(np.sort(tri,axis=1),axis=0,return_index=True); tri=tri[np.sort(unique)]
    tree=cKDTree(xy); edges={}
    for name,(points,conn) in boundaries.items():
        dist,ids=tree.query(points)
        if dist.max()>1e-7: raise ValueError(f'Boundary not on source mesh: {name}')
        edges[name]=ids[conn]
    if not {'smj','xmj','in','out'}<=edges.keys(): raise ValueError('Missing named boundaries')
    if np.ptp(times)>1e-9: raise ValueError('Zone time mismatch')
    return dict(x=xy[:,0],y=xy[:,1],triangles=tri,wall_loops={k:ordered(edges[k]) for k in ('smj','xmj')},
                outer_loop=ordered(np.concatenate([edges['in'],edges['out']])),
                fields={k:a[first,variables.index(k)] for k in FIELDS}, time=times[0])
