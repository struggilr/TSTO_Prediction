"""Create paired P1 datasets directly from the original five Tecplot trajectories."""
import argparse, json, time, os, gc
from pathlib import Path
import h5py
import numpy as np
import matplotlib.tri as mtri
import shapely
from .tecplot import read, FIELDS
from . import geometry as g

NAMES=['smj','xmj','outer','gap']


def barycentric(points, triangles, query, finder):
    ti=finder(query[:,0],query[:,1]); valid=ti>=0
    ids=triangles[np.maximum(ti,0)]; p=points[ids]
    det=g.cross_2d(p[:,1]-p[:,0],p[:,2]-p[:,0])
    b=g.cross_2d(query-p[:,0],p[:,2]-p[:,0])/det
    c=g.cross_2d(p[:,1]-p[:,0],query-p[:,0])/det
    weights=np.column_stack([1-b-c,b,c]); weights[~valid]=0
    return ids, weights, valid


def project_to_closed_curve(query, curve):
    """Return nearest polyline segment and interpolation fraction for closed loops."""
    query=np.asarray(query,dtype=np.float64);curve=np.asarray(curve,dtype=np.float64)
    edge=np.roll(curve,-1,axis=0)-curve
    denom=np.maximum(np.sum(edge*edge,axis=1),1e-30)
    rel=query[:,None,:]-curve[None,:,:]
    frac=np.clip(np.sum(rel*edge[None,:,:],axis=2)/denom[None,:],0.,1.)
    projection=curve[None,:,:]+frac[:,:,None]*edge[None,:,:]
    nearest=np.argmin(np.sum((query[:,None,:]-projection)**2,axis=2),axis=1)
    return nearest,frac[np.arange(len(query)),nearest]


def boundary_trace(d,name,chart):
    """Sample exact solver boundary data onto a chart trace and create inverse donors.

    Triangulation finders are unreliable for queries lying exactly on a boundary
    segment.  We therefore use a one-dimensional P1 trace interpolation for the
    CFD wall and far-field loops, while retaining the ordinary two-dimensional
    donor search in the interior.
    """
    if name in d['wall_loops']:
        loop=np.asarray(d['wall_loops'][name],dtype=np.int64);row=0
    elif name=='outer' and 'outer_loop' in d:
        loop=np.asarray(d['outer_loop'],dtype=np.int64);row=chart['x'].shape[0]-1
    else:
        return None
    points=np.column_stack([d['x'],d['y']])
    source_curve=points[loop]
    # Forward boundary restriction: exact source polyline P1 interpolation.
    chart_curve=np.column_stack([chart['x'][row],chart['y'][row]])
    segment,fraction=project_to_closed_curve(chart_curve,source_curve)
    nxt=(segment+1)%len(loop)
    for channel,field_name in enumerate(FIELDS):
        values=d['fields'][field_name][loop]
        chart['field'][channel,row,:]=(1.-fraction)*values[segment]+fraction*values[nxt]
    chart['mask'][row,:]=True
    chart['source_valid'][row,:]=True

    # Inverse boundary restriction: project every original boundary node to the
    # corresponding logical chart trace and encode 1-D interpolation as a donor.
    inv_seg,inv_fraction=project_to_closed_curve(source_curve,chart_curve)
    nx=chart['x'].shape[1]
    col2=(inv_seg+1)%nx
    flat=row*nx+inv_seg
    ids=np.column_stack([flat,row*nx+col2,flat]).astype(np.int32)
    bary=np.column_stack([1.-inv_fraction,inv_fraction,np.zeros(len(loop))])
    xi=np.mod((inv_seg+inv_fraction)/nx,1.)
    eta=float(row/(chart['x'].shape[0]-1))
    weight=g.chart_weight(name,np.full(len(loop),eta),xi)
    return loop,ids,bary,weight


def sample(d,charts):
    points=np.column_stack([d['x'],d['y']]); triangles=d['triangles']
    if '_source' not in d:d['_source']=mtri.Triangulation(*points.T,triangles)
    source=d['_source']; finder=source.get_trifinder()
    bodies=shapely.union_all([shapely.Polygon(points[v]) for v in d['wall_loops'].values()])
    values=np.column_stack([d['fields'][f] for f in FIELDS]); result=[]
    for name,c in charts.items():
        xy=np.stack([c['x'],c['y']],axis=-1); shape=c['x'].shape
        ids,bary,valid=barycentric(points,triangles,xy.reshape(-1,2),finder)
        field=np.einsum('nk,nkc->nc',bary,values[ids]); field[~valid]=0
        c['source_valid']=valid.reshape(shape)
        tri=g.structured_triangles(*shape,c['periodic'])
        polygons=shapely.polygons(xy.reshape(-1,2)[tri])
        collision=shapely.area(shapely.intersection(polygons,bodies))>1e-12*max(bodies.area,1.)
        signed=g.cross_2d(xy.reshape(-1,2)[tri[:,1]]-xy.reshape(-1,2)[tri[:,0]],xy.reshape(-1,2)[tri[:,2]]-xy.reshape(-1,2)[tri[:,0]])
        c['valid_donor_cells']=valid[tri].all(axis=1)&~collision&(signed>0)
        c['fields']={'probe':field[:,0].reshape(shape)}
        # All geometric invalidity remains explicit; never extrapolate into holes.
        c['field']=field.T.reshape(5,*shape).astype('f4')
        c['mask']=valid.reshape(shape)
        trace=boundary_trace(d,name,c)
        if trace is not None:
            # Recheck donor cells after repairing boundary-row validity.
            c['valid_donor_cells']=c['source_valid'].ravel()[tri].all(axis=1)&~collision&(signed>0)
        _,eta,xi,inv_ids,inv_bary=g.inverse_chart_values(c,d['x'],d['y'],'probe')
        weight=g.chart_weight(name,eta,xi) if name!='cartesian' else np.isfinite(eta).astype(float)
        if trace is not None:
            loop,trace_ids,trace_bary,trace_weight=trace
            inv_ids[loop]=trace_ids;inv_bary[loop]=trace_bary
            eta[loop]=float(0. if name!='outer' else 1.)
            xi[loop]=np.mod((trace_ids[:,0]%(c['x'].shape[0]*c['x'].shape[1]))%c['x'].shape[1] + trace_bary[:,1],c['x'].shape[1])/c['x'].shape[1]
            weight[loop]=trace_weight
        weight[inv_ids[:,0]<0]=0
        result.append(dict(name=name,nonpositive_cells=int((signed<=0).sum()),valid_fraction=float(valid.mean()),solid_cells_masked=int(collision.sum()),coverage=float((weight>0).mean())))
        c['inv_ids']=inv_ids;c['inv_bary']=inv_bary.astype('f4');c['inv_weight']=weight.astype('f4')
    return result


def body_chart(d,name,anchor,ny,nx,scale):
    p=g._oriented_boundary(d,d['wall_loops'][name])
    p=np.roll(p,-np.lexsort((p[:,1],p[:,0]))[0],axis=0)
    key='corner_'+name
    if key not in anchor:
        prev=p-np.roll(p,1,axis=0);nxt=np.roll(p,-1,axis=0)-p
        norm=np.linalg.norm(prev,axis=1)*np.linalg.norm(nxt,axis=1)
        keep=np.flatnonzero((abs(g.cross_2d(prev,nxt))>1e-6*norm)|(np.sum(prev*nxt,axis=1)<=0))
        if len(keep)>nx:raise ValueError('Increase nx to preserve boundary corners')
        q=p[keep];length=np.linalg.norm(np.roll(q,-1,axis=0)-q,axis=1)
        allocation=np.ones(len(q),dtype=int)
        for _ in range(nx-len(q)):
            allocation[np.argmax(length/allocation)]+=1
        anchor[key]=(len(p),keep,allocation)
    count,keep,pieces=anchor[key]
    if len(p)!=count:raise ValueError('Boundary node topology changed')
    p=p[keep];wall=np.vstack([u+np.arange(n)[:,None]/n*(v-u) for u,v,n in zip(p,np.roll(p,-1,axis=0),pieces)])
    edges=np.roll(wall,-1,axis=0)-wall;normal=np.column_stack([-edges[:,1],edges[:,0]])
    normal/=np.linalg.norm(normal,axis=1)[:,None];bisector=normal+np.roll(normal,1,axis=0)
    den=np.sum(bisector*normal,axis=1)
    if den.min()<1e-8:raise ValueError('Unsupported boundary cusp')
    displacement=bisector/den[:,None];radial=np.sinh(2*np.linspace(0,1,ny))/np.sinh(2)
    for offset in 3*scale*.8**np.arange(60):
        xy=wall[None,:,:]+radial[:,None,None]*offset*displacement[None,:,:]
        c=dict(x=xy[:,:,0],y=xy[:,:,1],periodic=True)
        c.update(g.logical_metrics(c['x'],c['y'],True))
        tri=g.structured_triangles(ny,nx,True);pp=xy.reshape(-1,2)
        if c['J'].min()>0 and g.cross_2d(pp[tri[:,1]]-pp[tri[:,0]],pp[tri[:,2]]-pp[tri[:,0]]).min()>0:return c
    raise ValueError('No positive corner-preserving O chart')


def create_charts(d,anchor,ny,nx,scale):
    outer=g.outer_annulus_chart(d,nx,ny,.8*scale,anchor,.05)
    charts={name:body_chart(d,name,anchor,ny,nx,scale) for name in ('smj','xmj')}
    charts.update(outer=outer,gap=g.gap_h_chart(d,outer,nx,ny,anchor))
    return charts


def write_frame(path,d,charts,cart,audit,origin,elapsed):
    temporary=path.with_suffix('.partial.h5')
    with h5py.File(temporary,'w') as f:
        f.attrs.update(time=d['time'],source=str(origin),preprocess_seconds=elapsed,geometry_protocol='first-frame fixed outer anchor; instantaneous target geometry supplied to both routes',boundary_trace_protocol='nearest-polyline projection with 1-D P1 boundary restriction and projected trace donors for body and far-field boundary nodes')
        src=f.create_group('source')
        for k,a in dict(x=d['x'],y=d['y'],triangles=d['triangles'],fields=np.stack([d['fields'][v] for v in FIELDS])).items():
            src.create_dataset(k,data=a,compression='lzf')
        for route,cs in [('taoca',charts),('cartesian',{'cartesian':cart})]:
            r=f.create_group(route)
            for name,c in cs.items():
                group=r.create_group(name)
                for k,a in dict(fields=c['field'],mask=c['mask'],xy=np.stack([c['x'],c['y']]).astype('f4'),inv_ids=c['inv_ids'],inv_bary=c['inv_bary'],inv_weight=c['inv_weight'],J=c['J'].astype('f4')).items():
                    group.create_dataset(k,data=a,compression='lzf')
                group.attrs['periodic']=c['periodic']
        f.attrs['audit']=json.dumps(audit)
    os.replace(temporary,path)


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=Path('E:/fenli'));p.add_argument('--output',type=Path,default=Path('datasets/fenli_paired'));p.add_argument('--case',default='all');p.add_argument('--limit',type=int);p.add_argument('--start',type=int,default=0);p.add_argument('--stop',type=int);p.add_argument('--ny',type=int,default=48);p.add_argument('--nx',type=int,default=256)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    cases=sorted(a.source.glob('Ma*')) if a.case=='all' else [a.source/a.case]
    for case in cases:
        files=list(case.glob('*.dat'))
        # These exports repeat SOLUTIONTIME=0: recover actual times from filenames.
        def timestamp(path):
            import re
            m=re.search(r'-(\d+\.\d{6})\.5',path.name)
            if m:return float(m[1])
            raise ValueError(f'No filename time in {path}')
        files=sorted(files,key=timestamp);times=[timestamp(f) for f in files]
        if len(set(times))!=len(times):raise ValueError('Duplicate time')
        out=a.output/case.name;out.mkdir(exist_ok=True)
        first=read(files[0]);scale=np.ptp(first['x'][first['wall_loops']['xmj']])/4
        anchor=g.sequence_anchor([first],a.nx,.8*scale)
        create_charts(first,anchor,a.ny,a.nx,scale)  # Fix corner allocations from frame zero, even on resume.
        xmin,xmax=first['x'].min(),first['x'].max();ymin,ymax=first['y'].min(),first['y'].max()
        xx,yy=np.meshgrid(np.linspace(xmin,xmax,a.nx),np.linspace(ymin,ymax,4*a.ny))
        records=[]
        for i,origin in enumerate(files[:a.limit]):
            path=out/f'{i:04d}.h5'
            if i<a.start or (a.stop is not None and i>=a.stop):
                records.append(dict(index=i,time=times[i],file=f'{case.name}/{path.name}'));continue
            if path.exists():
                records.append(dict(index=i,time=times[i],file=f'{case.name}/{path.name}'));continue
            start=time.perf_counter();d=first if i==0 else read(origin);d['time']=times[i]
            charts=create_charts(d,anchor,a.ny,a.nx,scale)
            cart=dict(x=xx,y=yy,periodic=False,cartesian=True);cart.update(g.logical_metrics(xx,yy,False))
            audit=sample(d,charts);audit+=sample(d,{'cartesian':cart})
            write_frame(path,d,charts,cart,audit,origin,time.perf_counter()-start)
            records.append(dict(index=i,time=times[i],file=f'{case.name}/{path.name}'))
            print(json.dumps(dict(case=case.name,index=i,time=times[i],seconds=round(time.perf_counter()-start,2),audit=audit)),flush=True)
            d.pop('_source',None)
            for c in charts.values():
                c.pop('inverse_cache',None);c.pop('inverse_cache_query',None)
            del d,charts,cart
            gc.collect()  # Matplotlib Triangulation / TriFinder contain reference cycles.
        (out/'frames.json').write_text(json.dumps(records,indent=2),encoding='utf-8')


if __name__=='__main__':main()
