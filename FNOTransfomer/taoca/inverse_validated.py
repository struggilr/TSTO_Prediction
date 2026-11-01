"""Mask- and obstacle-aware complete inverse; missing support is never filled."""
import numpy as np
import shapely
from shapely.strtree import STRtree
from .geometry import structured_triangles

def validated_inverse(values,coordinates,valid,solid_polygons=None,weights=None,query=None):
    if query is None:
        yy,xx=np.meshgrid(np.linspace(-20,50,280),np.linspace(-10,100,440),indexing='ij');query=np.stack([xx,yy],-1)
    shape=query.shape[:-1];qflat=query.reshape(-1,2)
    total=np.zeros((values.shape[1],len(qflat)));den=np.zeros(len(qflat));stats=[]
    obstacle=shapely.union_all([shapely.Polygon(p) for p in solid_polygons]) if solid_polygons is not None else None
    for k,(chart,xy,mask) in enumerate(zip(values,coordinates,valid)):
        triangles=structured_triangles(*xy.shape[:2],k<3);p=xy.reshape(-1,2);pp=p[triangles]
        e=pp[:,1]-pp[:,0];f=pp[:,2]-pp[:,0];area=e[:,0]*f[:,1]-e[:,1]*f[:,0]
        oriented=np.isfinite(pp).all((1,2))&(area>1e-10)
        keep=oriented&mask.ravel()[triangles].all(1)
        tri=triangles[keep];pp=pp[keep];area=area[keep]
        polygons=shapely.polygons(pp)
        crossed=0
        if obstacle is not None and len(tri):
            ii=np.flatnonzero(shapely.intersects(polygons,obstacle));ok=np.ones(len(tri),bool)
            if len(ii):ok[ii]=shapely.area(shapely.intersection(polygons[ii],obstacle))<=1e-10
            crossed=int((~ok).sum());tri=tri[ok];pp=pp[ok];area=area[ok];polygons=polygons[ok]
        stats.append(dict(chart=k,total=len(triangles),nonpositive=int((~oriented).sum()),invalid_vertex=int((oriented&~mask.ravel()[triangles].all(1)).sum()),solid_crossing=crossed,accepted=len(tri)))
        if not len(tri):continue
        qi,ti=STRtree(polygons).query(shapely.points(qflat))
        a=pp[ti,0];e=pp[ti,1]-a;f=pp[ti,2]-a;q=qflat[qi]-a
        u=(q[:,0]*f[:,1]-q[:,1]*f[:,0])/area[ti];v=(e[:,0]*q[:,1]-e[:,1]*q[:,0])/area[ti]
        inside=(u>=-1e-7)&(v>=-1e-7)&(u+v<=1+1e-7);qi=qi[inside];ti=ti[inside];u=u[inside];v=v[inside]
        if not len(qi):continue
        order=np.lexsort((np.sum((pp[ti].mean(1)-qflat[qi])**2,1),qi));qs=qi[order];take=order[np.r_[True,qs[1:]!=qs[:-1]]]
        qi=qi[take];ti=ti[take];bary=np.stack([1-u[take]-v[take],u[take],v[take]],1).clip(0,1);bary/=bary.sum(1,keepdims=True)
        value=np.einsum('cnk,nk->cn',chart.reshape(chart.shape[0],-1)[:,tri[ti]],bary)
        if weights is None:weight=np.ones(len(qi))
        else:weight=np.einsum('nk,nk->n',weights[k].ravel()[tri[ti]],bary).clip(1e-6)
        total[:,qi]+=value*weight;den[qi]+=weight
    support=den>0;total[:,support]/=den[support]
    total[:,~support]=np.nan
    return total.reshape((values.shape[1],)+shape).astype('f4'),support.reshape(shape),stats
