"""Complete donor search with explicit periodic chart connectivity."""
import numpy as np
import shapely
from shapely.strtree import STRtree
from .geometry import structured_triangles

def complete_inverse(values,coordinates,periodic=False):
    yy,xx=np.meshgrid(np.linspace(-20,50,280),np.linspace(-10,100,440),indexing='ij');query=np.c_[xx.ravel(),yy.ravel()]
    total=np.zeros((values.shape[1],len(query)));den=np.zeros(len(query));stats=[]
    for k,(chart,xy) in enumerate(zip(values,coordinates)):
        triangles=structured_triangles(*xy.shape[:2],periodic and k<3);p=xy.reshape(-1,2);pp=p[triangles]
        e=pp[:,1]-pp[:,0];f=pp[:,2]-pp[:,0];area=e[:,0]*f[:,1]-e[:,1]*f[:,0];good=area>1e-10
        tri=triangles[good];pp=pp[good];area=area[good]
        polygons=shapely.polygons(pp);tree=STRtree(polygons)
        qi,ti=tree.query(shapely.points(query))
        a=pp[ti,0];e=pp[ti,1]-a;f=pp[ti,2]-a;q=query[qi]-a
        u=(q[:,0]*f[:,1]-q[:,1]*f[:,0])/area[ti];v=(e[:,0]*q[:,1]-e[:,1]*q[:,0])/area[ti]
        inside=(u>=-1e-7)&(v>=-1e-7)&(u+v<=1+1e-7)
        qi=qi[inside];ti=ti[inside];u=u[inside];v=v[inside]
        distance=np.sum((pp[ti].mean(1)-query[qi])**2,1);order=np.lexsort((distance,qi));qsorted=qi[order]
        first=np.r_[True,qsorted[1:]!=qsorted[:-1]];take=order[first];qi=qi[take];ti=ti[take]
        bary=np.stack([1-u[take]-v[take],u[take],v[take]],1)
        total[:,qi]+=np.einsum('cnk,nk->cn',chart.reshape(chart.shape[0],-1)[:,tri[ti]],bary);den[qi]+=1
        stats.append(dict(chart=k,covered=len(qi),nonpositive=int((~good).sum()),triangles=len(triangles),min_signed_area=float(area.min())))
    support=den>0;total[:,support]/=den[support]
    return total.reshape(-1,280,440).astype('f4'),support.reshape(280,440),stats

