"""Inspectable per-chart legal donors and validation-selectable non-oracle fusion."""
import numpy as np
import shapely
from shapely.strtree import STRtree
from scipy.ndimage import distance_transform_edt
from .geometry import structured_triangles

def confidence_maps(masks,coordinates):
    result=[]
    for k,(mask,p) in enumerate(zip(masks,coordinates)):
        d=distance_transform_edt(np.tile(mask,(1,3)))[:,mask.shape[1]:2*mask.shape[1]] if k<3 else distance_transform_edt(mask)
        di=np.gradient(p,axis=0);dj=np.gradient(p,axis=1);edge=np.maximum((di*di).sum(-1),(dj*dj).sum(-1))
        result.append((d/(d+2))**2/np.maximum(edge,1e-5))
    return result

def build_maps(coordinates,masks,solid_polygons):
    yy,xx=np.meshgrid(np.linspace(-20,50,280),np.linspace(-10,100,440),indexing='ij');queries=np.stack([xx,yy],-1).reshape(-1,2)
    obstacle=shapely.union_all([shapely.Polygon(p) for p in solid_polygons]);shapely.prepare(obstacle);confidence=confidence_maps(masks,coordinates);maps=[]
    for k,(xy,mask) in enumerate(zip(coordinates,masks)):
        original=structured_triangles(*xy.shape[:2],k<3);p=xy.reshape(-1,2);pp=p[original]
        e=pp[:,1]-pp[:,0];f=pp[:,2]-pp[:,0];area=e[:,0]*f[:,1]-e[:,1]*f[:,0];oriented=np.isfinite(pp).all((1,2))&(area>1e-10)
        good=oriented&mask.ravel()[original].all(1);tri=original[good];pp=pp[good];area=area[good];polys=shapely.polygons(pp)
        ii=np.flatnonzero(shapely.intersects(obstacle,polys));accepted=np.ones(len(tri),bool)
        if len(ii):accepted[ii]=shapely.area(shapely.intersection(polys[ii],obstacle))<=1e-10
        tri=tri[accepted];pp=pp[accepted];area=area[accepted];polys=polys[accepted]
        stat=dict(chart=k,total=len(original),nonpositive=int((~oriented).sum()),invalid_vertex=int((oriented&~mask.ravel()[original].all(1)).sum()),solid_crossing=int((~accepted).sum()),accepted=len(tri))
        if not len(tri):maps.append(dict(q=np.array([],int),ids=np.zeros((0,3),int),bary=np.zeros((0,3)),quality=np.array([]),stats=stat));continue
        qi,ti=STRtree(polys).query(shapely.points(queries));a=pp[ti,0];e=pp[ti,1]-a;f=pp[ti,2]-a;q=queries[qi]-a
        u=(q[:,0]*f[:,1]-q[:,1]*f[:,0])/area[ti];v=(e[:,0]*q[:,1]-e[:,1]*q[:,0])/area[ti]
        contained=(u>=-1e-7)&(v>=-1e-7)&(u+v<=1+1e-7);qi=qi[contained];ti=ti[contained];u=u[contained];v=v[contained]
        if not len(qi):maps.append(dict(q=qi,ids=np.zeros((0,3),int),bary=np.zeros((0,3)),quality=np.array([]),stats=stat));continue
        order=np.lexsort((np.sum((pp[ti].mean(1)-queries[qi])**2,1),qi));sortedq=qi[order];take=order[np.r_[True,sortedq[1:]!=sortedq[:-1]]]
        qi=qi[take];ti=ti[take];bary=np.stack([1-u[take]-v[take],u[take],v[take]],1).clip(0,1);bary/=bary.sum(1,keepdims=True);ids=tri[ti]
        weight=np.einsum('nk,nk->n',confidence[k].ravel()[ids],bary).clip(1e-6)
        maps.append(dict(q=qi,ids=ids,bary=bary,quality=weight,stats=stat))
    return maps

def apply_maps(values,maps,extra_valid=None):
    channels=values[0].shape[0];out=np.full((len(maps),channels,280*440),np.nan,'f4');quality=np.zeros((len(maps),280*440),'f4')
    for k,(value,m) in enumerate(zip(values,maps)):
        keep=np.ones(len(m['q']),bool) if extra_valid is None else extra_valid[k].ravel()[m['ids']].all(1)
        q=m['q'][keep];ids=m['ids'][keep];bary=m['bary'][keep]
        out[k][:,q]=np.einsum('cnk,nk->cn',value.reshape(channels,-1)[:,ids],bary);quality[k,q]=m['quality'][keep]
    return out.reshape(len(maps),channels,280,440),quality.reshape(len(maps),280,440)

def fuse(per_chart,quality,mode='quality'):
    available=np.isfinite(per_chart).all(1);w=np.where(available,quality,0).astype('f8')
    if mode=='equal':w=available.astype('f8')
    elif mode=='background_x4':w[2]*=4
    elif mode=='background_x16':w[2]*=16
    elif mode=='finest':
        chosen=w.argmax(0);w=np.stack([(chosen==k)&available[k] for k in range(len(w))]).astype('f8')
    elif mode!='quality':raise ValueError(mode)
    den=w.sum(0);support=den>0
    result=(np.nan_to_num(per_chart)*w[:,None]).sum(0)/np.maximum(den,1e-12)[None];result[:,~support]=np.nan
    return result.astype('f4'),support
