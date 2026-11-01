"""Coherent five-channel donor weights; no reference fields enter fusion."""
import numpy as np
from scipy.ndimage import gaussian_filter
from .chart_transfer_v3 import fuse

MODES=['quality','quality_squared','quality_fourth','finest','edge_coherent_2','edge_coherent_4']

def weights_and_fuse(charts,quality,mode):
    available=np.isfinite(charts).all(1)
    if mode in ['quality','finest']:return fuse(charts,quality,mode)
    q=np.where(available,quality,0).astype('f8')
    # Relative scaling prevents overflow and preserves the geometric ordering.
    q/=np.maximum(q.max(0,keepdims=True),1e-20)
    if mode.startswith('quality_'):
        q=q**(2 if mode=='quality_squared' else 4)
    elif mode.startswith('edge_coherent_'):
        edge=[]
        for k in range(len(charts)):
            safe=available[k].copy()
            safe[1:]&=available[k,:-1];safe[:-1]&=available[k,1:]
            safe[:,1:]&=available[k,:,:-1];safe[:,:-1]&=available[k,:,1:]
            magnitude=np.zeros(safe.shape)
            for c in [0,1]:
                gy,gx=np.gradient(np.nan_to_num(charts[k,c]).astype('f8'))
                magnitude+=np.hypot(gx,gy)
            edge.append(np.where(safe,magnitude,0))
        edge=np.array(edge);maximum=edge.max(0)
        # Only predicted normalized p/rho are inspected. No truth gradient.
        score=q*(1+2*edge/np.maximum(maximum,1e-8)[None])
        winner=score.argmax(0)
        selected=np.stack([gaussian_filter((winner==k).astype('f8'),.65) for k in range(len(q))])
        strength=int(mode.rsplit('_',1)[1]);gate=np.clip(maximum/.3,0,1)
        q=q*(1+strength*selected*gate[None])
    else:raise ValueError(mode)
    q=np.where(available,q,0);den=q.sum(0);support=den>0
    result=(np.nan_to_num(charts)*q[:,None]).sum(0)/np.maximum(den,1e-20)[None]
    result[:,~support]=np.nan
    return result.astype('f4'),support
