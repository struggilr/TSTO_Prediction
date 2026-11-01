"""Differentiable exact legal inverse/fusion on a sparse physical-grid stencil."""
from pathlib import Path
import numpy as np
import torch

def pack_maps(maps,stride=4,phase=(0,0)):
    # A label-independent regular stencil; never choose points by test errors.
    selected=[]
    for m in maps:
        keep=(m['q']//440%stride==phase[0])&(m['q']%440%stride==phase[1]);selected.append({k:np.asarray(m[k])[keep] for k in ['q','ids','bary','quality']})
    allq=np.unique(np.concatenate([m['q'] for m in selected]));den=np.zeros(len(allq),'f8');result={'q':allq.astype('i4')}
    for m in selected:np.add.at(den,np.searchsorted(allq,m['q']),m['quality'])
    for k,m in enumerate(selected):
        index=np.searchsorted(allq,m['q']);weight=m['bary']*(m['quality']/np.maximum(den[index],1e-20))[:,None]
        result[f'index{k}']=index.astype('i4');result[f'ids{k}']=m['ids'].astype('i4');result[f'weight{k}']=weight.astype('f4')
    return result

def project(pred,operator):
    """pred [K,5,Hmax,W], same donor coefficients as V3 evaluation."""
    result=pred.new_zeros((5,len(operator['q'])))
    for k in range(4):
        ids=torch.as_tensor(operator[f'ids{k}'],device=pred.device,dtype=torch.long);index=torch.as_tensor(operator[f'index{k}'],device=pred.device,dtype=torch.long);weight=torch.as_tensor(operator[f'weight{k}'],device=pred.device,dtype=pred.dtype)
        value=(pred[k].reshape(5,-1)[:,ids]*weight).sum(-1)
        result=result.index_add(1,index,value)
    return result

def load_window(folder,case,start):
    with np.load(Path(folder)/f'{case}_s{start:03d}.npz') as f:return [{k.split('_',1)[1]:f[k] for k in f.files if k.startswith(f't{t}_')} for t in range(10)]

def physical_loss(out,batch,operators,near,shock):
    total=out['fields'].sum()*0;parts={k:0. for k in ['global','shock','wall']};sse=np.zeros(5);counts=0
    for t,op in enumerate(operators):
        q=torch.as_tensor(op['q'],device=out['fields'].device,dtype=torch.long);pred=project(out['fields'][0,t],op);target=batch['target'][0,t].reshape(5,-1)[:,q];valid=batch['target_valid'][0,t].reshape(-1)[q]>.5
        err=(pred-target).square();edge=shock[0,t,0].reshape(-1)[q]>.5;wall=near[0,t,0].reshape(-1)[q]>.5
        glob=err[:,valid].mean();eg=err[:2,valid&edge].mean() if (valid&edge).any() else glob*0;wa=err[:,valid&wall].mean() if (valid&wall).any() else glob*0
        total+=glob+.5*eg+.2*wa
        parts['global']+=float(glob.detach())/10;parts['shock']+=float(eg.detach())/10;parts['wall']+=float(wa.detach())/10
    return total/10,parts
