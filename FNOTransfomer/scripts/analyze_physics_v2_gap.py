import os,json,sys
os.environ['MKL_THREADING_LAYER']='GNU'
from pathlib import Path
import numpy as np
import h5py
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from taoca.physics_losses import regional_maps
def main():
    torch.set_num_threads(4);root=ROOT/'outputs/physics_v2';groups={}
    for path in sorted((root/'cartesian/predictions').glob('*.h5')):
        with h5py.File(path) as a,h5py.File(root/'taoca/predictions'/path.name) as b:
            ref=a['reference'][:];x=a['predicted'][:];y=b['predicted'][:];valid=a['valid'][:].astype(bool)
            common=valid&a['support'][:]&b['support'][:]&np.isfinite(x).all(1)&np.isfinite(y).all(1)
            near,shock=regional_maps(torch.from_numpy(ref)[None],torch.from_numpy(valid.astype('f4'))[None]);near=near[0,:,0].numpy()>.5;shock=shock[0,:,0].numpy()>.5
            for t in range(10):
                for key,mask in [('all',common[t]),('case/'+str(a.attrs['case']),common[t]),('lead/'+str(t+1),common[t]),('wall',common[t]&near[t]),('shock',common[t]&shock[t]),('other',common[t]&~(near[t]|shock[t]))]:
                    g=groups.setdefault(key,dict(n=0,A=np.zeros(5),B=np.zeros(5)));g['n']+=int(mask.sum());g['A']+=((x[t][:,mask]-ref[t][:,mask])**2).sum(1);g['B']+=((y[t][:,mask]-ref[t][:,mask])**2).sum(1)
    output={}
    for key,g in groups.items():
        a=np.sqrt(g['A']/max(1,g['n']));b=np.sqrt(g['B']/max(1,g['n']));output[key]=dict(n=g['n'],cartesian=a.tolist(),taoca=b.tolist(),taoca_error_change_percent=(100*(b/a-1)).tolist())
    output['training']={}
    for route in ['cartesian','taoca']:
        h=json.loads((root/route/'history.json').read_text());best=min(h,key=lambda x:x['val']['loss']);output['training'][route]=dict(best_epoch=best['epoch'],best=best,last=h[-1])
    (root/'gap_analysis.json').write_text(json.dumps(output,indent=2));print(json.dumps({k:v for k,v in output.items() if k in ['all','wall','shock','other']},indent=2));print('best epochs',{k:v['best_epoch'] for k,v in output['training'].items()})
if __name__=='__main__':main()
