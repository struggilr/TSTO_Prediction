"""Training-only ideal-gas compatibility audit; no test labels used."""
import json
from pathlib import Path
import numpy as np
import h5py
ROOT=Path(__file__).resolve().parents[1]
def main():
    meta=json.loads((ROOT/'datasets/joint_nearest/manifest.json').read_text());xs=[];ys=[]
    for case,r in meta['cases'].items():
        n=round(.7*len(r['times']))
        with h5py.File(ROOT/'datasets/joint_nearest'/(case+'.h5')) as f:
            for t in np.linspace(0,n-1,8).astype(int):
                values=f['fields'][t].reshape(5,-1);mask=f['valid'][t].ravel().astype(bool)
                ids=np.flatnonzero(mask)[::max(1,int(mask.sum())//3000)]
                p,rho,temp=values[:3,ids];good=(rho>0)&(temp>0)&np.isfinite(p)
                xs.append((rho*temp)[good]);ys.append(p[good])
    x=np.concatenate(xs).astype('f8');y=np.concatenate(ys).astype('f8');design=np.c_[x,np.ones(len(x))]
    coef=np.linalg.lstsq(design,y,rcond=None)[0]
    for _ in range(4):
        residual=y-design@coef;cut=max(1.,np.quantile(abs(residual),.90));weight=np.minimum(1,cut/np.maximum(abs(residual),1e-8));coef=np.linalg.lstsq(design*np.sqrt(weight[:,None]),y*np.sqrt(weight),rcond=None)[0]
    residual=y-design@coef;nrmse=float(np.sqrt(np.mean(residual**2))/meta['std'][0]);enabled=bool(0<coef[0]<2000 and nrmse<.10)
    result=dict(gas_constant=float(coef[0]),pressure_offset=float(coef[1]),fit='p = R*rho*T + offset',nrmse_training_pressure_std=nrmse,enabled=enabled,samples=len(x),scope='Only chronological training frames; compatibility fit, not independent identification of fluid composition or reference pressure.')
    out=ROOT/'outputs/physics_v2';out.mkdir(parents=True,exist_ok=True);(out/'thermodynamics.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
