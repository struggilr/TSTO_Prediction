from pathlib import Path
import h5py,json,numpy as np
ROOT=Path(__file__).resolve().parents[1];base=ROOT/'datasets'/'fenli_v2';manifest={}
for split in ('train','test'):
 p=base/f'{split}.h5';stats={}
 with h5py.File(p,'r') as f:
  cases={k:{'frames':int(g.attrs['num_frames']),'shape':list(g['t0'].shape)} for k,g in f.items() if k.startswith('Ma_')}
  stats={'mean':f['norm_stats/mean'][:].tolist(),'std':f['norm_stats/std'][:].tolist()}
 manifest[split]={'cases':cases,'stats':stats}
(base/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8');print(json.dumps(manifest,indent=2))
