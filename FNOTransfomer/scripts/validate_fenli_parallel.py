from pathlib import Path
import h5py,json
ROOT=Path(__file__).resolve().parents[1];base=ROOT/'datasets'/'fenli_v2_parallel';manifest={}
for case in sorted(base.iterdir()):
 if not case.is_dir():continue
 rec={}
 for split in ('train','test'):
  with h5py.File(case/f'{split}.h5','r') as f:
   groups={k:{'frames':int(g.attrs['num_frames']),'shape':list(g['t0'].shape)} for k,g in f.items() if k.startswith('Ma_') and 't0' in g}
   rec[split]=groups
 manifest[case.name]=rec
(base/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(manifest,ensure_ascii=False,indent=2))

