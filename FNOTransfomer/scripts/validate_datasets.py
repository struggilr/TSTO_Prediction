from pathlib import Path
import json,numpy as np
ROOT=Path(__file__).resolve().parents[1]; out=ROOT/'datasets';m=json.loads((out/'dataset_manifest.json').read_text(encoding='utf-8'));errors=[]
for r in m['records']:
 for sub in ('source','taoca'):
  p=out/sub/f"{r['time_label']}.npz"
  if not p.exists():errors.append(f'missing {p}');continue
  with np.load(p) as a:
   if any(not np.all(np.isfinite(a[k])) for k in a.files if a[k].dtype.kind=='f'):errors.append(f'nonfinite {p}')
   if sub=='source' and a['pressure'].shape != (64,96):errors.append(f'source shape {p}')
   if sub=='taoca' and not all(k in a for k in ('smj_valid_donor_cells','xmj_valid_donor_cells','outer_valid_donor_cells','gap_valid_donor_cells')):errors.append(f'mask {p}')
report={'ok':not errors,'errors':errors,'frames':len(m['records']),'source_nodes':[r['source_nodes'] for r in m['records']],'taoca_shapes':m['records'][0]['taoca_shapes']}
(out/'validation_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(report,ensure_ascii=False));raise SystemExit(1 if errors else 0)
