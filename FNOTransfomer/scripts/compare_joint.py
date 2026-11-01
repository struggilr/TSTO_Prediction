import json
from pathlib import Path
root=Path(__file__).resolve().parents[1];runs={}
for route in ['cartesian','taoca']:
    folder=root/'outputs'/('joint_'+route)
    run=json.loads((folder/'run.json').read_text());ev=json.loads((folder/'evaluation.json').read_text());hist=json.loads((folder/'history.json').read_text())
    assert not ev['smoke_only']
    runs[route]=dict(run=run,evaluation=ev,train_seconds=sum(r['train_seconds'] for r in hist))
assert runs['cartesian']['run']['parameters']==runs['taoca']['run']['parameters']
for key in ['history','horizon','batch_size','epochs']:
    assert runs['cartesian']['run']['arguments'][key]==runs['taoca']['run']['arguments'][key]
(root/'outputs/joint_comparison.json').write_text(json.dumps(runs,indent=2))
print('Saved outputs/joint_comparison.json; compare full-domain imputed errors and coverage together.')
