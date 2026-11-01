"""Verify V9 interrupted state, then resume without changing frozen experiment files."""
import os
os.environ['MKL_THREADING_LAYER']='GNU'
import json, hashlib, datetime
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs/local_v9'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
    return h.hexdigest()
def verify():
    run=json.loads((OUT/'taoca/run.json').read_text())
    for name,wanted in run['source_sha256'].items(): assert sha(ROOT/name)==wanted,name
    for path,key in [('outputs/physics_v2/cartesian/best.pt','cartesian_checkpoint_sha256'),('outputs/metric_v8/taoca/best.pt','initialization_sha256')]: assert sha(ROOT/path)==run[key],path
    base=ROOT/'outputs/physical_v7'
    audit=json.loads((base/'operator_full_audit.json').read_text());assert audit['verified'] and audit['windows']==211
    for name,wanted in audit['operator_sha256'].items(): assert sha(base/'operators'/name)==wanted,name
    saved=torch.load(OUT/'taoca/last.pt',map_location='cpu',weights_only=False)
    history=json.loads((OUT/'taoca/history.json').read_text())
    assert saved['epoch']==len(history)==len(saved['history']) and saved['history']==history
    assert saved['best']==min(r['val']['score'] for r in history)
    assert all(torch.isfinite(v).all() for v in saved['model'].values())
    for key in ['optimizer','scheduler','rng','cuda_rng','python_rng','numpy_rng']: assert key in saved
    assert saved['scheduler']['last_epoch']==saved['epoch']
    assert torch.cuda.is_available()
    report=dict(verified=True,root=str(ROOT),epoch=saved['epoch'],remaining_epochs=run['arguments']['epochs']-saved['epoch'],operators_verified=211,checkpoint_sha256=sha(OUT/'taoca/last.pt'),gpu=torch.cuda.get_device_name(0),checked_at=datetime.datetime.now().isoformat())
    (OUT/'migration_h_verified.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
if __name__=='__main__':
    verify()
    if '--verify-only' not in __import__('sys').argv:
        from run_local_v9_h import status,run
        try:
            status('training',epochs=60,migrated_root=str(ROOT));run('train_local_v9.py','train.log','--resume')
            status('comparing');run('scripts/compare_local_v9.py','comparison.log')
            status('plotting');run('scripts/analyze_local_v9.py','analysis.log')
            final=json.loads((OUT/'comparison.json').read_text());status('complete',all_five_global_channels_better=final['all_five_global_channels_better'],pressure_density_shock_better=final['pressure_density_shock_better'])
        except Exception as e: status('failed',error=repr(e));raise

