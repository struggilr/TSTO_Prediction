"""Verify frozen code, operator bytes and full training state before resuming."""
import os
os.environ['MKL_THREADING_LAYER']='GNU'
import json,hashlib,datetime
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/physical_v7'
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda:f.read(8*1024*1024),b''):h.update(data)
    return h.hexdigest()
def main():
    run=json.loads((OUT/'taoca/run.json').read_text())
    for name,wanted in run['source_sha256'].items():assert sha(ROOT/name)==wanted,name
    assert sha(ROOT/'outputs/physics_v2/cartesian/best.pt')==run['cartesian_checkpoint_sha256']
    assert sha(ROOT/'outputs/atlas_v3/taoca/best.pt')==run['initialization_sha256']
    audit=json.loads((OUT/'operator_full_audit.json').read_text());assert audit['verified'] and audit['windows']==211
    for name,wanted in audit['operator_sha256'].items():assert sha(OUT/'operators'/name)==wanted,name
    saved=torch.load(OUT/'taoca/last.pt',map_location='cpu',weights_only=False);history=json.loads((OUT/'taoca/history.json').read_text())
    assert saved['epoch']==len(saved['history'])==len(history)
    assert saved['history']==history and saved['best']==min(r['val']['score'] for r in history)
    assert all(torch.isfinite(t).all() for t in saved['model'].values())
    for field in ['optimizer','scheduler','rng','cuda_rng','python_rng','numpy_rng']:assert field in saved
    assert saved['scheduler']['last_epoch']==saved['epoch']
    report=dict(verified=True,checked_at=datetime.datetime.now().isoformat(),epoch=saved['epoch'],remaining_epochs=run['arguments']['epochs']-saved['epoch'],all_frozen_sources_verified=True,operator_files_verified=len(audit['operator_sha256']),checkpoint_sha256=sha(OUT/'taoca/last.pt'),optimizer_scheduler_and_rng_present=True,cartesian_checkpoint_unchanged=True,initialization_checkpoint_unchanged=True)
    (OUT/'resume_verified_20261020.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
if __name__=='__main__':main()
