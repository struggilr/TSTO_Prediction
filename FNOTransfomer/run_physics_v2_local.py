"""Sequential local-GPU training; persistent logs, resumable checkpoints, then comparison."""
import os,json,sys,subprocess,datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'outputs/physics_v2'
def status(**kw):
    tmp=OUT/'pipeline_status.partial.json';tmp.write_text(json.dumps(dict(updated=datetime.datetime.now().isoformat(),**kw),indent=2));tmp.replace(OUT/'pipeline_status.json')
def main():
    os.environ['MKL_THREADING_LAYER']='GNU';OUT.mkdir(exist_ok=True,parents=True)
    verified=json.loads((OUT/'verification.json').read_text())
    if not all(verified[k] for k in ['invalid_donor_and_solid_rejection','physical_gradient_affine','cartesian_no_future_label_leakage','taoca_no_future_label_leakage']):raise RuntimeError('Verification gates failed')
    if not json.loads((OUT/'geometry_preflight.json').read_text())['verified']:raise RuntimeError('Geometry preflight failed')
    try:
        for route in ['cartesian','taoca']:
            status(state='training',active_route=route,epochs_per_route=100,pid=os.getpid())
            with (OUT/(route+'_train.log')).open('a',encoding='utf8') as log:
                subprocess.run([sys.executable,'-u',str(ROOT/'train_physics_v2.py'),'--route',route,'--epochs','100','--batch-size','1','--resume'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        status(state='comparing',pid=os.getpid())
        with (OUT/'comparison.log').open('w',encoding='utf8') as log:subprocess.run([sys.executable,str(ROOT/'scripts/compare_physics_v2.py')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        status(state='complete',routes=['cartesian','taoca'],comparison='comparison.json')
    except Exception as e:status(state='failed',error=repr(e));raise
if __name__=='__main__':main()
