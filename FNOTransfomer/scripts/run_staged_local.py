"""Fresh controlled flow training with identical frozen motion surrogate."""
import os,json,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parents[1];os.chdir(root)
os.environ.update(MKL_THREADING_LAYER='GNU',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
status=root/'outputs/staged_training_status.json'
for route in ['cartesian','taoca']:
    status.write_text(json.dumps(dict(state='running',route=route,epochs=100,pid=os.getpid())))
    with (root/f'outputs/staged_{route}_local.log').open('w') as log:
        result=subprocess.run([sys.executable,'-u','train_joint.py','--route',route,'--device','cuda','--trajectory-model','outputs/shared_trajectory/model.json','--batch-size','2','--workers','0','--epochs','100'],stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:
        status.write_text(json.dumps(dict(state='failed',route=route,code=result.returncode)));sys.exit(result.returncode)
status.write_text(json.dumps(dict(state='completed',comparison='Use matched full-window evaluation; old compare_joint.py reads old joint runs.')))
