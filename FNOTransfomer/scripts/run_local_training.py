import json,os,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parents[1]
os.chdir(root)
os.environ.update(MKL_THREADING_LAYER='GNU',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
status=root/'outputs/local_training_status.json'
for route in ['cartesian','taoca']:
    status.write_text(json.dumps(dict(state='running',route=route,epochs=100,batch_size=2,pid=os.getpid())))
    with (root/f'outputs/joint_{route}_local.log').open('w') as log:
        result=subprocess.run([sys.executable,'-u','train_joint.py','--route',route,'--device','cuda','--batch-size','2','--workers','0','--epochs','100'],stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:
        status.write_text(json.dumps(dict(state='failed',route=route,code=result.returncode)))
        sys.exit(result.returncode)
result=subprocess.run([sys.executable,'scripts/compare_joint.py'])
status.write_text(json.dumps(dict(state='completed' if result.returncode==0 else 'comparison_failed')))
