"""Launch B-only training only after representation and implementation gates pass."""
import os,sys,json,subprocess,datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'outputs/atlas_v3'
def status(**data):
    path=OUT/'pipeline_status.json';tmp=path.with_suffix('.partial.json');tmp.write_text(json.dumps(dict(updated=datetime.datetime.now().isoformat(),**data),indent=2));tmp.replace(path)
def main():
    os.environ['MKL_THREADING_LAYER']='GNU';selection=json.loads((OUT/'allocation_selection.json').read_text());tests=json.loads((OUT/'implementation_tests.json').read_text())
    if not selection['training_authorized_by_gate'] or not tests['verified']:raise RuntimeError('Gates not passed')
    if tests.get('tested_heights')!=selection['selected']['heights']:raise RuntimeError('Selected allocation was not functionally tested')
    try:
        status(state='training',route='taoca',epochs=100,heights=selection['selected']['heights'],pid=os.getpid())
        with (OUT/'taoca_train.log').open('a',encoding='utf8') as log:subprocess.run([sys.executable,'-u',str(ROOT/'train_atlas_v3.py'),'--resume'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        status(state='comparing')
        with (OUT/'comparison.log').open('w',encoding='utf8') as log:subprocess.run([sys.executable,str(ROOT/'scripts/compare_atlas_v3.py')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        status(state='complete',comparison='comparison.json')
    except Exception as e:status(state='failed',error=repr(e));raise
if __name__=='__main__':main()
