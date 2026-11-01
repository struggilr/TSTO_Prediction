"""Run one prespecified local-capacity trial after completed validation diagnostics."""
import os,sys,json,subprocess,datetime,traceback,hashlib
from pathlib import Path
os.environ['MKL_THREADING_LAYER']='GNU'
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'outputs/local_v9'
def status(state,**kw):
    OUT.mkdir(exist_ok=True);p=OUT/'pipeline_status.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(dict(state=state,pid=os.getpid(),updated=datetime.datetime.now().isoformat(),**kw),indent=2));tmp.replace(p)
def run(script,log,*args):
    with (OUT/log).open('a',encoding='utf-8') as f:subprocess.run([sys.executable,'-u',str(ROOT/script),*args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
def main():
    d=ROOT/'outputs/shock_diagnosis_v9'
    assert json.loads((d/'data_consistency_audit.json').read_text())['verified']
    assert json.loads((d/'local_real_test.json').read_text())['verified']
    assert json.loads((d/'local_summary_status.json').read_text())['state']=='complete'
    status('smoke_training');run('train_local_v9.py','smoke.log','--smoke')
    assert json.loads((OUT/'taoca_smoke/status.json').read_text())['state']=='smoke_passed'
    status('training',epochs=60);run('train_local_v9.py','train.log','--resume')
    status('comparing');run('scripts/compare_local_v9.py','comparison.log')
    status('plotting');run('scripts/analyze_local_v9.py','analysis.log')
    result=json.loads((OUT/'comparison.json').read_text());status('complete',all_five_global_channels_better=result['all_five_global_channels_better'],pressure_density_shock_better=result['pressure_density_shock_better'])
if __name__=='__main__':
    try:main()
    except Exception:status('failed',error=traceback.format_exc());raise
