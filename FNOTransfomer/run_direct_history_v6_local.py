"""Prepare -> contracts -> real optimizer smoke -> 100 epochs -> evaluate -> figures."""
import os,sys,json,time,subprocess,datetime,argparse
from pathlib import Path
import psutil
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'outputs/direct_history_v6'
def status(state,**kw):
    p=OUT/'pipeline_status.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(dict(state=state,updated=datetime.datetime.now().isoformat(),pid=os.getpid(),**kw),indent=2));tmp.replace(p)
def run(script,log,*args):
    with (OUT/log).open('a',encoding='utf-8') as f:subprocess.run([sys.executable,'-u',str(ROOT/script),*args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare-pid',type=int);a=p.parse_args();OUT.mkdir(exist_ok=True);os.environ['MKL_THREADING_LAYER']='GNU'
    try:
        status('preparing',epochs=100)
        if a.prepare_pid:
            while psutil.pid_exists(a.prepare_pid):time.sleep(10)
            if not (ROOT/'datasets/taoca_direct_history_v6/manifest.json').exists():raise RuntimeError('Existing preparation failed; see prepare_error.log')
        else:run('scripts/prepare_direct_history_v6.py','prepare.log')
        status('checking_contracts');run('scripts/test_direct_history_v6.py','implementation_tests.log')
        status('smoke_training');run('train_direct_history_v6.py','smoke.log','--smoke')
        smoke=json.loads((OUT/'taoca_smoke/evaluation.json').read_text());assert smoke['smoke_only'] and smoke['forecast_instances']==10
        status('training',epochs=100);run('train_direct_history_v6.py','train.log','--resume')
        status('comparing');run('scripts/compare_direct_history_v6.py','comparison.log')
        status('plotting');run('scripts/analyze_direct_history_v6.py','analysis.log')
        status('complete',report='analysis/分析报告.md')
    except Exception as e:status('failed',error=repr(e));raise
if __name__=='__main__':main()
