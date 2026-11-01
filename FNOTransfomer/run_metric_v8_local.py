"""Queue the prespecified metric trial behind V7, without overlapping GPU training."""
import os,sys,time,json,subprocess,datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'outputs/metric_v8'
def status(state,**kw):
    p=OUT/'pipeline_status.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(dict(state=state,updated=datetime.datetime.now().isoformat(),pid=os.getpid(),**kw),indent=2));tmp.replace(p)
def run(script,log,*args):
    with (OUT/log).open('a',encoding='utf-8') as f:subprocess.run([sys.executable,'-u',str(ROOT/script),*args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
def main():
    os.environ['MKL_THREADING_LAYER']='GNU';OUT.mkdir(exist_ok=True)
    try:
        status('waiting_for_v7_evaluation')
        result=ROOT/'outputs/physical_v7/comparison.json'
        while not result.exists():
            s=json.loads((ROOT/'outputs/physical_v7/pipeline_status.json').read_text())
            if s['state']=='failed':raise RuntimeError('V7 failed; diagnose before starting another trial')
            time.sleep(20)
        scores=json.loads(result.read_text())
        if scores['all_five_global_channels_better'] and scores['pressure_density_shock_better']:status('not_needed_v7_met_accuracy_criterion');return
        tests=json.loads((ROOT/'outputs/diagnosis_v8/metric_candidate_tests.json').read_text());assert tests['verified'] and tests['additional_parameters']==816
        status('smoke_training');run('train_metric_v8.py','smoke.log','--smoke')
        assert json.loads((OUT/'taoca_smoke/status.json').read_text())['state']=='smoke_passed'
        status('training',epochs=60);run('train_metric_v8.py','train.log','--resume')
        status('comparing');run('scripts/compare_metric_v8.py','comparison.log')
        status('plotting');run('scripts/analyze_metric_v8.py','analysis.log')
        final=json.loads((OUT/'comparison.json').read_text());status('complete',all_five_global_channels_better=final['all_five_global_channels_better'],pressure_density_shock_better=final['pressure_density_shock_better'])
    except Exception as e:status('failed',error=repr(e));raise
if __name__=='__main__':main()
