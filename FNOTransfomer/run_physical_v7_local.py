"""Validation-fixed next experiment, queued behind V6 evaluation; preserves every result."""
import os,sys,json,time,subprocess,datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'outputs/physical_v7'
def status(state,**kw):
    p=OUT/'pipeline_status.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(dict(state=state,updated=datetime.datetime.now().isoformat(),pid=os.getpid(),**kw),indent=2));tmp.replace(p)
def run(script,log,*args):
    with (OUT/log).open('a',encoding='utf-8') as f:subprocess.run([sys.executable,'-u',str(ROOT/script),*args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
def main():
    os.environ['MKL_THREADING_LAYER']='GNU';OUT.mkdir(exist_ok=True)
    try:
        status('waiting_for_v6_evaluation')
        comparison=ROOT/'outputs/direct_history_v6/comparison.json'
        while not comparison.exists():
            state=json.loads((ROOT/'outputs/direct_history_v6/pipeline_status.json').read_text())
            if state['state']=='failed':raise RuntimeError('V6 failed; diagnose it before starting another experiment')
            time.sleep(20)
        old=json.loads(comparison.read_text());a=old['regions']['all']['nrmse']['FNO_Cartesian'];b=old['regions']['all']['nrmse']['FNO_TAOCA_V6'];sa=old['regions']['shock']['nrmse']['FNO_Cartesian'][:2];sb=old['regions']['shock']['nrmse']['FNO_TAOCA_V6'][:2]
        if all(x<y for x,y in zip(b,a)) and all(x<y for x,y in zip(sb,sa)):status('not_needed_v6_met_accuracy_criterion');return
        status('preparing_geometry_operators');run('scripts/prepare_physical_operators_v7.py','prepare.log')
        status('testing');run('scripts/test_physical_operator_v7.py','tests.log');run('train_physical_v7.py','smoke.log','--smoke')
        assert json.loads((OUT/'taoca_smoke/status.json').read_text())['state']=='smoke_passed'
        status('training',epochs=60);run('train_physical_v7.py','train.log','--resume')
        status('comparing');run('scripts/compare_physical_v7.py','comparison.log')
        status('plotting');run('scripts/analyze_physical_v7.py','analysis.log')
        result=json.loads((OUT/'comparison.json').read_text());status('complete',all_five_global_channels_better=result['all_five_global_channels_better'],pressure_density_shock_better=result['pressure_density_shock_better'],report='analysis/分析报告.md')
    except Exception as e:status('failed',error=repr(e));raise
if __name__=='__main__':main()
