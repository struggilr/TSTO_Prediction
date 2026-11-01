"""Resume after verified audit-only manifest repair, without regenerating it."""
import os,json
from run_direct_history_v6_local import ROOT,OUT,status,run
def main():
    os.environ['MKL_THREADING_LAYER']='GNU'
    assert json.loads((OUT/'resume_audit.json').read_text())['verified']
    try:
        status('training',epochs=100,resume_from=70)
        run('train_direct_history_v6.py','train.log','--resume')
        status('comparing');run('scripts/compare_direct_history_v6.py','comparison.log')
        status('plotting');run('scripts/analyze_direct_history_v6.py','analysis.log')
        status('complete',report='analysis/分析报告.md')
    except Exception as e:status('failed',error=repr(e));raise
if __name__=='__main__':main()
