"""Continue existing checkpoint after verification; never recreate immutable inputs."""
import os,json
from run_physical_v7_local import ROOT,OUT,status,run
def main():
    os.environ['MKL_THREADING_LAYER']='GNU'
    try:
        status('checking_resume');run('scripts/check_resume_physical_v7.py','resume_verification.log')
        verified=json.loads((OUT/'resume_verified_20261020.json').read_text());assert verified['verified']
        status('training',epochs=60,resume_from=verified['epoch']);run('train_physical_v7.py','train.log','--resume')
        status('comparing');run('scripts/compare_physical_v7.py','comparison.log')
        status('plotting');run('scripts/analyze_physical_v7.py','analysis.log')
        final=json.loads((OUT/'comparison.json').read_text());status('complete',all_five_global_channels_better=final['all_five_global_channels_better'],pressure_density_shock_better=final['pressure_density_shock_better'],report='analysis/分析报告.md')
    except Exception as e:status('failed',error=repr(e));raise
if __name__=='__main__':main()
