"""Resumable process-parallel paired dataset preparation (no training)."""
import argparse, concurrent.futures, os, subprocess, sys, json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',default='E:/fenli');p.add_argument('--workers',type=int,default=3);a=p.parse_args()
    logs=ROOT/'outputs/preprocessing';logs.mkdir(parents=True,exist_ok=True)
    jobs=[(c.name,start,start+20) for c in sorted(Path(a.source).glob('Ma*')) for start in range(0,len(list(c.glob('*.dat'))),20)]
    def run(job):
        case,start,stop=job;log=logs/f'{case}_{start:04d}.log'
        expected=min(stop,len(list((Path(a.source)/case).glob('*.dat'))))
        if all((ROOT/'datasets/fenli_paired'/case/f'{i:04d}.h5').exists() for i in range(start,expected)):
            print(case,start,'already complete',flush=True);return
        env=os.environ.copy();env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_THREADING_LAYER='GNU')
        with log.open('w',encoding='utf-8') as out:
            result=subprocess.run([sys.executable,'-u','-m','taoca.prepare','--source',a.source,'--case',case,'--start',str(start),'--stop',str(stop)],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,env=env)
        print(case,start,result.returncode,flush=True)
        if result.returncode:raise RuntimeError(f'Preparation failed: {log}')
    with concurrent.futures.ThreadPoolExecutor(a.workers) as pool:
        for future in concurrent.futures.as_completed([pool.submit(run,j) for j in jobs]):future.result()
    sys.path.insert(0,str(ROOT))
    from taoca.data import finalize
    meta=finalize(ROOT/'datasets/fenli_paired');print(json.dumps({k:len(v) for k,v in meta['cases'].items()}),flush=True)
    env=os.environ.copy();env.update(MKL_THREADING_LAYER='GNU',OMP_NUM_THREADS='2')
    for script in ['scripts/audit_paired.py','scripts/smoke_pipeline.py','scripts/write_delivery_report.py','scripts/package_server.py']:
        subprocess.run([sys.executable,'-u',script],cwd=ROOT,env=env,check=True)


if __name__=='__main__':main()
