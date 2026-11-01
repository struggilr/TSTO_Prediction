"""Prepare the isolated, boundary-trace TAOCA v2 64x384 mapping dataset."""
import argparse,concurrent.futures,json,os,subprocess,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',default='H:/fenli')
    p.add_argument('--output',type=Path,default=ROOT/'outputs/TAOCA_boundary_v2_64x384/fenli_paired')
    p.add_argument('--workers',type=int,default=3)
    p.add_argument('--chunk',type=int,default=20)
    p.add_argument('--ny',type=int,default=64)
    p.add_argument('--nx',type=int,default=384)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    logdir=a.output.parent/'logs';logdir.mkdir(parents=True,exist_ok=True)
    cases=sorted(Path(a.source).glob('Ma*'))
    if len(cases)!=5: raise RuntimeError(f'Expected five Ma trajectories, found {len(cases)} under {a.source}')
    jobs=[]
    for case in cases:
        n=len(list(case.glob('*.dat')))
        if n not in (96,101): raise RuntimeError(f'Unexpected frame count {case.name}: {n}')
        for start in range(0,n,a.chunk): jobs.append((case.name,start,min(start+a.chunk,n)))
    def execute(job):
        case,start,stop=job;folder=a.output/case
        if all((folder/f'{i:04d}.h5').exists() for i in range(start,stop)):
            print(f'{case} {start}:{stop} already complete',flush=True);return
        log=logdir/f'{case}_{start:04d}.log'
        env=os.environ.copy();env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_THREADING_LAYER='GNU')
        cmd=[sys.executable,'-u','-m','taoca.prepare','--source',a.source,'--output',str(a.output),
             '--case',case,'--start',str(start),'--stop',str(stop),'--ny',str(a.ny),'--nx',str(a.nx)]
        with log.open('w',encoding='utf-8') as out:
            result=subprocess.run(cmd,cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,env=env)
        print(f'{case} {start}:{stop} exit={result.returncode}',flush=True)
        if result.returncode: raise RuntimeError(f'Preparation failed: {log}')
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        list(pool.map(execute,jobs))
    sys.path.insert(0,str(ROOT))
    from taoca.data import finalize
    meta=finalize(a.output)
    status={'output':str(a.output),'cases':{k:len(v) for k,v in meta['cases'].items()},
            'resolution':[a.ny,a.nx],'charts':4,'frames':sum(map(len,meta['cases'].values())),
            'boundary_trace':'source-polyline P1 restriction and wall/farfield trace donor remapping'}
    (a.output.parent/'preparation_status.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
    print(json.dumps(status,indent=2),flush=True)

if __name__=='__main__': main()
