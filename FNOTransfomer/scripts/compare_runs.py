import json,argparse
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--a',type=Path,default=Path('outputs/cartesian'));p.add_argument('--b',type=Path,default=Path('outputs/taoca'));p.add_argument('--output',type=Path,default=Path('outputs/comparison.json'));a=p.parse_args();result={};runs=[]
    for route,folder in [('cartesian',a.a),('taoca',a.b)]:
        run=json.loads((folder/'run.json').read_text());ev=json.loads((folder/'evaluation.json').read_text());history=json.loads((folder/'history.json').read_text())
        if ev.get('smoke_only'):raise ValueError('Smoke results cannot be compared as trained models')
        runs.append(run);result[route]=dict(gpu=run['gpu'],parameters=run['parameters'],epochs=len(history),total_train_seconds=sum(e['train_seconds'] for e in history),mean_epoch_seconds=sum(e['train_seconds'] for e in history)/len(history),best_validation_mse=min(e['val_mse'] for e in history),evaluation=ev)
    for key in ['dataset_manifest_sha256','model_config','parameters']:
        if runs[0][key]!=runs[1][key]:raise ValueError(f'Different {key}')
    for key in ['seed','batch_size','epochs','lr']:
        if runs[0]['arguments'][key]!=runs[1]['arguments'][key]:raise ValueError(f'Different {key}')
    result['limitations']='Concurrent GPUs share I/O. Repeat seeds and swap GPUs for a paper. Representation MSE differs in sampling measure; rank accuracy by common physical test errors.'
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
