import sys,json,hashlib,zipfile
from pathlib import Path
import numpy as np
import h5py
ROOT=Path(__file__).resolve().parents[1]


def main():
    data=ROOT/'datasets/joint_nearest';manifest=json.loads((data/'manifest.json').read_text())
    assert sum(c['frames'] for c in manifest['cases'].values())==500
    for case,record in manifest['cases'].items():
        path=data/(case+'_geometry.h5')
        if path.exists():continue
        with h5py.File(path.with_suffix('.partial.h5'),'w') as out:
            xy=out.create_dataset('xy',(record['frames'],4,2,48,256),dtype='f4',compression='lzf')
            for i in range(record['frames']):
                with h5py.File(ROOT/'datasets/fenli_paired'/case/f'{i:04d}.h5') as f:
                    xy[i]=np.stack([f['taoca'][name]['xy'][:] for name in ['smj','xmj','outer','gap']])
        path.with_suffix('.partial.h5').replace(path)
    files=[p for p in (ROOT/'taoca').glob('*.py')]
    files +=[p for p in data.iterdir() if p.suffix in ['.json','.h5'] and '.partial.' not in p.name]
    files +=[ROOT/p for p in ['train_joint.py','requirements.txt','JOINT_PROTOCOL.md','run_joint_dual_gpu.sh','scripts/check_cuda.py','scripts/check_joint_contract.py','scripts/prepare_joint.py','scripts/verify_package.py','scripts/compare_joint.py','LINUX_JOINT.md']]
    for route in ['cartesian','taoca']:
        files +=[ROOT/f'outputs/joint_{route}_smoke/{name}.json' for name in ['run','history','evaluation']]
    records=[dict(file=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files]
    # Separate manifest so original conditional package remains distinguishable.
    text=json.dumps(dict(files=records,frames=500,formal_training_completed=False,protocol='history-only joint flow/trajectory'),indent=2)
    dest=ROOT.parent/'FNOTransfomer_TAOCA_joint_server.zip'
    with zipfile.ZipFile(dest.with_suffix('.partial.zip'),'w',compression=zipfile.ZIP_STORED,allowZip64=True) as z:
        for p in files:z.write(p,'FNOTransfomer_TAOCA/'+p.relative_to(ROOT).as_posix())
        z.writestr('FNOTransfomer_TAOCA/server_package_manifest.json',text)
    dest.with_suffix('.partial.zip').replace(dest);print(json.dumps(dict(package=str(dest),bytes=dest.stat().st_size,files=len(files)+1)))


if __name__=='__main__':main()
