"""Build a portable training delivery only after the 500-frame audit succeeds."""
import hashlib,json,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    data=ROOT/'datasets/fenli_paired';audit=json.loads((data/'audit.json').read_text());smoke=json.loads((ROOT/'outputs/smoke_pipeline.json').read_text())
    if audit['summary']['frames']!=500 or len(smoke['results'])!=2:raise ValueError('Incomplete validation')
    files=[]
    for folder in ['taoca','datasets/fenli_paired']:
        files.extend(p for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py','.json','.h5'))
    files.extend(ROOT/p for p in ['train_compare.py','requirements.txt','run_dual_gpu.sh','README.md','scripts/compare_runs.py','scripts/audit_paired.py','scripts/check_training_contract.py','scripts/verify_package.py','outputs/smoke_pipeline.json'])
    files.extend(ROOT/p for p in ['数据与训练接口交付报告.md','LINUX_5090.md','scripts/check_cuda.py','outputs/windows_5090_validation.json'])
    digest=[]
    for p in sorted(files):
        sha=hashlib.sha256()
        with p.open('rb') as f:
            while block:=f.read(8*1024*1024):sha.update(block)
        digest.append(dict(file=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha.hexdigest()))
    manifest=ROOT/'server_package_manifest.json';manifest.write_text(json.dumps(dict(files=digest,frames=500,formal_training_completed=False),indent=2),encoding='utf-8')
    files.append(manifest);destination=ROOT.parent/'FNOTransfomer_TAOCA_server.zip';temporary=destination.with_suffix('.partial.zip')
    # HDF5 is already compressed. ZIP_STORED avoids another large compression pass.
    with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_STORED,allowZip64=True) as archive:
        for p in files:archive.write(p,arcname='FNOTransfomer_TAOCA/'+p.relative_to(ROOT).as_posix())
    temporary.replace(destination)
    status=ROOT/'outputs/implementation_status.json'
    status.write_text(json.dumps(dict(status='ready_for_server_training',frames=500,dataset_audit_passed=True,real_data_smoke_passed=True,formal_training_completed=False,package=str(destination)),indent=2),encoding='utf-8')
    print(json.dumps(dict(package=str(destination),bytes=destination.stat().st_size,files=len(files))),flush=True)


if __name__=='__main__':main()
