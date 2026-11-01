"""Verify the delivered file sizes and SHA-256 digests before server training."""
import argparse,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--size-only',action='store_true');a=p.parse_args()
    manifest=json.loads((ROOT/'server_package_manifest.json').read_text());total=0
    for record in manifest['files']:
        path=ROOT/record['file']
        if not path.is_file() or path.stat().st_size!=record['bytes']:raise ValueError(f'Missing or wrong file size: {path}')
        if not a.size_only:
            sha=hashlib.sha256()
            with path.open('rb') as f:
                while block:=f.read(8*1024*1024):sha.update(block)
            if sha.hexdigest()!=record['sha256']:raise ValueError(f'SHA-256 mismatch: {path}')
        total+=record['bytes']
    print(json.dumps(dict(verified=True,files=len(manifest['files']),bytes=total,sha256_checked=not a.size_only,frames=manifest['frames'])))


if __name__=='__main__':main()
