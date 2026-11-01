from pathlib import Path
import sys, json, shutil, subprocess
ROOT=Path(__file__).resolve().parents[1]
V2=ROOT.parents[1] / "models/2026-09-24_FNOTransfomerCDDIM_V2.0"
raw=Path(r'E:\fenli'); out=ROOT/'datasets'/'fenli_v2'
out.mkdir(parents=True,exist_ok=True)
# Reuse the maintained Tecplot parser and fixed-coordinate rasterizer from V2.0.
cmd=[sys.executable,str(V2/'dataset_tools'/'dat2npz_time_split_v2.py'),'--data-root',str(raw),'--output-dir',str(out)]
print('RUN',' '.join(cmd));raise SystemExit(subprocess.call(cmd))
