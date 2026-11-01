from pathlib import Path
import sys,subprocess,concurrent.futures,tempfile,shutil
ROOT=Path(__file__).resolve().parents[1];V2=ROOT.parents[1] / "models/2026-09-24_FNOTransfomerCDDIM_V2.0";raw=Path(r'E:\fenli');out=ROOT/'datasets'/'fenli_v2_parallel';out.mkdir(parents=True,exist_ok=True)
def one(ma):
 d=out/f'Ma{ma}';d.mkdir(exist_ok=True)
 with tempfile.TemporaryDirectory() as tmp:
  t=Path(tmp)/f'Ma{ma}';t.mkdir()
  for f in (raw/f'Ma{ma}').glob('*.dat'):shutil.copy2(f,t/f.name)
  cmd=[sys.executable,str(V2/'dataset_tools'/'dat2npz_time_split_v2.py'),'--data-root',tmp,'--output-dir',str(d)]
  return subprocess.run(cmd,capture_output=True,text=True).returncode
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
 for ma,code in zip((5.0,5.5,6.0,6.5,7.0),ex.map(one,(5.0,5.5,6.0,6.5,7.0))):print(ma,code,flush=True)
