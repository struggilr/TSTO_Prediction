from pathlib import Path
import sys
ROOT=Path(r"E:\1-AllCode\mesh transformation")
WORK=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(WORK.parent/'taoca_validation_1013'))
import complete_cgns_io as cgns
from benchmark import charts_for,sample_source
from overset_atlas_transform import sequence_anchor
import numpy as np, json, hashlib, time
import matplotlib.tri as mtri

LABELS=[f"0.{i:02d}0000" for i in range(1,11)]
SOURCE_ROOT=ROOT/'cgns_hdf5_cache'
FIELDS=('Pressure','Density','Temperature','VelocityX','VelocityY','Mach')

def main(out=WORK/'datasets'):
    out=Path(out); src=out/'source'; ta=out/'taoca';src.mkdir(parents=True,exist_ok=True);ta.mkdir(parents=True,exist_ok=True)
    paths=[SOURCE_ROOT/f'0718_3-C0-{x}.5-1to2_7-Ma5.hdf5.cgns' for x in LABELS]
    seq=[cgns.parse_cgns(p) for p in paths]
    anchor=sequence_anchor(seq,256,.8);anchor['body_reference']={}
    records=[]; started=time.perf_counter()
    for label,path,data in zip(LABELS,paths,seq):
        field={k:data['fields'][k].astype(np.float32) for k in FIELDS}
        source_valid=np.zeros(len(data['x']),dtype=np.uint8);source_valid[np.unique(data['triangles'])]=1
        gx,gy=np.meshgrid(np.linspace(data['x'].min(),data['x'].max(),96),np.linspace(data['y'].min(),data['y'].max(),64))
        triang=mtri.Triangulation(data['x'],data['y'],data['triangles'])
        raster={k.lower():np.nan_to_num(mtri.LinearTriInterpolator(triang,v)(gx,gy).filled(np.nan),nan=0.).astype(np.float32) for k,v in field.items()}
        srcmask=np.isfinite(mtri.LinearTriInterpolator(triang,field['Mach'])(gx,gy).filled(np.nan)).astype(np.uint8)
        np.savez_compressed(src/f'{label}.npz',x=gx.astype(np.float32),y=gy.astype(np.float32),source_valid=srcmask,**raster)
        charts=charts_for(data,anchor,1,.5,.05);sample_source(data,charts)
        payload={};coverage={}
        for name,c in charts.items():
            coverage[name]=float(c['valid_donor_cells'].mean())
            for key in ('x','y','J','source_valid','valid_donor_cells'):payload[f'{name}_{key}']=c[key].astype(np.float32 if key not in ('source_valid','valid_donor_cells') else np.uint8)
            for key in FIELDS:payload[f'{name}_{key.lower()}']=np.nan_to_num(c['fields'][key],nan=0.).astype(np.float32)
        np.savez_compressed(ta/f'{label}.npz',**payload)
        records.append(dict(time_label=label,source_file=str(path),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),source_nodes=len(data['x']),source_cells=len(data['triangles']),source_referenced_nodes=int(source_valid.sum()),taoca_shapes={n:list(c['x'].shape) for n,c in charts.items()},taoca_donor_fraction=coverage))
        print(label,records[-1]['source_nodes'],flush=True)
    manifest=dict(version='0.1',fields=FIELDS,time_labels=LABELS,history_len=4,pred_len=2,dt_assumed=.01,source_format='complete TRI_3 + QUAD_4 split to positive TRI_3',atlas='TAOCA four charts; factor .5; float32 training payload; masks uint8',records=records,elapsed_seconds=time.perf_counter()-started)
    (out/'dataset_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(manifest,ensure_ascii=False))
if __name__=='__main__':main()
