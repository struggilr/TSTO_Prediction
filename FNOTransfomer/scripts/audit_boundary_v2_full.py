"""Full-frame mapping audit for original vs boundary-trace refined TAOCA."""
import argparse,json
from pathlib import Path
import h5py
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
CHARTS=['smj','xmj','outer','gap']

def route_support(f,route):
    names=CHARTS if route=='taoca' else ['cartesian']
    return np.any(np.stack([f[f'{route}/{n}/inv_weight'][:]>0 for n in names]),axis=0)

def inverse(f,route):
    names=CHARTS if route=='taoca' else ['cartesian'];n=len(f['source/x'])
    q=np.zeros((5,n),dtype=np.float64);den=np.zeros(n,dtype=np.float64)
    for name in names:
        c=f[f'{route}/{name}'];v=c['fields'][:].reshape(5,-1);ids=c['inv_ids'][:]
        bary=c['inv_bary'][:];w=c['inv_weight'][:]
        vals=np.einsum('cnk,nk->cn',v[:,np.maximum(ids,0)],bary)
        q+=vals*w;den+=w
    valid=den>0;q[:,valid]/=den[valid];q[:,~valid]=np.nan
    return q,valid

def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,default=ROOT/'datasets/fenli_paired')
    p.add_argument('--candidate',type=Path,default=ROOT/'outputs/TAOCA_boundary_v2_64x384/fenli_paired')
    p.add_argument('--output',type=Path,default=ROOT/'outputs/TAOCA_boundary_v2_64x384/full_audit.json')
    a=p.parse_args();meta=json.loads((a.baseline/'manifest.json').read_text(encoding='utf-8'))
    std=np.asarray(meta['normalization']['std'],dtype=np.float64);case_rows=[];resolution=None
    nrmse={k:[] for k in ['TAOCA_original','TAOCA_boundary_v2','Cartesian']}
    coverage={k:[] for k in ['TAOCA_original','TAOCA_boundary_v2','Cartesian','common_all_three']}
    chart_rows={route:{n:{'mask_first':[],'mask_last':[],'nonpositive_J':[]} for n in CHARTS}
                for route in ['TAOCA_original','TAOCA_boundary_v2']}
    for case in sorted(meta['cases']):
        for record in meta['cases'][case]:
            bp=a.baseline/record['file'];cp=a.candidate/record['file']
            with h5py.File(bp,'r') as b,h5py.File(cp,'r') as c:
                if resolution is None:resolution=list(c['taoca/smj/xy'].shape[1:])
                elif resolution!=list(c['taoca/smj/xy'].shape[1:]):raise RuntimeError('Candidate chart resolution changes across frames')
                src=b['source/fields'][:]
                if src.shape!=c['source/fields'].shape or not np.array_equal(src,c['source/fields'][:]):
                    raise RuntimeError(f'Source mismatch: {case} frame {record["index"]}')
                sb=route_support(b,'taoca');sc=route_support(c,'taoca');sx=route_support(b,'cartesian')
                common=sb&sc&sx
                if not common.any():raise RuntimeError(f'Empty support: {case} frame {record["index"]}')
                qb,_=inverse(b,'taoca');qc,_=inverse(c,'taoca');qx,_=inverse(b,'cartesian')
                for key,q in [('TAOCA_original',qb),('TAOCA_boundary_v2',qc),('Cartesian',qx)]:
                    e=q[:,common]-src[:,common];r=np.sqrt(np.mean(e*e,axis=1))/std
                    nrmse[key].append(r.tolist())
                for key,s in [('TAOCA_original',sb),('TAOCA_boundary_v2',sc),('Cartesian',sx),('common_all_three',common)]:
                    coverage[key].append(float(s.mean()))
                chart_meta=[]
                for route,handle in [('TAOCA_original',b),('TAOCA_boundary_v2',c)]:
                    for name in CHARTS:
                        chart=handle[f'taoca/{name}'];mask=chart['mask'][:];jac=chart['J'][:]
                        chart_rows[route][name]['mask_first'].append(float(mask[0].mean()))
                        chart_rows[route][name]['mask_last'].append(float(mask[-1].mean()))
                        nbad=int(np.count_nonzero(jac<=0));chart_rows[route][name]['nonpositive_J'].append(nbad)
                        if route=='TAOCA_boundary_v2':
                            chart_meta.append({'chart':name,'min_J':float(np.min(jac)),'nonpositive_J':nbad,
                                               'first_row_valid':float(mask[0].mean()),'last_row_valid':float(mask[-1].mean())})
                case_rows.append({'case':case,'index':record['index'],'time':record['time'],
                                  'TAOCA_original_coverage':float(sb.mean()),
                                  'TAOCA_boundary_v2_coverage':float(sc.mean()),
                                  'Cartesian_coverage':float(sx.mean()),'common_coverage':float(common.mean()),
                                  'chart_geometry':chart_meta})
        print(f'{case}: audited {len(meta["cases"][case])} frames',flush=True)
    mean={k:np.mean(v,axis=0).tolist() for k,v in nrmse.items()}
    original=np.asarray(mean['TAOCA_original']);candidate=np.asarray(mean['TAOCA_boundary_v2'])
    relative=(candidate-original)/original*100
    result={'frame_count':len(case_rows),'field_names':meta['fields'],'training_std':std.tolist(),
            'resolution':resolution,'sample_points_per_route':4*resolution[0]*resolution[1],
            'metric_scope':'per-frame nodal P1 round-trip NRMSE on the intersection of original TAOCA, candidate TAOCA, and original Cartesian supports; frame means are then averaged across 500 frames',
            'coverage':{k:{'mean':float(np.mean(v)),'min':float(np.min(v)),'max':float(np.max(v))} for k,v in coverage.items()},
            'mean_nrmse':mean,'candidate_relative_change_vs_original_percent':relative.tolist(),
            'chart_boundary_valid_fraction':{route:{n:{'first_row_mean':float(np.mean(v['mask_first'])),'last_row_mean':float(np.mean(v['mask_last']))}
                                                     for n,v in by_chart.items()} for route,by_chart in chart_rows.items()},
            'nonpositive_J_total_by_chart':{route:{n:int(np.sum(v['nonpositive_J'])) for n,v in by_chart.items()}
                                            for route,by_chart in chart_rows.items()},
            'per_frame':case_rows}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='per_frame'},indent=2,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
