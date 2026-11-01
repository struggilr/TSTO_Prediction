"""Single-frame resolution and atlas-fusion ablations for TAOCA mapping."""
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
FILES=['Pressure','Density','Temperature','Velocity X','Velocity Y']
CHARTS=['smj','xmj','outer','gap']


def chart_donors(f,res):
    values=[];stored=[];geom=[];h2s=[];valids=[]
    for name in CHARTS:
        c=f[f'taoca/{name}'];v=c['fields'][:].reshape(5,-1).astype(np.float64)
        ids=c['inv_ids'][:];bary=c['inv_bary'][:].astype(np.float64);w=c['inv_weight'][:].astype(np.float64)
        mask=(ids[:,0]>=0)&(w>0)
        safe=np.maximum(ids,0)
        mapped=np.einsum('cnk,nk->cn',v[:,safe],bary)
        xy=c['xy'][:].reshape(2,-1).T.astype(np.float64)
        p=xy[safe];edges=np.stack([p[:,1]-p[:,0],p[:,2]-p[:,1],p[:,0]-p[:,2]],axis=1)
        h2=np.max(np.sum(edges*edges,axis=2),axis=1)
        gweight=w*h2
        values.append(mapped);stored.append(w);geom.append(gweight);h2s.append(h2);valids.append(mask)
    return np.asarray(values),np.asarray(stored),np.asarray(geom),np.asarray(h2s),np.asarray(valids)


def fuse(donors,mode):
    values,stored,geom,h2,valid=donors
    if mode=='geometric_over_h2':weights=stored
    elif mode=='geometry_only':weights=geom
    elif mode=='equal_chart_vote':weights=valid.astype(np.float64)
    elif mode=='finest_donor':
        score=np.where(valid,h2,np.inf);choice=np.argmin(score,axis=0)
        weights=np.zeros_like(score);has=valid.any(axis=0);weights[choice[has],np.where(has)[0]]=1.
    else:raise ValueError(mode)
    den=weights.sum(axis=0);out=np.einsum('kcn,kn->cn',values,weights)
    ok=den>0;out[:,ok]/=den[ok];out[:,~ok]=np.nan
    return out,ok


def high_gradient_nodes(xy,tri,pressure,pct=95):
    p0=xy[tri[:,0]];p1=xy[tri[:,1]];p2=xy[tri[:,2]]
    d1=p1-p0;d2=p2-p0;det=d1[:,0]*d2[:,1]-d1[:,1]*d2[:,0]
    good=np.abs(det)>1e-18;gmag=np.zeros(len(tri),dtype=np.float64)
    z0=pressure[tri[:,0]];z1=pressure[tri[:,1]];z2=pressure[tri[:,2]]
    gx=np.zeros(len(tri));gy=np.zeros(len(tri))
    gx[good]=((z1-z0)[good]*d2[good,1]-(z2-z0)[good]*d1[good,1])/det[good]
    gy[good]=(d1[good,0]*(z2-z0)[good]-d2[good,0]*(z1-z0)[good])/det[good]
    gmag=np.hypot(gx,gy)
    threshold=np.percentile(gmag[good],pct)
    cells=gmag>=threshold
    nodes=np.zeros(len(xy),dtype=bool);nodes[np.unique(tri[cells])]=True
    return nodes,float(threshold),float(cells.mean())


def metrics(true,pred,mask,std,shock):
    use=mask.copy();out=[]
    for k in range(5):
        e=pred[k]-true[k]
        g=np.sqrt(np.nanmean(e[use]**2));sg=np.sqrt(np.nanmean(e[use&shock]**2))
        bg=np.sqrt(np.nanmean(e[use&~shock]**2))
        out.append({'field':FILES[k],'rmse':float(g),'nrmse':float(g/std[k]),
                    'shock_rmse':float(sg),'shock_nrmse':float(sg/std[k]),
                    'nonshock_rmse':float(bg),'nonshock_nrmse':float(bg/std[k])})
    return out


def main():
    paths={48:ROOT/'outputs/TAOCA_boundary_v2_48x256/fenli_paired/Ma5.0/0029.h5',
           64:ROOT/'outputs/TAOCA_boundary_v2_64x384/fenli_paired/Ma5.0/0029.h5',
           32:ROOT/'outputs/TAOCA_boundary_v2_32x192_snapshot30/Ma5.0/Ma5.0/0029.h5'}
    result_dir=ROOT/'outputs/TAOCA_boundary_v2_48x256/snapshot30_Ma5.0/shock_error_refinement'
    result_dir.mkdir(parents=True,exist_ok=True)
    data={}
    for res,path in paths.items():
        with h5py.File(path,'r') as f:
            if res==48:
                truth=f['source/fields'][:].astype(np.float64)
                xy=np.column_stack([f['source/x'][:],f['source/y'][:]])
                tri=f['source/triangles'][:].astype(np.int32)
                time=float(f.attrs['time'])
                source_file=str(f.attrs['source'])
            else:
                if not np.array_equal(truth,f['source/fields'][:]):raise RuntimeError('Source field mismatch between resolutions')
            donors=chart_donors(f,res)
            data[res]={'donors':donors}
    std=np.std(truth,axis=1)
    global32,valid32=fuse(data[32]['donors'],'geometric_over_h2')
    global48,valid48=fuse(data[48]['donors'],'geometric_over_h2')
    global64,valid64=fuse(data[64]['donors'],'geometric_over_h2')
    common=valid32&valid48&valid64
    shock,threshold,shock_cells_fraction=high_gradient_nodes(xy,tri,truth[0],95)
    shock_use=shock&common
    method_rows={}
    modes=['geometric_over_h2','geometry_only','equal_chart_vote','finest_donor']
    dimkey={32:'32x192',48:'48x256',64:'64x384'}
    for res in (32,48,64):
        for mode in modes:
            pred,valid=fuse(data[res]['donors'],mode)
            use=common if res==48 else common
            method_rows[f'{dimkey[res]}_{mode}']={'coverage':float(valid.mean()),
                 'common_support_metrics':metrics(truth,pred,use,std,shock)}
    # Per-chart diagnostics identify which chart donors are reliable in the high-gradient band.
    single_chart={}
    for res in (32,48,64):
        values,stored,geom,h2,val=data[res]['donors']
        for i,name in enumerate(CHARTS):
            use=common&val[i]
            single_chart[f'{res}_{name}']={'coverage':float(val[i].mean()),
                 'common_support_count':int(use.sum()),
                 'metrics':metrics(truth,values[i],use,std,shock)}
    result={'source_file':source_file,'snapshot_index_1based':30,'time':time,'shock_definition':
            'top 5% source triangles by piecewise-linear pressure gradient magnitude; node mask is union of their vertices',
            'pressure_gradient_threshold':threshold,'shock_triangle_fraction':shock_cells_fraction,
            'shock_node_fraction':float(shock.mean()),'valid_coverage':{'32x192':float(valid32.mean()),'48x256':float(valid48.mean()),'64x384':float(valid64.mean()),
            'common_support':float(common.mean()),'common_support_high_gradient_nodes':int(shock_use.sum())},
            'normalization_std_source_nodes':std.tolist(),'ablation_metrics':method_rows,'single_chart_metrics':single_chart}
    (result_dir/'snapshot30_resolution_fusion_ablation.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')

    # Plot global and pressure-gradient-band NRMSE for the blending ablations.
    plt.rcParams.update({'font.family':'Times New Roman','font.size':9,'figure.facecolor':'white','axes.facecolor':'white'})
    fig,axs=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
    channels=['Pressure','Density','Temperature','Velocity X','Velocity Y'];colors={32:'#40916c',48:'#1764ab',64:'#d84a4a'}
    xidx=np.arange(5);width=.24
    for panel,(metric_key,title) in enumerate([('nrmse','Common-support NRMSE'),('shock_nrmse','High-gradient NRMSE')]):
        for res,offset,label in [(32,-width,'32×192 atlas'),(48,0,'48×256 atlas'),(64,width,'64×384 atlas')]:
            vals=[]
            mode='geometric_over_h2'
            for k in range(5):vals.append(method_rows[f'{dimkey[res]}_{mode}']['common_support_metrics'][k][metric_key])
            axs[panel].bar(xidx+offset,vals,width,color=colors[res],alpha=.84,label=label)
        axs[panel].set_title(title,fontweight='bold');axs[panel].set_xticks(xidx,channels,rotation=22,ha='right')
        axs[panel].set_ylabel('NRMSE');axs[panel].grid(axis='y',alpha=.22);axs[panel].legend(frameon=False)
    fig.suptitle(f'Ma=5.0 snapshot 30 (t={time:.2f}): resolution ablation',fontweight='bold',fontsize=13)
    fig.savefig(result_dir/'snapshot30_resolution_shock_nrmse.png',dpi=300,bbox_inches='tight');plt.close(fig)

    rows=['# Ma=5.0 第 30 帧激波误差与分辨率消融','',f'- 源文件：`{source_file}`',f'- 时间：t={time:.2f}',
          '- 高梯度区定义：源三角网格中压力 P1 梯度幅值最高的 5% 单元，其顶点并集。它是梯度诊断区，不是经验证的物理激波追踪器。',
          f'- 32×192 覆盖率：{valid32.mean():.3%}；48×256 覆盖率：{valid48.mean():.3%}；64×384 覆盖率：{valid64.mean():.3%}',
          f'- 三种分辨率共同支撑节点：{common.mean():.3%}', '',
          '## 分辨率比较（共同支撑域，当前几何/h² 融合）','',
          '| 通道 | 32×192 全域 | 48×256 全域 | 64×384 全域 | 32×192 高梯度区 | 48×256 高梯度区 | 64×384 高梯度区 |','|---|---:|---:|---:|---:|---:|---:|']
    m32=method_rows['32x192_geometric_over_h2']['common_support_metrics'];m48=method_rows['48x256_geometric_over_h2']['common_support_metrics'];m64=method_rows['64x384_geometric_over_h2']['common_support_metrics']
    for a,b,c in zip(m32,m48,m64):rows.append(f'| {a["field"]} | {a["nrmse"]:.5g} | {b["nrmse"]:.5g} | {c["nrmse"]:.5g} | {a["shock_nrmse"]:.5g} | {b["shock_nrmse"]:.5g} | {c["shock_nrmse"]:.5g} |')
    rows += ['', '![全域与高压力梯度区误差对比](snapshot30_resolution_shock_nrmse.png)','',
             '## 融合消融','', '同一分辨率的 `geometric_over_h2`、`geometry_only`、`equal_chart_vote` 与 `finest_donor` 指标见 JSON 文件。当前对比仅是单帧、非训练的映射回环试验；全域融合结果会随支撑区和有效 donor 变化，应结合覆盖率一起看。', '',
             '## 解读边界','',
             '降低分辨率可减少张量和训练计算量，但会使激波高梯度误差增加；提高分辨率能降低映射误差，代价是更多计算。固定预算下可考虑把节点从平滑区移到激波预测带。该诊断不包含模型预测、面积守恒或多个时间帧统计。']
    (result_dir/'snapshot30_shock_error_ablation.md').write_text('\n'.join(rows)+'\n',encoding='utf-8')
    print(json.dumps({'output':str(result_dir),'coverage':result['valid_coverage'],
                      'resolution_fusion_metrics':{k:v['common_support_metrics'] for k,v in method_rows.items()},
                      'shock_node_fraction':result['shock_node_fraction']},indent=2,ensure_ascii=False))

if __name__=='__main__':main()
