"""Render one TAOCA atlas and its inverse-mapped CFD snapshot."""
import argparse, json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

FIELDS=['Pressure','Density','Temperature','Velocity X','Velocity Y']
CHARTS=['smj','xmj','outer','gap']


def reconstruct(f):
    src=f['source'];truth=src['fields'][:]
    n=truth.shape[1]
    out=np.zeros_like(truth,dtype=np.float64);den=np.zeros(n,dtype=np.float64)
    for name in CHARTS:
        c=f[f'taoca/{name}'];vals=c['fields'][:].reshape(5,-1).astype(np.float64)
        ids=c['inv_ids'][:];bary=c['inv_bary'][:].astype(np.float64);w=c['inv_weight'][:].astype(np.float64)
        safe=np.maximum(ids,0)
        mapped=np.einsum('cnk,nk->cn',vals[:,safe],bary)
        out+=mapped*w[None,:];den+=w
    valid=den>0
    out[:,valid]/=den[valid]
    out[:,~valid]=np.nan
    return truth,out,valid


def triangle_for_display(f,valid=None):
    src=f['source'];tri=src['triangles'][:].astype(np.int32)
    mesh=mtri.Triangulation(src['x'][:],src['y'][:],triangles=tri)
    if valid is not None:
        mesh.set_mask(~valid[tri].all(axis=1))
    return mesh


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',type=Path,default=Path('outputs/TAOCA_boundary_v2_48x256/fenli_paired/Ma5.0/0029.h5'))
    p.add_argument('--output',type=Path,default=Path('outputs/TAOCA_boundary_v2_48x256/snapshot30_Ma5.0'))
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    with h5py.File(a.input,'r') as f:
        truth,pred,valid=reconstruct(f)
        src=f['source'];x=src['x'][:];y=src['y'][:]
        mesh=triangle_for_display(f);valid_mesh=triangle_for_display(f,valid)
        xy={n:f[f'taoca/{n}/xy'][:] for n in CHARTS}
        fields={n:f[f'taoca/{n}/fields'][:] for n in CHARTS}
        masks={n:f[f'taoca/{n}/mask'][:] for n in CHARTS}
        time=float(f.attrs['time']);source_file=str(f.attrs.get('source',a.input))
    std=np.std(truth,axis=1);err=pred-truth
    rmse=np.sqrt(np.nanmean(np.where(valid[None,:],err**2,np.nan),axis=1))
    nrmse=rmse/std
    coverage=float(valid.mean())

    plt.rcParams.update({'font.family':'Times New Roman','font.size':10,'axes.titlesize':10,'axes.labelsize':9,
                         'xtick.labelsize':8,'ytick.labelsize':8,'figure.facecolor':'white','axes.facecolor':'white'})
    # Source CFD vs atlas round-trip on the same original FE topology.
    fig,axs=plt.subplots(5,3,figsize=(13.2,13.5),constrained_layout=True)
    coltitles=['Original CFD on source FE mesh','TAOCA → physical mesh (round trip)','Absolute error']
    for j,t in enumerate(coltitles):axs[0,j].set_title(t,fontweight='bold',pad=9)
    for k,name in enumerate(FIELDS):
        lo,hi=np.nanpercentile(truth[k],[1,99])
        levels=np.linspace(lo,hi,41)
        im0=axs[k,0].tricontourf(mesh,truth[k],levels=levels,cmap='turbo',extend='both')
        im1=axs[k,1].tricontourf(valid_mesh,pred[k],levels=levels,cmap='turbo',extend='both')
        emax=max(float(np.nanpercentile(np.abs(err[k,valid]),99.5)),1e-12)
        im2=axs[k,2].tricontourf(valid_mesh,np.abs(err[k]),levels=np.linspace(0,emax,41),cmap='magma',extend='max')
        axs[k,0].set_ylabel(name)
        for ax in axs[k]:
            ax.set_aspect('equal',adjustable='box');ax.set_xlabel('x');ax.set_ylabel(ax.get_ylabel() or 'y')
        fig.colorbar(im0,ax=axs[k,0],fraction=.046,pad=.03)
        fig.colorbar(im1,ax=axs[k,1],fraction=.046,pad=.03)
        fig.colorbar(im2,ax=axs[k,2],fraction=.046,pad=.03)
        axs[k,0].text(.02,.98,f'NRMSE={nrmse[k]:.4g}',transform=axs[k,0].transAxes,va='top',ha='left',
                      bbox=dict(facecolor='white',edgecolor='none',alpha=.85,pad=2))
    fig.suptitle(f'Ma=5.0, snapshot 30 (t={time:.2f}): TAOCA round-trip',fontsize=14,fontweight='bold')
    fig.savefig(a.output/'snapshot30_roundtrip_5channels.png',dpi=300,bbox_inches='tight')
    fig.savefig(a.output/'snapshot30_roundtrip_5channels.pdf',bbox_inches='tight')
    plt.close(fig)

    # Four logical charts, all displayed in physical coordinates.
    fig,axs=plt.subplots(5,4,figsize=(15,13.5),constrained_layout=True)
    for j,n in enumerate(CHARTS):axs[0,j].set_title({'smj':'SMJ O-chart','xmj':'XMJ O-chart','outer':'Outer annulus','gap':'Bridge H-chart'}[n],fontweight='bold')
    for k,name in enumerate(FIELDS):
        lo,hi=np.nanpercentile(truth[k],[1,99]);levels=np.linspace(lo,hi,41)
        for j,n in enumerate(CHARTS):
            cxy=xy[n];xx,yy=cxy[0],cxy[1]
            z=np.ma.array(fields[n][k],mask=~masks[n])
            im=axs[k,j].contourf(xx,yy,z,levels=levels,cmap='turbo',extend='both')
            axs[k,j].set_aspect('equal',adjustable='box');axs[k,j].set_xlabel('x')
            if j==0:axs[k,j].set_ylabel(f'{name}\ny')
            else:axs[k,j].set_ylabel('y')
        fig.colorbar(im,ax=axs[k,:].tolist(),fraction=.015,pad=.012)
    fig.suptitle(f'Ma=5.0, snapshot 30 (t={time:.2f}): four TAOCA charts in physical space',fontsize=14,fontweight='bold')
    fig.savefig(a.output/'snapshot30_taoca_4charts_5channels.png',dpi=300,bbox_inches='tight')
    fig.savefig(a.output/'snapshot30_taoca_4charts_5channels.pdf',bbox_inches='tight')
    plt.close(fig)

    fields_json=[]
    for i,name in enumerate(FIELDS):
        fields_json.append({'field':name,'RMSE':float(rmse[i]),'NRMSE':float(nrmse[i]),
                            'truth_min':float(np.min(truth[i])),'truth_max':float(np.max(truth[i])),
                            'error_p99':float(np.nanpercentile(np.abs(err[i,valid]),99))})
    result={'source_file':source_file,'snapshot_index_1based':30,
            'time':time,'resolution':[48,256],'charts':CHARTS,'source_nodes':int(truth.shape[1]),
            'covered_nodes':int(valid.sum()),'coverage':coverage,
            'metric_definition':'nodewise unweighted P1 mapping round-trip; NRMSE normalized by this-frame source-node standard deviation',
            'metrics':fields_json}
    (a.output/'snapshot30_metrics.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    lines=['# Ma=5.0 第 30 帧 TAOCA 网格变换往返结果','',f'- 源文件：`{result["source_file"]}`',f'- 时间：t={time:.3f}',f'- TAOCA 分辨率：每张图 48×256，共四张图册',f'- 源节点覆盖率：{coverage:.4%}（{valid.sum():,}/{len(valid):,}）','',
           '这里的往返过程为：原始非结构 CFD 场 → 四张 TAOCA 计算图 → donor/receptor 加权回映射至原始源网格。NRMSE 按当前帧源网格节点标准差归一化，未做面积加权；该结果是单帧映射诊断，不是神经网络预测误差。','',
           '## 五通道往返误差','', '| 通道 | RMSE | NRMSE | 绝对误差 P99 |','|---|---:|---:|---:|']
    for m in fields_json:lines.append(f'| {m["field"]} | {m["RMSE"]:.6g} | {m["NRMSE"]:.6g} | {m["error_p99"]:.6g} |')
    lines += ['', '## 图件','', '![四图册五通道物理坐标分布](snapshot30_taoca_4charts_5channels.png)','', '![原始 CFD、TAOCA 往返和绝对误差](snapshot30_roundtrip_5channels.png)','',
              '## 解读','',f'该帧 TAOCA 覆盖率为 {coverage:.2%}。覆盖区域内，五通道 NRMSE 见上表。四图册图展示各局部坐标图采样到的物理流场；往返图在同一原始 FE 网格上比较源场与回映射场，并单列绝对误差。空白区域代表没有有效 donor，不能解释为零值。','',
              '颜色范围使用源场第 1–99 百分位进行对照显示，误差图使用绝对误差第 99.5 百分位作为显示上限；表格统计不裁剪误差。']
    (a.output/'snapshot30_TAOCA往返分析.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2,ensure_ascii=False))

if __name__=='__main__':main()
