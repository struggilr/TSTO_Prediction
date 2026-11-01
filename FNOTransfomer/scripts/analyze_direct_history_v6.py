"""Wait for the existing V3 run, then export honest comparison figures and report."""
import os
os.environ['MKL_THREADING_LAYER']='GNU'
import sys,json,time,traceback,argparse
from pathlib import Path
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'outputs/direct_history_v6'
OUT=BASE/'analysis'
FIELDS=['Pressure','Density','Temperature','VelocityX','VelocityY']
UNITS=['Pa','kg/m3','K','m/s','m/s']
COLORS=['#0000ff','#888888','#ff0000','#009999']
NAMES=['FNO_Cartesian','FNO_TAOCA_V3','FNO_TAOCA_V6','Persistence']
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def status(state,**kw):
    p=BASE/'analysis_status.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(dict(state=state,time=time.strftime('%Y-%m-%d %H:%M:%S'),**kw),indent=2),encoding='utf-8');tmp.replace(p)
def save(fig,name,vector=True):
    p=OUT/'figures'/name;p.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(str(p)+'.png',dpi=240,bbox_inches='tight')
    if vector:
        fig.savefig(str(p)+'.pdf',bbox_inches='tight');fig.savefig(str(p)+'.svg',bbox_inches='tight')
    plt.close(fig)
def analyze(cloud_limit=None):
    plt.rcParams.update({'font.family':'Times New Roman','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    OUT.mkdir(parents=True,exist_ok=True)
    comparison=read(BASE/'comparison.json');rows=read(BASE/'test_frame_metrics.json')
    for region,g in comparison['regions'].items():
        fig,ax=plt.subplots(figsize=(7.2,3.3),layout='constrained')
        for k,name in enumerate(NAMES):ax.plot(range(5),g['nrmse'][name],label=name,color=COLORS[k],ls=['--',':','-','-.'][k],marker='o',ms=3,alpha=.8)
        ax.set_xticks(range(5),FIELDS);ax.set(ylabel='NRMSE (training standard deviation)',title=region+' region; common legal support');ax.legend(fontsize=7);save(fig,'error_'+region)
    runs=[ROOT/'outputs/physics_v2/cartesian',ROOT/'outputs/atlas_v3/taoca',BASE/'taoca']
    fig,axs=plt.subplots(1,2,figsize=(9,3.2),layout='constrained')
    costs=[]
    for k,(folder,name) in enumerate(zip(runs,NAMES)):
        history=read(folder/'history.json');costs.append(sum(r['train_seconds'] for r in history)/3600)
        for ax,split in zip(axs,['train','val']):ax.plot([r['epoch'] for r in history],[r[split]['loss'] for r in history],color=COLORS[k],ls=['--',':','-'][k],label=name,alpha=.85);ax.set(xlabel='Epoch',ylabel=split+' objective');ax.set_yscale('log')
    axs[0].legend(fontsize=7);save(fig,'training_curves')
    fig,ax=plt.subplots(figsize=(6,3),layout='constrained');ax.bar(NAMES[:3],costs,color=COLORS[:3],alpha=.7);ax.set_ylabel('Sum of logged training time (h)');save(fig,'training_cost')
    for grouping in ['lead','Mach']:
        keys=sorted(set(r['lead'] if grouping=='lead' else r['window'].split('_s')[0] for r in rows))
        fig,axs=plt.subplots(1,5,figsize=(15,3),layout='constrained')
        for k,name in enumerate(NAMES):
            vals=[]
            for key in keys:
                rr=[r for r in rows if (r['lead'] if grouping=='lead' else r['window'].split('_s')[0])==key];n=sum(r['n'] for r in rr)
                vals.append(np.sqrt(sum(np.array(r[name])**2*r['n'] for r in rr)/max(1,n)))
            vals=np.array(vals)
            for j,ax in enumerate(axs):ax.plot(range(len(keys)),vals[:,j],label=name,color=COLORS[k],ls=['--',':','-','-.'][k],alpha=.8);ax.set(title=FIELDS[j],ylabel='Pooled NRMSE');ax.set_xticks(range(len(keys)),keys,rotation=45)
        axs[0].legend(fontsize=6);save(fig,'error_by_'+grouping)
    fig,ax=plt.subplots(figsize=(8,3),layout='constrained');ax.plot([r['n']/(280*440) for r in rows],color=COLORS[2]);ax.set(xlabel='Forecast instance (overlapping windows)',ylabel='Common support / grid size');save(fig,'coverage')
    cmap=LinearSegmentedColormap.from_list('Small Rainbow',['blue','cyan','lime','yellow','red'])
    meta=read(ROOT/'datasets/joint_nearest/manifest.json');mean=np.array(meta['mean']);std=np.array(meta['std'])
    paths=sorted((BASE/'taoca/predictions').glob('*.h5'));count=0;links=[];trajectory=[]
    for wi,path in enumerate(paths):
        with h5py.File(path) as b,h5py.File(runs[0]/'predictions'/path.name) as a:
            ref=b['reference'][:];valid=b['valid'][:].astype(bool);pa=a['predicted'][:];pb=b['predicted'][:];common=valid&b['support'][:].astype(bool)&a['support'][:].astype(bool)
            trajectory.append((path.stem,b['pose'][:],b['target_pose'][:]))
        for t in range(len(ref)):
            if cloud_limit is not None and count>=cloud_limit:break
            for j in range(5):
                arrays=[ref[t,j]*std[j]+mean[j],pa[t,j]*std[j]+mean[j],pb[t,j]*std[j]+mean[j]]
                finite=np.concatenate([x[common[t]&np.isfinite(x)] for x in arrays]);lo,hi=float(finite.min()),float(finite.max());detail=np.quantile(arrays[0][common[t]],[.01,.99])
                fig,axs=plt.subplots(2,3,figsize=(13,6.5),layout='constrained')
                for row,(vmin,vmax) in enumerate([(lo,hi),detail]):
                    for col,(arr,name) in enumerate(zip(arrays,['CFD reference','FNO_Cartesian','FNO_TAOCA_V6'])):
                        ax=axs[row,col];im=ax.imshow(np.where(common[t],arr,np.nan),origin='lower',extent=[-10,100,-20,50],cmap=cmap,vmin=vmin,vmax=max(vmax,vmin+1e-12),interpolation='nearest');ax.set(title=name,xlabel='x (m)',ylabel='y (m)')
                    fig.colorbar(im,ax=list(axs[row]),shrink=.75,extend='both' if row else 'neither',label=FIELDS[j]+' ('+UNITS[j]+'); '+('full range' if row==0 else 'reference 1-99%; tails saturated'))
                fig.suptitle(f'{path.stem}, lead {t+1}; {FIELDS[j]}; common support, no smoothing')
                name=f'clouds/{path.stem}_lead{t+1:02d}_{FIELDS[j]}';save(fig,name,False);links.append(name+'.png')
                if wi%11==0 and t in [0,4,9]:
                    fig,axs=plt.subplots(1,2,figsize=(10,3.5),layout='constrained');errors=[np.abs(arrays[k]-arrays[0]) for k in [1,2]];hi=max(float(np.nanmax(e[common[t]])) for e in errors)
                    for ax,e,name2 in zip(axs,errors,['FNO_Cartesian','FNO_TAOCA_V6']):im=ax.imshow(np.where(common[t],e,np.nan),origin='lower',extent=[-10,100,-20,50],cmap=cmap,vmin=0,vmax=max(hi,1e-12),interpolation='nearest');ax.set_title(name2+' absolute error')
                    fig.colorbar(im,ax=list(axs),label=UNITS[j]);save(fig,f'errors/{path.stem}_lead{t+1:02d}_{FIELDS[j]}',False)
            count+=1
            status('rendering',completed_forecast_instances=count,total_forecast_instances=len(rows))
        if cloud_limit is not None and count>=cloud_limit:break
    if trajectory:
        name,p,q=trajectory[0];fig,axs=plt.subplots(2,3,figsize=(10,5),layout='constrained')
        for j,ax in enumerate(axs.flat):ax.plot(q[:,j],color=COLORS[0],ls='--',label='CFD reference');ax.plot(p[:,j],color=COLORS[2],label='Shared trajectory predictor');ax.set(title=f'Body {j//3+1}: '+['x (m)','y (m)','angle (rad)'][j%3],xlabel='Forecast lead')
        axs[0,0].legend(fontsize=6);save(fig,'shared_trajectory_example')
    (OUT/'gallery.html').write_text('<html><meta charset="utf-8"><body><h1>Forecast fields: full range and reference 1–99% detail</h1>'+''.join(f'<p><a href="figures/{s}">{s}</a></p>' for s in links)+'</body></html>',encoding='utf-8')
    lines=['# TAOCA V6原始CFD历史直接采样训练结果自动分析','',f"评估包含{comparison['forecast_instances']}个预测实例；滑动窗口相互重叠，不代表独立样本。",'','轨迹预测器共享；Cartesian权重固定。所有定量比较采用三个版本共同合法支持域。','', '|区域|变量|FNO_Cartesian|FNO_TAOCA_V6|TAOCA相对误差下降|','|---|---|---:|---:|---:|']
    for region,g in comparison['regions'].items():
        for j,field in enumerate(FIELDS):
            a=g['nrmse'][NAMES[0]][j];b=g['nrmse'][NAMES[2]][j];lines.append(f'|{region}|{field}|{a:.6f}|{b:.6f}|{100*(a-b)/a:.2f}%|')
    lines+=['','正的下降百分比表示改善，负值表示退化。不能用局部改进替代全局优势证明。','', '## 图与解释','训练曲线用于检查优化及验证集选择；分步长曲线反映预测误差累积；分工况图反映不同已知马赫数序列的差异。wall为近壁区域，shock为高压力梯度代理区域，并非严格激波识别。','']
    for name in ['training_curves','error_all','error_wall','error_shock','error_by_lead','error_by_Mach','coverage','training_cost','shared_trajectory_example']:lines +=[f'![{name}](figures/{name}.png)','']
    lines+=['[全部逐帧五通道云图索引](gallery.html)','云图不插值平滑。上排使用各方法共同全数值范围；下排采用参考场1–99%范围，明确标示尾部截断。空白代表固体或共同支持域外，不能解读为零值。','', '## 结论边界','本实验为单随机种子、已知工况、滑动窗口条件预测。测试集在研究中已被反复查看，结果属于探索性证据。训练耗时来自日志，硬件负载可能不同，不能据此推导严格加速比。没有证明严格守恒、求解器级GCL或独立新轨迹泛化。']
    (OUT/'分析报告.md').write_text('\n'.join(lines),encoding='utf-8')
    status('complete',output=str(OUT),cloud_figures=len(links),preview=cloud_limit is not None)
def main():
    p=argparse.ArgumentParser();p.add_argument('--wait',action='store_true');p.add_argument('--preview',action='store_true');a=p.parse_args()
    if a.wait:
        status('waiting_for_training_and_comparison',pid=os.getpid())
        while not (BASE/'comparison.json').exists():
            state=read(BASE/'pipeline_status.json')
            if state.get('state',state.get('status'))=='failed':raise RuntimeError('Training pipeline failed; see its logs')
            time.sleep(20)
    analyze(1 if a.preview else None)
if __name__=='__main__':
    try:main()
    except Exception:
        status('failed',error=traceback.format_exc());raise
