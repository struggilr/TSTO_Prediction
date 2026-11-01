"""Validation-only paired shock profiles and raw-source direct sampling audit."""
import os
os.environ['MKL_THREADING_LAYER']='GNU'
import sys,json,time,csv,traceback,gc
from pathlib import Path
import numpy as np
import h5py
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import map_coordinates,binary_erosion,distance_transform_edt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from taoca.allocated_data_v3 import AllocatedWindows
from taoca.metric_model_v8 import MetricPredictor as AllocatedPredictor
from taoca.chart_transfer_v3 import build_maps,apply_maps,fuse
from taoca.physics_evaluate import polygons
from taoca.physics_losses import regional_maps
from taoca.joint import sample_grid
OUT=ROOT/'outputs/shock_diagnosis_v9'
FIELDS=['Pressure','Density','Temperature','VelocityX','VelocityY']
COLORS={'Raw CFD P1':'#222222','Cartesian truth':'#888888','TAOCA cascade RT':'#007f7f','TAOCA direct RT':'#d9a000','FNO_Cartesian':'#0000ff','FNO_TAOCA_V8':'#ff0000'}
def dump(name,value):
    p=OUT/name;tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(p)
def source_sample(src,queries,mean,std):
    tri=mtri.Triangulation(src['x'],src['y'],src['triangles']);finder=tri.get_trifinder();points=np.stack([src['x'],src['y']],1);values=src['fields']
    result=[]
    for query in queries:
        shape=query.shape[:-1];p=query.reshape(-1,2);idx=finder(p[:,0],p[:,1]);valid=idx>=0;ids=src['triangles'][np.maximum(idx,0)];v=points[ids];a=v[:,1]-v[:,0];b=v[:,2]-v[:,0];r=p-v[:,0];det=a[:,0]*b[:,1]-a[:,1]*b[:,0]
        u=(r[:,0]*b[:,1]-r[:,1]*b[:,0])/det;w=(a[:,0]*r[:,1]-a[:,1]*r[:,0])/det;bary=np.stack([1-u-w,u,w],1)
        out=np.einsum('cnk,nk->cn',values[:,ids],bary);out=(out-mean[:,None])/std[:,None];out[:,~valid]=0
        result.append((out.reshape(5,*shape).astype('f4'),valid.reshape(shape)))
    del finder,tri;gc.collect();return result
def profile_metrics(distance,values):
    """Peak-gradient position and nearest 10/90 crossings; reject multi-wave profiles."""
    if not np.isfinite(values).all():return dict(valid=False,reason='nonfinite profile')
    left=float(np.median(values[:12]));right=float(np.median(values[-12:]));jump=right-left
    if abs(jump)<1e-8:return dict(valid=False,reason='negligible endpoint jump')
    z=(values-left)/jump;der=np.gradient(values,distance);peak=int(np.argmax(abs(der)))
    if peak<8 or peak>=len(values)-8:return dict(valid=False,reason='peak near profile boundary')
    def crossings(level):
        ids=np.flatnonzero((z[:-1]-level)*(z[1:]-level)<=0);ids=ids[abs(z[ids+1]-z[ids])>1e-12]
        return [float(distance[i]+(level-z[i])/(z[i+1]-z[i])*(distance[i+1]-distance[i])) for i in ids]
    a=crossings(.1);b=crossings(.9)
    if not a or not b:return dict(valid=False,reason='missing 10/90 crossing')
    x10=min(a,key=lambda x:abs(x-distance[peak]));x90=min(b,key=lambda x:abs(x-distance[peak]));variation=float(np.abs(np.diff(z)).sum())
    if x90<=x10 or variation>3 or not(x10-.5<=distance[peak]<=x90+.5):return dict(valid=False,reason='multi-wave or nonmonotone profile',variation=variation)
    return dict(valid=True,position=float(distance[peak]),jump=jump,width=x90-x10,variation=variation)
def accumulate(acc,key,pred,ref,mask):
    g=acc.setdefault(key,dict(n=0,sse=np.zeros(5)));g['n']+=int(mask.sum());g['sse']+=np.square(pred[:,mask].astype('f8')-ref[:,mask]).sum(1)
def export(fig,name):
    for suffix in ['png','pdf','svg']:fig.savefig(OUT/'figures'/(name+'.'+suffix),dpi=350,bbox_inches='tight')
    plt.close(fig)
def main():
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'figures').mkdir(exist_ok=True);(OUT/'profiles').mkdir(exist_ok=True)
    dump('figure_contract.json',dict(conclusion='Locate mapping versus prediction losses without presuming TAOCA superiority',evidence=['paired raw-source and cascade roundtrip','same physical normal-line profiles','position, jump and 10-90 width'],archetype='image plate + quant',backend='Python matplotlib',export='PNG 350dpi PDF SVG; Times New Roman; Cartesian blue, TAOCA red',integrity='No field smoothing; validation only; same physical masks; profiles with multiple waves rejected from scalar shock metrics; gradient localization is diagnostic and uses reference labels, never prediction inputs'))
    plt.rcParams.update({'font.family':'Times New Roman','font.size':8,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    torch.set_num_threads(4);meta=json.loads((ROOT/'datasets/joint_nearest/manifest.json').read_text());mean=np.array(meta['mean']);std=np.array(meta['std']);heights=json.loads((ROOT/'outputs/atlas_v3/taoca/run.json').read_text())['heights']
    ds=AllocatedWindows(ROOT/'datasets/joint_nearest',ROOT/'datasets/fenli_paired','taoca','val',heights=heights)
    model=AllocatedPredictor(ROOT/'outputs/shared_trajectory/model.json',heights).cuda().eval();model.load_state_dict(torch.load(ROOT/'outputs/metric_v8/taoca/best.pt',map_location='cuda',weights_only=True))
    from taoca.spatial_physics import SpatialPredictor
    cart=SpatialPredictor(ROOT/'outputs/shared_trajectory/model.json').cuda().eval();cart.load_state_dict(torch.load(ROOT/'outputs/physics_v2/cartesian/best.pt',map_location='cuda',weights_only=True))
    cds=AllocatedWindows(ROOT/'datasets/joint_nearest',ROOT/'datasets/fenli_paired','cartesian','val',heights=heights)
    yy,xx=np.meshgrid(np.linspace(-20,50,280),np.linspace(-10,100,440),indexing='ij');fixed=np.stack([xx,yy],-1);dx=110/439;dy=70/279;acc={};profile_rows=[];frame_rows=[]
    for index in range(len(ds)):
        item=ds[index];batch={k:v[None].cuda() if torch.is_tensor(v) else v for k,v in item.items()};ci=cds[index];cb={k:v[None].cuda() if torch.is_tensor(v) else v for k,v in ci.items()}
        with torch.no_grad():
            args=['history','mach','pose_history','dt','last_xy','templates'];out=model(*[batch[k] for k in args],10,'taoca',batch['last_base'],batch['last_valid']);co=cart(*[cb[k] for k in args],10,'cartesian',cb['last_base'],cb['last_valid'])
            sampled=sample_grid(batch['target'],out['xy']);tv=sample_grid(batch['target_valid'][:,:,None],out['xy'])>.999
        near,shock=regional_maps(item['target'][None],item['target_valid'][None]);case=item['case'];start=int(item['start'])
        for t in [0,4,9]:
            frame=start+t;path=ROOT/'datasets/fenli_paired'/case/f'{frame:04d}.h5'
            with h5py.File(path) as f:
                assert abs(float(f.attrs['time'])-meta['cases'][case]['times'][frame])<1e-7
                src={k:f['source'][k][:] for k in ['x','y','triangles','fields']}
            xy=[out['xy'][0,t,k,:h].cpu().numpy() for k,h in enumerate(heights)];mask=[out['geometry'][0,t,k,0,:h].cpu().numpy()>.5 for k,h in enumerate(heights)]
            direct=source_sample(src,xy+[fixed],mean,std);del src
            maps=build_maps(xy,mask,polygons(item['templates'].numpy(),out['pose'][0,t].cpu().numpy()))
            legal=[(tv[0,t,k,0,:h].cpu().numpy()&direct[k][1]) for k,h in enumerate(heights)]
            cv=[sampled[0,t,k,:,:h].cpu().numpy() for k,h in enumerate(heights)];dv=[d[0] for d in direct[:4]]
            cr,cq=apply_maps(cv,maps,legal);dr,dq=apply_maps(dv,maps,legal);cascade,cs=fuse(cr,cq);rawrt,rs=fuse(dr,dq)
            pv=[out['fields'][0,t,k,:,:h].cpu().numpy() for k,h in enumerate(heights)];pr,pq=apply_maps(pv,maps);pred,ps=fuse(pr,pq)
            ref=item['target'][t].numpy();raw,rawvalid=direct[-1];cp=co['fields'][0,t,0].cpu().numpy();valid=item['target_valid'][t].numpy()>.5
            common=valid&rawvalid&cs&rs&ps&np.isfinite(cp).all(0)
            regions={'all':common,'wall':common&(near[0,t,0].numpy()>.5),'shock':common&(shock[0,t,0].numpy()>.5)}
            values={'Raw CFD P1':raw,'Cartesian truth':ref,'TAOCA cascade RT':cascade,'TAOCA direct RT':rawrt,'FNO_Cartesian':cp,'FNO_TAOCA_V8':pred}
            for region,use in regions.items():
                for name,value in values.items():
                    accumulate(acc,name+'/vs_raw/'+region,value,raw,use);accumulate(acc,name+'/vs_dataset/'+region,value,ref,use)
            gy,gx=np.gradient(raw[0],dy,dx);mag=np.hypot(gx,gy);allowed=binary_erosion(common,iterations=14)&(distance_transform_edt(valid)>8);score=np.where(allowed,mag,0);chosen=[]
            for _ in range(3):
                iy,ix=np.unravel_index(score.argmax(),score.shape)
                if score[iy,ix]<.02:break
                chosen.append((iy,ix));score[(xx-xx[iy,ix])**2+(yy-yy[iy,ix])**2<36]=0
            for point,(iy,ix) in enumerate(chosen):
                direction=np.array([gx[iy,ix],gy[iy,ix]])/max(mag[iy,ix],1e-12);distance=np.linspace(-3,3,121);line=fixed[iy,ix]+distance[:,None]*direction;coords=np.stack([(line[:,1]+20)/dy,(line[:,0]+10)/dx]);linevalid=map_coordinates(common.astype('f4'),coords,order=1,mode='constant',cval=0)>.999
                if not linevalid.all():continue
                profiles={name:np.array([map_coordinates(np.nan_to_num(v[j]),coords,order=1,mode='constant',cval=np.nan) for j in range(5)]) for name,v in values.items()}
                tag=f'{case}_s{start:03d}_lead{t+1:02d}_p{point}';data={'distance_m':distance,'x_m':line[:,0],'y_m':line[:,1]}
                fig,axes=plt.subplots(2,2,figsize=(7.2,5.3),layout='constrained')
                for j in [0,1]:
                    for name,profile in profiles.items():
                        physical=profile[j]*std[j]+mean[j];data[name+'_'+FIELDS[j]]=physical;metric=profile_metrics(distance,physical)
                        profile_rows.append(dict(case=case,start=start,lead=t+1,profile=point,field=FIELDS[j],method=name,**metric))
                        ax=axes[j,0 if name in ['Raw CFD P1','Cartesian truth','TAOCA cascade RT','TAOCA direct RT'] else 1]
                        if name=='Raw CFD P1':
                            axes[j,1].plot(distance,physical,color=COLORS[name],lw=1.4,label=name)
                        ax.plot(distance,physical,color=COLORS[name],ls='--' if name=='FNO_Cartesian' else '-',lw=1.1,alpha=.9,label=name)
                    for ax in axes[j]:ax.set(xlabel='Distance along reference normal (m)',ylabel=FIELDS[j]+(' (Pa)' if j==0 else ' (kg/m3)'));ax.legend(fontsize=5.5)
                axes[0,0].set_title('Representation / mapping');axes[0,1].set_title('Neural prediction');fig.suptitle(tag+'; identical physical profile',fontsize=9);export(fig,tag)
                with (OUT/'profiles'/(tag+'.csv')).open('w',newline='',encoding='utf-8-sig') as f:
                    writer=csv.writer(f);writer.writerow(data);writer.writerows(zip(*data.values()))
            fig,ax=plt.subplots(figsize=(7.2,4),layout='constrained');pressure=np.where(common,raw[0]*std[0]+mean[0],np.nan);lo,hi=np.nanquantile(pressure,[.01,.99]);cmap=LinearSegmentedColormap.from_list('Small Rainbow',['blue','cyan','lime','yellow','red'])
            im=ax.imshow(pressure,origin='lower',extent=[-10,100,-20,50],cmap=cmap,vmin=lo,vmax=hi,interpolation='nearest')
            for point,(iy,ix) in enumerate(chosen):
                direction=np.array([gx[iy,ix],gy[iy,ix]])/max(mag[iy,ix],1e-12);line=fixed[iy,ix]+np.array([-3,3])[:,None]*direction;ax.plot(line[:,0],line[:,1],color='black',lw=1);ax.text(xx[iy,ix],yy[iy,ix]+1,str(point),fontsize=8)
            ax.set(title=f'{case}, lead {t+1}: reference-normal profile locations',xlabel='x (m)',ylabel='y (m)');fig.colorbar(im,ax=ax,extend='both',label='Raw CFD P1 pressure (Pa); reference 1-99%');export(fig,f'{case}_s{start:03d}_lead{t+1:02d}_locations')
            frame_rows.append(dict(case=case,start=start,lead=t+1,frame=frame,common_count=int(common.sum()),profiles=len(chosen)))
            dump('status.json',dict(state='running',completed_frames=len(frame_rows),total_frames=len(ds)*3));print(json.dumps(frame_rows[-1]),flush=True)
        del item,batch,out,cb,co;gc.collect()
    metrics={k:dict(n=v['n'],nrmse=np.sqrt(v['sse']/max(v['n'],1)).tolist()) for k,v in acc.items()};dump('paired_metrics.json',metrics);dump('profile_metrics.json',profile_rows);dump('frames.json',frame_rows)
    a=metrics['TAOCA cascade RT/vs_raw/shock']['nrmse'];b=metrics['TAOCA direct RT/vs_raw/shock']['nrmse'];gain=1-np.mean(b[:2])/np.mean(a[:2])
    summary=[]
    for field in FIELDS[:2]:
        for name in COLORS:
            pairs=[]
            for r in profile_rows:
                if r['field']!=field or r['method']!=name or not r['valid']:continue
                ref=next((v for v in profile_rows if all(v[k]==r[k] for k in ['case','start','lead','profile','field']) and v['method']=='Raw CFD P1'),None)
                if ref and ref['valid']:pairs.append([abs(r['position']-ref['position']),abs(r['jump']-ref['jump'])/max(abs(ref['jump']),1e-12),r['width']-ref['width']])
            if pairs:summary.append(dict(field=field,method=name,n=len(pairs),mean_position_error_m=float(np.mean(np.array(pairs)[:,0])),mean_relative_jump_error=float(np.mean(np.array(pairs)[:,1])),mean_width_bias_m=float(np.mean(np.array(pairs)[:,2]))))
    dump('profile_summary.json',summary)
    paired=[]
    for field in FIELDS[:2]:
        anchors=[r for r in profile_rows if r['field']==field and r['method']=='Raw CFD P1' and r['valid']]
        groups=[]
        for anchor in anchors:
            group={r['method']:r for r in profile_rows if all(r[k]==anchor[k] for k in ['case','start','lead','profile','field'])}
            if all(group.get(name,{}).get('valid',False) for name in ['FNO_Cartesian','FNO_TAOCA_V8']):groups.append(group)
        for name in ['FNO_Cartesian','FNO_TAOCA_V8']:
            if not groups:continue
            errors=np.array([[abs(g[name]['position']-g['Raw CFD P1']['position']),abs(g[name]['jump']-g['Raw CFD P1']['jump'])/max(abs(g['Raw CFD P1']['jump']),1e-12),g[name]['width']-g['Raw CFD P1']['width']] for g in groups])
            paired.append(dict(field=field,method=name,n=len(groups),position_mae_m=float(errors[:,0].mean()),relative_jump_mae=float(errors[:,1].mean()),width_bias_m=float(errors[:,2].mean())))
    dump('paired_neural_profile_summary.json',paired)
    lines=['# 激波误差定位与原始网格直接采样诊断','',f'验证集{len(ds)}个窗口，固定lead 1、5、10，共{len(frame_rows)}帧；没有访问测试集或训练新网络。','',f'相对于相同原始CFD P1物理网格参考，直接采样TAOCA的高梯度区域压力/密度平均往返NRMSE变化：下降{100*gain:.2f}%。','', '|变量|Cartesian再采样往返|原始网格直接采样往返|','|---|---:|---:|']
    for j in range(5):lines.append(f'|{FIELDS[j]}|{a[j]:.6f}|{b[j]:.6f}|')
    lines+=['','两种往返使用同一组预测几何、同一合法供体及共同支持域。直接采样使用原始快照的节点变量和原始三角形P1插值；目标时刻的真实场仅用于离线诊断，不能作为模型输入。','', '## 激波剖面','位置采用最大绝对梯度位置；强度采用剖面两端各12点中位数之差；宽度采用对应跃迁10%至90%的交点间距。多波、缺失交点和边界峰值被标记无效。剖面重采样仅用于测量，不对场做平滑。样本来自参考压力梯度较强且远离固体和支持边界的位置，不代表所有激波。','', '|变量|方法|有效剖面数|位置MAE/m|相对跃迁误差|宽度偏差/m|','|---|---|---:|---:|---:|---:|']
    for r in summary:lines.append(f"|{r['field']}|{r['method']}|{r['n']}|{r['mean_position_error_m']:.4f}|{r['mean_relative_jump_error']:.4f}|{r['mean_width_bias_m']:.4f}|")
    lines+=['','各方法通过有效性检查的剖面数量可能不同，以上均值用于定位问题，不作为不同方法的严格排名；原始CSV和有效性标记已保留。峰值位置在多峰条件下不稳定，不能将单个剖面结果当作全域激波位置误差。','', '## 下一步判断',('直接采样显示降低映射损失的潜力；应先建立同源双路线数据小样本并复核全域、近壁和覆盖，再训练。' if gain>=.05 else '直接采样尚未显示足够的高梯度压力/密度平均收益，不应立即重建全量数据；优先根据剖面检查网络位置与强度误差。'),'诊断不自动授权使用真实未来几何，不修改现有模型或基线。P1原始参考自身仍受源网格分辨率限制，不能视为解析真解。']
    images=sorted((OUT/'figures').glob('*.png'))
    for p in images[:5]:lines+=['',f'![{p.stem}](figures/{p.name})']
    (OUT/'诊断报告.md').write_text('\n'.join(lines),encoding='utf-8');dump('status.json',dict(state='complete',frames=len(frame_rows),profile_records=len(profile_rows),direct_sampling_pressure_density_gain=float(gain)))
if __name__=='__main__':
    try:main()
    except Exception:
        OUT.mkdir(parents=True,exist_ok=True);dump('status.json',dict(state='failed',error=traceback.format_exc()));raise

