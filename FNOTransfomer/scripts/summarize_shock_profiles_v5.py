"""Local single-jump measurements from existing CSVs; never reruns a network."""
import json,csv,time,traceback
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/shock_profiles_v5'
METHODS=['Cartesian truth','TAOCA cascade RT','TAOCA direct RT','FNO_Cartesian','FNO_TAOCA']
def crossings(x,z,level,center):
    ids=np.flatnonzero((z[:-1]-level)*(z[1:]-level)<=0);ids=ids[np.abs(np.diff(z)[ids])>1e-12]
    values=[x[i]+(level-z[i])/(z[i+1]-z[i])*(x[i+1]-x[i]) for i in ids]
    return min(values,key=lambda v:abs(v-center)) if values else None
def measure(x,y):
    grad=np.gradient(y,x);i=int(np.argmax(abs(grad)));jump=float(np.median(y[-3:])-np.median(y[:3]));result=dict(position=float(x[i]),jump=jump,width=None)
    if abs(jump)<1e-10 or i<2 or i>=len(x)-2:return result
    z=(y-np.median(y[:3]))/jump;a=crossings(x,z,.1,x[i]);b=crossings(x,z,.9,x[i])
    if a is not None and b is not None and b>a and abs(np.diff(z)).sum()<2.5:result['width']=float(b-a)
    return result
def main():
    while True:
        p=OUT/'status.json'
        if p.exists():
            s=json.loads(p.read_text())
            if s['state']=='failed':raise RuntimeError('Primary diagnosis failed')
            if s['state']=='complete':break
        time.sleep(10)
    meta=json.loads((ROOT/'datasets/joint_nearest/manifest.json').read_text());records=[]
    for path in sorted((OUT/'profiles').glob('*.csv')):
        with path.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
        x=np.array([float(r['distance_m']) for r in rows])
        for field,std in zip(['Pressure','Density'],meta['std'][:2]):
            ref=np.array([float(r['Raw CFD P1_'+field]) for r in rows]);g=np.gradient(ref,x);i=int(np.argmax(abs(g)));sign=np.sign(g[i]);left=right=i
            while left>1 and g[left-1]*sign>.1*abs(g[i]):left-=1
            while right<len(x)-2 and g[right+1]*sign>.1*abs(g[i]):right+=1
            left=max(0,left-5);right=min(len(x)-1,right+5);xs=x[left:right+1];ref=ref[left:right+1];rm=measure(xs,ref)
            for method in METHODS:
                y=np.array([float(r[method+'_'+field]) for r in rows])[left:right+1];m=measure(xs,y)
                records.append(dict(profile=path.stem,field=field,method=method,nrmse=float(np.sqrt(np.mean((y-ref)**2))/std),position_error_m=abs(m['position']-rm['position']),relative_jump_error=abs(m['jump']-rm['jump'])/max(abs(rm['jump']),1e-12),width_error_m=None if m['width'] is None or rm['width'] is None else m['width']-rm['width'],segment_start_m=float(xs[0]),segment_end_m=float(xs[-1])))
    (OUT/'local_shock_metrics.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    summary=[]
    for field in ['Pressure','Density']:
        for method in METHODS:
            r=[v for v in records if v['field']==field and v['method']==method];width=[v['width_error_m'] for v in r if v['width_error_m'] is not None]
            if r:summary.append(dict(field=field,method=method,n=len(r),mean_profile_nrmse=float(np.mean([v['nrmse'] for v in r])),median_position_error_m=float(np.median([v['position_error_m'] for v in r])),median_relative_jump_error=float(np.median([v['relative_jump_error'] for v in r])),valid_width_count=len(width),median_width_bias_m=float(np.median(width)) if width else None))
    (OUT/'local_shock_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    plt.rcParams.update({'font.family':'Times New Roman','font.size':8,'svg.fonttype':'none','pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(7.2,3),layout='constrained');colors=['#888888','#007f7f','#d9a000','#0000ff','#ff0000']
    for ax,field in zip(axes,['Pressure','Density']):
        values=[next(v for v in summary if v['field']==field and v['method']==m)['mean_profile_nrmse'] for m in METHODS];ax.bar(range(5),values,color=colors,alpha=.8);ax.set_xticks(range(5),METHODS,rotation=35,ha='right',fontsize=6);ax.set(title=field,ylabel='Mean local-profile NRMSE')
    for suffix in ['png','pdf','svg']:fig.savefig(OUT/'figures'/('local_profile_summary.'+suffix),dpi=350,bbox_inches='tight')
    plt.close(fig)
    lines=['','## 局部单跃迁段的补充统计','整条剖面可能包含后方膨胀，因此以下用原始参考场梯度峰值附近的同号强梯度段，并向两侧各扩展0.25m。所有方法使用完全相同的截取范围。该范围仅用于离线测量。','位置误差为局部最大绝对梯度位置差；跃迁为局部端点各3点中位数之差。若预测梯度峰值落在段边界，位置结果可能被截断，应结合剖面图解释。宽度只对能识别10%/90%交点的剖面统计，不作为无条件方法排名。','', '|变量|方法|剖面数|平均剖面NRMSE|位置误差中位数/m|相对跃迁误差中位数|有效宽度数|宽度偏差中位数/m|','|---|---|---:|---:|---:|---:|---:|---:|']
    for r in summary:lines.append(f"|{r['field']}|{r['method']}|{r['n']}|{r['mean_profile_nrmse']:.4f}|{r['median_position_error_m']:.3f}|{r['median_relative_jump_error']:.3f}|{r['valid_width_count']}|{r['median_width_bias_m']}|")
    lines+=['','![局部剖面对照](figures/local_profile_summary.png)','以上剖面是高梯度位置的诊断样本，不是独立工况，也不是全域精度统计。']
    with (OUT/'诊断报告.md').open('a',encoding='utf-8') as f:f.write('\n'.join(lines))
    (OUT/'local_summary_status.json').write_text(json.dumps(dict(state='complete',profile_records=len(records))),encoding='utf-8')
if __name__=='__main__':
    try:main()
    except Exception:
        (OUT/'local_summary_status.json').write_text(json.dumps(dict(state='failed',error=traceback.format_exc())),encoding='utf-8');raise
