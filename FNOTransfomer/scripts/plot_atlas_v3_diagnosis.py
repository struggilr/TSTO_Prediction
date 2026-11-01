import os,sys,json
os.environ['MKL_THREADING_LAYER']='GNU'
from pathlib import Path
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));style=(ROOT.parents[1]/"analysis/2026-10-18_TAOCA本次训练综合分析V1.0/期刊风格重绘/publication_style.py");sys.path.insert(0,str(style.parent))
from publication_style import apply_style,small_rainbow,export,BLUE,RED
from taoca.chart_transfer_v3 import build_maps,apply_maps,fuse
from taoca.physics_evaluate import polygons
def main():
    apply_style();out=ROOT/'outputs/atlas_v3';figures=out/'diagnostic_figures';meta=json.loads((ROOT/'datasets/joint_nearest/manifest.json').read_text());d=json.loads((out/'fusion_diagnosis.json').read_text())
    modes=['quality','equal','background_x4','background_x16','finest'];labels=['Quality','Equal','Background x4','Background x16','Max. quality']
    fig,ax=plt.subplots(figsize=(7.2,3),layout='constrained');x=np.arange(5)
    for i,(mode,label) in enumerate(zip(modes,labels)):ax.plot(x,d['metrics'][mode+'/all']['nrmse'],label='FNO_TAOCA: '+label,color=[RED,BLUE,'#007f7f','#d9a000','#666666'][i],ls=['-','--',':','-.','--'][i],marker=['o','s','^','d','x'][i],ms=3,alpha=.85)
    ax.set_xticks(x,['Pressure','Density','Temperature','Velocity X','Velocity Y']);ax.set_ylabel('Validation NRMSE');ax.legend(fontsize=6.5,ncol=2);export(fig,figures,'validation_fusion');plt.close(fig)
    path=sorted((out/'validation_native').glob('*.h5'))[0];t=9
    with h5py.File(path) as f:xy=f['xy'][t];mask=f['mask'][t];pose=f['pose'][t];templates=f['templates'][:];native=f['fields'][t];ref=f['reference'][t];valid=f['valid'][t].astype(bool);case=f.attrs['case'];start=int(f.attrs['start'])
    maps=build_maps(xy,mask,polygons(templates,pose));rec,q=apply_maps(native,maps);combined,support=fuse(rec,q,'quality')
    mean=meta['mean'][0];std=meta['std'][0];pressure=ref[0]*std+mean;lo,hi=np.quantile(pressure[valid],[.01,.99])
    fig,axes=plt.subplots(2,3,figsize=(15,9),layout='constrained');arrays=[np.where(valid,pressure,np.nan)]+[r[0]*std+mean for r in rec]+[combined[0]*std+mean]
    titles=['CFD reference','FNO_TAOCA: body 1 O','FNO_TAOCA: body 2 O','FNO_TAOCA: background','FNO_TAOCA: gap','FNO_TAOCA: quality fusion']
    for i,(ax,a,title) in enumerate(zip(axes.flat,arrays,titles)):
        im=ax.imshow(np.ma.masked_invalid(a),extent=[-10,100,-20,50],origin='lower',interpolation='nearest',cmap=small_rainbow(),vmin=lo,vmax=hi);ax.set(title=title,xlabel='x (m)',ylabel='y (m)');ax.text(-.10,1.05,chr(97+i),transform=ax.transAxes,fontweight='bold')
    fig.colorbar(im,ax=axes,extend='both',shrink=.7,label='Pressure (Pa); reference 1%-99% display range')
    fig.suptitle(f'Validation only: {case}, target start {start}, lead 10\nWhite: outside the displayed chart support / solid. Same scale; saturated tails disclosed.',fontsize=11)
    export(fig,figures,'validation_chart_pressure');plt.close(fig)
    selection=out/'allocation_selection.json'
    if selection.exists():
        s=json.loads(selection.read_text());fig,axes=plt.subplots(1,2,figsize=(7.2,3),layout='constrained')
        for i,(name,m) in enumerate(s['metrics'].items()):
            for ax,region in zip(axes,['all','shock']):ax.plot(range(5),m[region]['nrmse'],marker=['o','s','^','d'][i],ms=3,ls=['-','--',':','-.'][i],color=[BLUE,'#009999','#d9a000',RED][i],alpha=.85,label='FNO_TAOCA: '+name)
        for ax,title in zip(axes,['Full physical support','High-pressure-gradient region']):ax.set_xticks(range(5),['p','rho','T','u','v']);ax.set(title=title,ylabel='Truth roundtrip NRMSE')
        axes[0].legend(fontsize=5.5);export(fig,figures,'validation_allocation_roundtrip');plt.close(fig)
    print('Diagnostic figures exported',flush=True)
if __name__=='__main__':main()
