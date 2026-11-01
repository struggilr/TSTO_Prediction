"""Final V9 dense validation audit. No test field/geometry reads; no weight updates."""
import os
os.environ['MKL_THREADING_LAYER']='GNU'
import sys,json,hashlib,datetime,traceback,gc
from pathlib import Path
import numpy as np
import h5py
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from taoca.allocated_data_v3 import AllocatedWindows
from taoca.local_detail_v9 import LocalPredictor
from taoca.spatial_physics import SpatialPredictor
from taoca.domain_charts import repair_charts
from taoca.chart_transfer_v3 import build_maps,apply_maps,fuse
from taoca.physics_evaluate import polygons
from taoca.physics_losses import regional_maps
OUT=ROOT/'outputs/spacetime_v10/diagnosis'
METHODS=['FNO_Cartesian','FNO_TAOCA_V9'];FIELDS=['Pressure','Density','Temperature','VelocityX','VelocityY']

class ValidationOnlyWindows(AllocatedWindows):
    """Preserve exact validation slices while never loading post-validation payload."""
    _validation_records={}
    def record(self,case):
        n=len(self.meta['cases'][case]['times']);end=round(.7*n)+round(.1*n);key=(str(self.root.resolve()),case,end)
        if key not in self._validation_records:
            with h5py.File(self.root/(case+'.h5')) as f:
                fields=f['fields'][:end];valid=f['valid'][:end].astype('f4');pose=f['pose'][:end].astype('f4');times=f['time'][:end];templates=f['templates'][:].astype('f4')
            fields=(fields-self.mean)/self.std;fields*=valid[:,None]
            self._validation_records[key]=dict(fields=torch.from_numpy(fields),valid=torch.from_numpy(valid),pose=torch.from_numpy(pose),time=times,templates=torch.from_numpy(templates))
        r=self._validation_records[key]
        if self.route=='taoca' and 'coords' not in r:
            with h5py.File(self.root/(case+'_geometry.h5')) as f:xy=torch.from_numpy(f['xy'][:end])
            n,k,c,h,w=xy.shape
            coords=F.interpolate(xy.reshape(n*k,c,h,w),size=(70,440),mode='bilinear',align_corners=True).reshape(n,k,2,70,440).permute(0,1,3,4,2).contiguous()
            r['coords']=repair_charts(coords)
        return r

def dump(name,data):
    OUT.mkdir(parents=True,exist_ok=True);p=OUT/name;tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(p)
def status(state,**kw):dump('status.json',dict(state=state,pid=os.getpid(),updated=datetime.datetime.now().isoformat(),**kw))
def stat(pred,ref,mask):
    n=int(mask.sum());err=pred[:,mask].astype('f8')-ref[:,mask].astype('f8')
    return dict(n=n,sse=np.square(err).sum(1).tolist())
def add(acc,key,row):
    g=acc.setdefault(key,dict(n=0,sse=np.zeros(5)));g['n']+=row['n'];g['sse']+=row['sse']
def finish(acc):return {key:dict(n=g['n'],sse=g['sse'].tolist(),nrmse=np.sqrt(g['sse']/max(1,g['n'])).tolist()) for key,g in acc.items()}
def predict(model,b,route):return model(*[b[k] for k in ['history','mach','pose_history','dt','last_xy','templates']],10,route,b['last_base'],b['last_valid'])
def geometry(pose):
    bodies=np.asarray(pose).reshape(-1,2,3);delta=bodies[:,1,:2]-bodies[:,0,:2]
    return dict(center_separation_m=np.linalg.norm(delta,axis=-1).tolist(),relative_displacement_m=delta.tolist(),body_angles_rad=bodies[:,:,2].tolist(),relative_angle_rad=(bodies[:,1,2]-bodies[:,0,2]).tolist())

def main():
    torch.set_num_threads(4);status('loading')
    heights=[35,35,140,70];args=[ROOT/'datasets/joint_nearest',ROOT/'datasets/fenli_paired'];ds=ValidationOnlyWindows(*args,'taoca','val',heights=heights);cds=ValidationOnlyWindows(*args,'cartesian','val',heights=heights)
    assert len(ds)==len(cds)==5 and ds.samples==cds.samples
    final_status=json.loads((ROOT/'outputs/local_v9/taoca/status.json').read_text());assert final_status['state']=='complete'
    path=ROOT/'outputs/local_v9/taoca/best.pt';digest=hashlib.sha256(path.read_bytes()).hexdigest();cartpath=ROOT/'outputs/physics_v2/cartesian/best.pt';cartdigest=hashlib.sha256(cartpath.read_bytes()).hexdigest();run=json.loads((ROOT/'outputs/local_v9/taoca/run.json').read_text());assert cartdigest==run['cartesian_checkpoint_sha256']
    model=LocalPredictor(ROOT/'outputs/shared_trajectory/model.json',heights).cuda().eval();model.load_state_dict(torch.load(path,map_location='cuda',weights_only=True))
    cart=SpatialPredictor(ROOT/'outputs/shared_trajectory/model.json').cuda().eval();cart.load_state_dict(torch.load(cartpath,map_location='cuda',weights_only=True))
    yy,xx=np.meshgrid(np.arange(280),np.arange(440),indexing='ij');totals={};by_mach={};by_lead={};frames=[];geometries=[];pose_error=0.;ref_count=0;common_count=0
    for i in range(len(ds)):
        item=ds[i];ci=cds[i];assert item['case']==ci['case'] and item['start']==ci['start'];assert torch.equal(item['target'],ci['target']) and torch.equal(item['target_valid'],ci['target_valid'])
        b={k:v[None].cuda() if torch.is_tensor(v) else v for k,v in item.items()};cb={k:v[None].cuda() if torch.is_tensor(v) else v for k,v in ci.items()}
        with torch.inference_mode():out=predict(model,b,'taoca');co=predict(cart,cb,'cartesian')
        delta=float((out['pose']-co['pose']).abs().max());pose_error=max(pose_error,delta);assert delta==0
        near,shock=regional_maps(item['target'][None],item['target_valid'][None]);case=item['case'];start=int(item['start']);n=len(ds.meta['cases'][case]['times']);valend=round(.7*n)+round(.1*n);assert start+10<=valend
        geometries.append(dict(case=case,start=start,history_indices=[start-20,start],target_indices=[start,start+10],predicted_pose=out['pose'][0].cpu().numpy().tolist(),target_pose=item['target_pose'].numpy().tolist(),last_observed_pose=item['pose_history'][-1].numpy().tolist(),predicted_geometry=geometry(out['pose'][0].cpu().numpy()),reference_geometry=geometry(item['target_pose'].numpy()),history_geometry=geometry(item['pose_history'].numpy()),note='Reference target pose is diagnostic only, never an input; centroid distances are not minimum surface gaps'))
        for t in range(10):
            xy=[out['xy'][0,t,k,:h].cpu().numpy() for k,h in enumerate(heights)];mask=[out['geometry'][0,t,k,0,:h].cpu().numpy()>.5 for k,h in enumerate(heights)];values=[out['fields'][0,t,k,:,:h].cpu().numpy() for k,h in enumerate(heights)]
            maps=build_maps(xy,mask,polygons(item['templates'].numpy(),out['pose'][0,t].cpu().numpy()));rec,q=apply_maps(values,maps);pred,support=fuse(rec,q);cp=co['fields'][0,t,0].cpu().numpy();ref=item['target'][t].numpy();valid=item['target_valid'][t].numpy()>.5;common=valid&support&np.isfinite(pred).all(0)&np.isfinite(cp).all(0);ref_count+=int(valid.sum());common_count+=int(common.sum())
            lattice=(yy%4==t%4)&(xx%4==(t//4)%4);row=dict(case=case,start=start,lead=t+1,target_index=start+t,reference_count=int(valid.sum()),common_count=int(common.sum()),statistics={})
            for subset,sub in [('dense',common),('training_lattice',common&lattice),('unseen_lattice',common&~lattice)]:
                for region,rm in [('all',np.ones_like(common)),('shock',shock[0,t,0].numpy()>.5),('wall',near[0,t,0].numpy()>.5)]:
                    for name,value in [('FNO_TAOCA_V9',pred),('FNO_Cartesian',cp)]:
                        key=name+'/'+subset+'/'+region;s=stat(value,ref,sub&rm);row['statistics'][key]=s;add(totals,key,s);add(by_mach,case+'/'+key,s);add(by_lead,str(t+1)+'/'+key,s)
            frames.append(row)
        status('running',completed_windows=i+1,total_windows=len(ds));print(json.dumps(dict(completed_validation_window=i+1,total=5,case=case)),flush=True)
        del b,cb,out,co,item,ci;gc.collect()
    metrics=finish(totals);ratios={subset:(np.array(metrics['FNO_TAOCA_V9/'+subset+'/all']['nrmse'])/np.array(metrics['FNO_Cartesian/'+subset+'/all']['nrmse'])).tolist() for subset in ['dense','training_lattice','unseen_lattice']}
    result=dict(checkpoint_sha256=digest,cartesian_checkpoint_sha256=cartdigest,fields=FIELDS,validation_windows=5,forecast_instances=len(frames),common_coverage=common_count/ref_count,pose_max_difference=pose_error,metrics=metrics,by_mach=finish(by_mach),by_lead=finish(by_lead),global_ratios=ratios,scope='Final validation-selected V9 best only. Validation payload sliced before test begins. No test values loaded, no training/optimizer updates. Dense physical P1 inverse and quality fusion; fixed stencil phase matches V9 supervision. Ratios are model-relative comparisons, not proof of temporal distribution shift or causal stencil bias.')
    dump('validation_dense_metrics.json',result);dump('frame_metrics.json',frames);dump('geometry_strata.json',geometries)
    lines=['# 最终V9验证集稠密复核','', '仅使用五个验证窗口的50个预测实例。载入H5时已显式截断至验证区间结束，不读取测试区间字段或网格。参考未来位姿仅保存作离线诊断，不进入模型。','', '|变量|稠密V9/Cartesian|固定监督格点|其他格点|','|---|---:|---:|---:|']
    for j,name in enumerate(FIELDS):lines.append(f'|{name}|{ratios["dense"][j]:.6f}|{ratios["training_lattice"][j]:.6f}|{ratios["unseen_lattice"][j]:.6f}|')
    lines+=['','小于1表示V9误差更低。全部SSE、计数、按Mach及预测步长聚合数据已保存；几何摘要使用位姿中心距离，不是最小物面间距。五条轨迹不构成独立新工况，不能将同一序列内帧视作独立重复。']
    (OUT/'最终V9稠密验证诊断.md').write_text('\n'.join(lines),encoding='utf-8');status('complete',validation_windows=5,forecast_instances=len(frames),global_ratios=ratios);print(json.dumps(dict(state='complete',global_ratios=ratios)),flush=True)
if __name__=='__main__':
    try:main()
    except Exception:status('failed',error=traceback.format_exc());raise
