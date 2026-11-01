"""Validation-only dense/stencil gap and train-gradient conflict diagnostics."""
import os
os.environ['MKL_THREADING_LAYER']='GNU'
import sys,json,time,io,hashlib,traceback
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from taoca.allocated_data_v3 import AllocatedWindows
from taoca.allocated_model_v3 import AllocatedPredictor
from taoca.spatial_physics import SpatialPredictor
from taoca.chart_transfer_v3 import build_maps,apply_maps,fuse
from taoca.physics_evaluate import polygons
from taoca.physics_losses import regional_maps
from taoca.physical_supervision_v7 import load_window,project
OUT=ROOT/'outputs/diagnosis_v8'
def snapshot(path):
    for attempt in range(4):
        try:
            raw=path.read_bytes();state=torch.load(io.BytesIO(raw),map_location='cuda',weights_only=True);return state,hashlib.sha256(raw).hexdigest()
        except (RuntimeError,EOFError):
            if attempt==3:raise
            time.sleep(1)
def add(acc,key,pred,ref,mask):
    g=acc.setdefault(key,dict(n=0,sse=np.zeros(5)));g['n']+=int(mask.sum());g['sse']+=np.square(pred[:,mask].astype('f8')-ref[:,mask]).sum(1)
def forward(model,b,route):return model(*[b[k] for k in ['history','mach','pose_history','dt','last_xy','templates']],10,route,b['last_base'],b['last_valid'])
def main():
    OUT.mkdir(exist_ok=True);torch.set_num_threads(2);heights=[35,35,140,70];model=AllocatedPredictor(ROOT/'outputs/shared_trajectory/model.json',heights).cuda().eval();state,digest=snapshot(ROOT/'outputs/physical_v7/taoca/best.pt');model.load_state_dict(state);del state
    cart=SpatialPredictor(ROOT/'outputs/shared_trajectory/model.json').cuda().eval();cart.load_state_dict(torch.load(ROOT/'outputs/physics_v2/cartesian/best.pt',map_location='cuda',weights_only=True))
    args=[ROOT/'datasets/joint_nearest',ROOT/'datasets/fenli_paired'];ds=AllocatedWindows(*args,'taoca','val',heights=heights);cds=AllocatedWindows(*args,'cartesian','val',heights=heights);totals={};pose_error=0.
    yy,xx=np.meshgrid(np.arange(280),np.arange(440),indexing='ij')
    for i in range(len(ds)):
        item=ds[i];ci=cds[i];b={k:v[None].cuda() if torch.is_tensor(v) else v for k,v in item.items()};cb={k:v[None].cuda() if torch.is_tensor(v) else v for k,v in ci.items()}
        with torch.inference_mode():out=forward(model,b,'taoca');co=forward(cart,cb,'cartesian')
        pose_error=max(pose_error,float((out['pose']-co['pose']).abs().max()));near,shock=regional_maps(item['target'][None],item['target_valid'][None])
        for t in range(10):
            xy=[out['xy'][0,t,k,:h].cpu().numpy() for k,h in enumerate(heights)];mask=[out['geometry'][0,t,k,0,:h].cpu().numpy()>.5 for k,h in enumerate(heights)];values=[out['fields'][0,t,k,:,:h].cpu().numpy() for k,h in enumerate(heights)]
            maps=build_maps(xy,mask,polygons(item['templates'].numpy(),out['pose'][0,t].cpu().numpy()));rec,q=apply_maps(values,maps);pred,support=fuse(rec,q);cp=co['fields'][0,t,0].cpu().numpy();ref=item['target'][t].numpy();common=(item['target_valid'][t].numpy()>.5)&support&np.isfinite(cp).all(0)
            lattice=(yy%4==t%4)&(xx%4==(t//4)%4)
            for subset,sub in [('dense',common),('training_lattice',common&lattice),('unseen_lattice',common&~lattice)]:
                for region,rm in [('all',np.ones_like(common)),('shock',shock[0,t,0].numpy()>.5),('wall',near[0,t,0].numpy()>.5)]:
                    for name,values in [('FNO_TAOCA_V7',pred),('FNO_Cartesian',cp)]:add(totals,name+'/'+subset+'/'+region,values,ref,sub&rm)
        print(json.dumps(dict(validation_window=i+1,total=len(ds))),flush=True)
    metrics={k:dict(n=v['n'],nrmse=np.sqrt(v['sse']/max(1,v['n'])).tolist()) for k,v in totals.items()};(OUT/'validation_dense_metrics.json').write_text(json.dumps(dict(checkpoint_sha256=digest,scope='Validation only, interim V7 checkpoint; never test data; no network weights updated',pose_max_difference=pose_error,metrics=metrics),indent=2))
    # Use training data only to diagnose conflict among losses; do not update any model.
    del cart,co;train=AllocatedWindows(*args,'taoca','train',heights=heights);item=train[0];b={k:v[None].cuda() if torch.is_tensor(v) else v for k,v in item.items()};ops=load_window(ROOT/'outputs/physical_v7/operators',item['case'],item['start']);out=forward(model,b,'taoca');near,shock=regional_maps(b['target'],b['target_valid']);objectives=[];channel=out['fields'].new_zeros(5);edge=out['fields'].sum()*0
    for t,op in enumerate(ops):
        q=torch.as_tensor(op['q'],device='cuda',dtype=torch.long);values=project(out['fields'][0,t],op);ref=b['target'][0,t].reshape(5,-1)[:,q];valid=b['target_valid'][0,t].reshape(-1)[q]>.5;sm=shock[0,t,0].reshape(-1)[q]>.5;err=(values-ref).square();channel=channel+err[:,valid].mean(1)/10
        if (valid&sm).any():edge=edge+err[:2,valid&sm].mean()/10
    objectives=list(channel.unbind())+[edge];params=[p for p in model.parameters() if p.requires_grad];grad=[]
    for loss in objectives:
        g=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True);grad.append(torch.cat([(a if a is not None else torch.zeros_like(p)).reshape(-1) for a,p in zip(g,params)]).detach())
    vectors=torch.stack(grad);norm=vectors.norm(dim=1).clamp_min(1e-12);cos=(vectors@vectors.T/norm[:,None]/norm[None,:]).cpu().numpy();labels=['p_global','rho_global','T_global','u_global','v_global','p_rho_shock'];(OUT/'gradient_conflict.json').write_text(json.dumps(dict(checkpoint_sha256=digest,training_window=f'{item["case"]}_s{item["start"]:03d}',labels=labels,cosine=cos.tolist(),scope='Single training window diagnostic, not causal proof or a full training gradient distribution; no optimizer step'),indent=2))
    ratios={s:(np.array(metrics['FNO_TAOCA_V7/'+s+'/all']['nrmse'])/np.array(metrics['FNO_Cartesian/'+s+'/all']['nrmse'])).tolist() for s in ['dense','training_lattice','unseen_lattice']}
    lines=['# V7验证诊断：稀疏监督与通道梯度','', '本报告只使用验证集与一个训练窗口；不读取测试集，不更新模型。当前检查点为训练中间版本，不能当作最终结果。','', '|变量|稠密V7/Cartesian|训练格点比值|未训练格点比值|','|---|---:|---:|---:|']
    for j,name in enumerate(['p','rho','T','u','v']):lines.append(f'|{name}|{ratios["dense"][j]:.4f}|{ratios["training_lattice"][j]:.4f}|{ratios["unseen_lattice"][j]:.4f}|')
    lines+=['','数值小于1表示低于固定Cartesian误差。训练格点与其余格点的差异用于检测固定稀疏格点偏差；两者本身物理分布略有不同，所以同时用Cartesian作对照。','',f'高梯度压力/密度损失梯度与全域温度梯度的余弦：{cos[5,2]:.4f}；与全域Y速度：{cos[5,4]:.4f}。负值表示在该训练窗口存在一阶梯度方向冲突，不代表完整训练都冲突。','', '下一步候选应根据这里的证据区分：固定格点过拟合需要更完整/变化的物理监督；通道冲突需要平衡各通道目标。不得依据测试结果筛选具体权重。']
    (OUT/'诊断报告.md').write_text('\n'.join(lines),encoding='utf-8');print(json.dumps(dict(state='complete',ratios=ratios,shock_gradient_cos_T=float(cos[5,2]),shock_gradient_cos_v=float(cos[5,4]))),flush=True)
if __name__=='__main__':
    try:main()
    except Exception:
        OUT.mkdir(exist_ok=True);(OUT/'error.json').write_text(json.dumps(dict(error=traceback.format_exc())));raise
