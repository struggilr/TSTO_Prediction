"""Validation-gated shock fusion optimization on a frozen V3 network."""
import os
os.environ['MKL_THREADING_LAYER']='GNU'
import sys,json,time,traceback,hashlib
from pathlib import Path
import numpy as np
import h5py
import torch
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from taoca.allocated_data_v3 import AllocatedWindows
from taoca.allocated_model_v3 import AllocatedPredictor
from taoca.chart_transfer_v3 import build_maps,apply_maps
from taoca.shock_fusion_v4 import weights_and_fuse,MODES
from taoca.physics_evaluate import polygons
from taoca.physics_losses import regional_maps
from taoca.joint import sample_grid
OUT=ROOT/'outputs/shock_fusion_v4'
def write(name,value):
    p=OUT/name;tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2),encoding='utf-8');tmp.replace(p)
def add(acc,key,pred,ref,mask):
    mask=mask&np.isfinite(pred).all(0);g=acc.setdefault(key,dict(n=0,sse=np.zeros(5)));g['n']+=int(mask.sum());g['sse']+=np.square(pred[:,mask].astype('f8')-ref[:,mask]).sum(1)
def finish(acc):return {k:dict(n=v['n'],nrmse=np.sqrt(v['sse']/max(1,v['n'])).tolist()) for k,v in acc.items()}
def run_split(model,heights,split,modes):
    ds=AllocatedWindows(ROOT/'datasets/joint_nearest',ROOT/'datasets/fenli_paired','taoca',split,heights=heights)
    acc={};instances=0
    for index,batch in enumerate(DataLoader(ds,batch_size=1,num_workers=0)):
        b={k:v.cuda() if torch.is_tensor(v) else v for k,v in batch.items()}
        with torch.no_grad():
            out=model(*[b[k] for k in ['history','mach','pose_history','dt','last_xy','templates']],10,'taoca',b['last_base'],b['last_valid'])
            truth=sample_grid(b['target'],out['xy']) if split=='val' else None
            tv=sample_grid(b['target_valid'][:,:,None],out['xy'])>.999 if split=='val' else None
        near,shock=regional_maps(batch['target'],batch['target_valid'])
        for t in range(10):
            xy=[out['xy'][0,t,k,:h].cpu().numpy() for k,h in enumerate(heights)]
            mask=[out['geometry'][0,t,k,0,:h].cpu().numpy()>.5 for k,h in enumerate(heights)]
            native=[out['fields'][0,t,k,:,:h].cpu().numpy() for k,h in enumerate(heights)]
            maps=build_maps(xy,mask,polygons(batch['templates'][0].numpy(),out['pose'][0,t].cpu().numpy()))
            rec,q=apply_maps(native,maps);ref=batch['target'][0,t].numpy();valid=batch['target_valid'][0,t].numpy()>.5
            regions={'all':valid,'wall':valid&(near[0,t,0].numpy()>.5),'shock':valid&(shock[0,t,0].numpy()>.5)}
            for mode in modes:
                pred,support=weights_and_fuse(rec,q,mode)
                for region,m in regions.items():add(acc,mode+'/'+region,pred,ref,m&support)
            if split=='val':
                values=[truth[0,t,k,:,:h].cpu().numpy() for k,h in enumerate(heights)]
                good=[tv[0,t,k,0,:h].cpu().numpy() for k,h in enumerate(heights)]
                rt,rq=apply_maps(values,maps,good);roundtrip,rs=weights_and_fuse(rt,rq,'quality')
                for region,m in regions.items():add(acc,'truth_roundtrip/'+region,roundtrip,ref,m&rs)
                for k in range(4):
                    for region,m in regions.items():add(acc,f'chart{k}/'+region,rec[k],ref,m)
            instances+=1
        write('status.json',dict(state='running',split=split,windows=index+1,total_windows=len(ds),instances=instances))
        print(json.dumps(dict(split=split,window=index+1,total=len(ds))),flush=True)
    return finish(acc)
def main():
    OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
    contract=dict(candidates=MODES,selection_split='validation only',selection='min mean pressure/density shock NRMSE among candidates reducing it by >=3%, full-domain mean <=1.01 baseline, wall mean <=1.03 baseline, identical support counts',test_policy='Evaluate only selected candidate versus frozen quality baseline; never reselect using test results',notes='Predicted-field gradients only in online fusion. Shared weights for all five channels. Gaussian filtering affects donor-selection weights only, never the predicted field values. No model retraining. Separate-chart errors have different support, not directly comparable.')
    write('contract.json',contract)
    checkpoint=ROOT/'outputs/atlas_v3/taoca/best.pt';digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    heights=json.loads((ROOT/'outputs/atlas_v3/taoca/run.json').read_text())['heights']
    model=AllocatedPredictor(ROOT/'outputs/shared_trajectory/model.json',heights).cuda().eval();model.load_state_dict(torch.load(checkpoint,map_location='cuda',weights_only=True))
    val=run_split(model,heights,'val',MODES);write('validation.json',val)
    base=val['quality/shock']['nrmse'];allowed=[];decisions={}
    for mode in MODES[1:]:
        gain=1-np.mean(val[mode+'/shock']['nrmse'][:2])/np.mean(base[:2])
        global_ratio=np.mean(val[mode+'/all']['nrmse'])/np.mean(val['quality/all']['nrmse'])
        wall_ratio=np.mean(val[mode+'/wall']['nrmse'])/np.mean(val['quality/wall']['nrmse'])
        counts=all(val[mode+'/'+r]['n']==val['quality/'+r]['n'] for r in ['all','wall','shock'])
        passes=bool(gain>=.03 and global_ratio<=1.01 and wall_ratio<=1.03 and counts)
        decisions[mode]=dict(shock_pressure_density_gain=float(gain),global_ratio=float(global_ratio),wall_ratio=float(wall_ratio),identical_counts=counts,passes=passes)
        if passes:allowed.append(mode)
    selected=min(allowed,key=lambda m:np.mean(val[m+'/shock']['nrmse'][:2])) if allowed else 'quality'
    write('selection.json',dict(selected=selected,changed=bool(allowed),decisions=decisions,checkpoint_sha256=digest))
    test=None
    if allowed:test=run_split(model,heights,'test',['quality',selected]);write('test.json',test)
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==digest
    lines=['# TAOCA激波融合优化V4','',f'验证集选择结果：{selected}。网络与轨迹权重保持不变。','', '|候选|压力/密度高梯度误差下降|全域误差比|近壁误差比|通过|','|---|---:|---:|---:|---|']
    for name,d in decisions.items():lines.append(f"|{name}|{100*d['shock_pressure_density_gain']:.2f}%|{d['global_ratio']:.4f}|{d['wall_ratio']:.4f}|{d['passes']}|")
    lines+=['','没有通过门槛时保留V3，不将失败候选作为改进成果。单图误差由于覆盖范围不同，仅用于诊断。真实场往返映射误差与预测误差不能直接相减。高梯度区域是压力梯度代理区域。本研究测试集已有多次查看，属于探索性结果。']
    if test:
        lines+=['','## 锁定候选后的测试结果',json.dumps(test,ensure_ascii=False,indent=2)]
    (OUT/'优化分析.md').write_text('\n'.join(lines),encoding='utf-8');write('status.json',dict(state='complete',selected=selected,test_evaluated=test is not None))
if __name__=='__main__':
    try:main()
    except Exception:
        OUT.mkdir(parents=True,exist_ok=True);write('status.json',dict(state='failed',error=traceback.format_exc()));raise
