"""Recompute completed Cartesian/V2/V3/V6 on exactly the same test support."""
import os
os.environ['MKL_THREADING_LAYER']='GNU'
import sys,json,hashlib,contextlib,datetime
from pathlib import Path
import numpy as np
import h5py
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from taoca.physics_losses import regional_maps
def main():
    torch.set_num_threads(4);out=ROOT/'outputs/version_recheck_20261020';out.mkdir(exist_ok=True)
    folders={'FNO_Cartesian':ROOT/'outputs/physics_v2/cartesian','FNO_TAOCA_V2':ROOT/'outputs/physics_v2/taoca','FNO_TAOCA_V3':ROOT/'outputs/atlas_v3/taoca','FNO_TAOCA_V6':ROOT/'outputs/direct_history_v6/taoca'}
    runs={k:json.loads((p/'run.json').read_text()) for k,p in folders.items()};base=runs['FNO_Cartesian'];names=None
    for name,folder in folders.items():
        assert len(json.loads((folder/'history.json').read_text()))==100,name
        assert not runs[name]['arguments']['smoke'],name
        for field in ['epochs','seed','batch_size','lr']:assert runs[name]['arguments'][field]==base['arguments'][field],(name,field)
        assert runs[name]['parameters']==base['parameters']
        files={p.name for p in (folder/'predictions').glob('*.h5')};assert len(files)==54
        if names is None:names=files
        else:assert names==files
    baseline_sha=hashlib.sha256((folders['FNO_Cartesian']/'best.pt').read_bytes()).hexdigest()
    assert baseline_sha==runs['FNO_TAOCA_V3']['cartesian_checkpoint_sha256']==runs['FNO_TAOCA_V6']['cartesian_checkpoint_sha256']
    totals={};frames=[];truth_count=0
    for i,filename in enumerate(sorted(names)):
        with contextlib.ExitStack() as stack:
            files={k:stack.enter_context(h5py.File(folder/'predictions'/filename)) for k,folder in folders.items()};first=files['FNO_Cartesian'];ref=first['reference'][:];valid=first['valid'][:].astype(bool);pose=first['pose'][:];pred={};common=valid.copy()
            for name,f in files.items():
                assert np.array_equal(ref,f['reference'][:]),(name,filename,'reference')
                assert np.array_equal(valid,f['valid'][:].astype(bool)),(name,filename,'valid')
                assert np.array_equal(pose,f['pose'][:]),(name,filename,'pose')
                pred[name]=f['predicted'][:];common&=f['support'][:].astype(bool)&np.isfinite(pred[name]).all(1)
            wall,shock=regional_maps(torch.from_numpy(ref)[None],torch.from_numpy(valid.astype('f4'))[None]);wall=wall[0,:,0].numpy()>.5;shock=shock[0,:,0].numpy()>.5;truth_count+=int(valid.sum())
            for t in range(len(ref)):
                row=dict(window=Path(filename).stem,lead=t+1,n=int(common[t].sum()),nrmse={})
                for region,mask in [('all',common[t]),('wall',common[t]&wall[t]),('shock',common[t]&shock[t])]:
                    g=totals.setdefault(region,dict(n=0,sse={k:np.zeros(5) for k in folders}));n=int(mask.sum());g['n']+=n
                    for name,values in pred.items():
                        error=np.square(values[t][:,mask].astype('f8')-ref[t][:,mask]).sum(1);g['sse'][name]+=error
                        if region=='all':row['nrmse'][name]=np.sqrt(error/max(1,n)).tolist()
                frames.append(row)
        print(json.dumps(dict(windows=i+1,total=len(names))),flush=True)
    metrics={r:dict(n=g['n'],nrmse={name:np.sqrt(error/max(1,g['n'])).tolist() for name,error in g['sse'].items()}) for r,g in totals.items()}
    result=dict(checked_at=datetime.datetime.now().isoformat(),forecast_instances=len(frames),common_coverage=totals['all']['n']/truth_count,cartesian_checkpoint_sha256=baseline_sha,all_references_validity_and_predicted_poses_exact=True,regions=metrics,protocol='540 overlapping forecast instances; four completed versions on exactly the same intersection of legal support; shared train-standard-deviation normalization; float64 SSE accumulation; single seed; repeatedly viewed test remains exploratory',version_changes={'V2':'Uniform chart allocation','V3':'35/35/140/70 chart allocation','V6':'V3 geometry/network/loss with original-CFD direct sampling for historical chart inputs only'})
    (out/'comparison.json').write_text(json.dumps(result,indent=2));(out/'frame_metrics.json').write_text(json.dumps(frames))
    lines=['# Cartesian与TAOCA已完成版本统一复核','',f'复核时间：{result["checked_at"]}。四个版本均完成100轮、同一随机种子及优化器配置、340349参数。统一540个重叠预测实例及四路线共同合法区域。有效覆盖率{result["common_coverage"]*100:.5f}%。','', '各字段的NRMSE使用原训练集标准差归一化；不是物理量百分比误差。高压力梯度区域为既定压力梯度代理区域。','']
    fields=['压力','密度','温度','X速度','Y速度']
    for region,title in [('all','全域'),('wall','近壁'),('shock','高压力梯度')]:
        m=metrics[region]['nrmse'];lines +=[f'## {title}','', '|通道|Cartesian|V2|V3|V6|V6相对V3误差变化|','|---|---:|---:|---:|---:|---:|']
        for j,field in enumerate(fields):lines.append('|'+field+'|'+ '|'.join(f'{m[name][j]:.6f}' for name in folders)+f'|{100*(m["FNO_TAOCA_V6"][j]/m["FNO_TAOCA_V3"][j]-1):+.2f}%|')
        lines+=['','最后一列正值表示V6误差更大；负值表示改善。','']
    m=metrics['all']['nrmse'];means={k:float(np.mean(v)) for k,v in m.items()};lines+=['## 综合判断','', '五通道等权平均NRMSE（便于概览，不替代逐通道结果）：']
    for name,value in means.items():lines.append(f'- {name}：{value:.6f}')
    lines+=['',f'V6相对V3的五通道等权平均NRMSE变化为{100*(means["FNO_TAOCA_V6"]/means["FNO_TAOCA_V3"]-1):+.2f}%。','V4是未通过门槛的融合诊断，V5是映射/剖面诊断，不应当作为独立训练版本。未完成完整测试的V7与只通过冒烟测试的V8不参与最终精度排名。','', '当前缓存的共同参考场、有效区和预测轨迹逐项一致，Cartesian检查点哈希与V3/V6记录一致。模型训练与推理耗时可能受机器负载影响，本复核不据此宣称速度优势。540个实例有时间重叠，不是540个独立样本。','重复使用同一测试集的结果仍是探索性证据。此前不同版本评估表的支持范围不完全相同，应以本次统一支持域的表格为准。']
    (out/'版本统一复核.md').write_text('\n'.join(lines),encoding='utf-8');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
