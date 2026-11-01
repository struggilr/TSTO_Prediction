"""Prepare a controlled 816-parameter geometry-feature extension of V7."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    text=(ROOT/'train_physical_v7.py').read_text(encoding='utf-8')
    text=text.replace('from taoca.allocated_model_v3 import AllocatedPredictor','from taoca.metric_model_v8 import MetricPredictor').replace('model=AllocatedPredictor(', 'model=MetricPredictor(')
    text=text.replace("BASE=ROOT/'outputs/physical_v7'", "BASE=ROOT/'outputs/metric_v8'")
    text=text.replace('folder.mkdir(exist_ok=True)','folder.mkdir(parents=True,exist_ok=True)')
    for name in ['operators','operator_tests.json','prepare_status.json']:
        text=text.replace(f"BASE/'{name}'",f"ROOT/'outputs/physical_v7/{name}'")
    text=text.replace("model.load_state_dict(torch.load(ROOT/'outputs/atlas_v3/taoca/best.pt',map_location='cuda',weights_only=True))", "missing,extra=model.load_state_dict(torch.load(ROOT/'outputs/atlas_v3/taoca/best.pt',map_location='cuda',weights_only=True),strict=False);assert not extra and all(k.startswith('metric_encoder.') for k in missing)")
    text=text.replace("'train_physical_v7.py','taoca/physical_supervision_v7.py'", "'train_metric_v8.py','taoca/metric_features_v8.py','taoca/metric_model_v8.py','taoca/physical_supervision_v7.py'")
    text=text.replace('Exploratory TAOCA-specific inverse-consistent fine tuning;', 'Exploratory metric-conditioned TAOCA fine tuning; V7 objective and schedule unchanged; 816 extra zero-initialized feature-conditioning parameters;')
    (ROOT/'train_metric_v8.py').write_text(text,encoding='utf-8')
    text=(ROOT/'scripts/compare_physical_v7.py').read_text(encoding='utf-8').replace("root=ROOT/'outputs/physical_v7'","root=ROOT/'outputs/metric_v8'").replace('FNO_TAOCA_V7','FNO_TAOCA_V8')
    text=text.replace("assert run['parameters']==a['parameters']", "assert run['parameters']==a['parameters']+816")
    text=text.replace('TAOCA V7 has different loss and additional fine-tuning budget','TAOCA V8 has an additional 816-parameter metric encoder, different loss and additional fine-tuning budget')
    text=text.replace("totals={};rows=[];reference_count=0", "totals={};rows=[];reference_count=0;v7_sse=np.zeros(5);v7_count=0")
    text=text.replace("common=valid&b['support'][:]&fa['support'][:]&fb['support'][:]", "common=valid&b['support'][:]&fa['support'][:]&fb['support'][:]\n            with h5py.File(ROOT/'outputs/physical_v7/taoca/predictions'/path.name) as fv7:\n                assert np.array_equal(ref,fv7['reference'][:]);previous=fv7['predicted'][:];common&=fv7['support'][:]&np.isfinite(previous).all(1)")
    text=text.replace("reference_count+=int(valid.sum())", "reference_count+=int(valid.sum());v7_sse+=np.square(previous.transpose(1,0,2,3)[:,common].astype('f8')-ref.transpose(1,0,2,3)[:,common]).sum(1);v7_count+=int(common.sum())")
    text=text.replace("result['all_five_global_channels_better']", "result['v7_same_support_all_nrmse']=np.sqrt(v7_sse/max(1,v7_count)).tolist();result['all_five_global_channels_better']",1)
    (ROOT/'scripts/compare_metric_v8.py').write_text(text,encoding='utf-8')
    text=(ROOT/'scripts/analyze_physical_v7.py').read_text(encoding='utf-8').replace('outputs/physical_v7','outputs/metric_v8').replace('FNO_TAOCA_V7','FNO_TAOCA_V8').replace('V7物理网格回映射监督','V8局部几何度量与物理回映射监督').replace('V7改变损失并增加微调预算','V8增加816个几何编码参数、改变损失并增加微调预算')
    (ROOT/'scripts/analyze_metric_v8.py').write_text(text,encoding='utf-8')
if __name__=='__main__':main()
