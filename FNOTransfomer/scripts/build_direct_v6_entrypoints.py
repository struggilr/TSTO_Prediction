"""Generate isolated V6 entrypoints from frozen, already-tested V3 orchestration."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    text=(ROOT/'train_atlas_v3.py').read_text(encoding='utf-8')
    text=text.replace('from taoca.allocated_data_v3 import AllocatedWindows','from taoca.direct_history_v6 import DirectHistoryWindows').replace('AllocatedWindows(','DirectHistoryWindows(')
    text=text.replace("root=ROOT/'outputs/atlas_v3';selection=json.loads((root/'allocation_selection.json').read_text())","root=ROOT/'outputs/direct_history_v6';selection=json.loads((ROOT/'outputs/atlas_v3/allocation_selection.json').read_text())")
    text=text.replace("'train_atlas_v3.py','outputs/atlas_v3/allocation_selection.json'","'train_direct_history_v6.py','taoca/direct_history_v6.py','datasets/taoca_direct_history_v6/manifest.json','outputs/atlas_v3/allocation_selection.json'")
    text=text.replace('TAOCA allocation changed only;', 'V6 changes historical chart-field sampling only to original CFD P1; target labels, physical prior, architecture, geometry and losses unchanged;')
    text=text.replace('"""TAOCA-only allocation experiment, gated before training; frozen V2 Cartesian baseline."""','"""Direct original-CFD history ablation; frozen V3 architecture and Cartesian baseline."""')
    (ROOT/'train_direct_history_v6.py').write_text(text,encoding='utf-8')
    text=(ROOT/'scripts/compare_atlas_v3.py').read_text(encoding='utf-8')
    text=text.replace("root=ROOT/'outputs/atlas_v3'","root=ROOT/'outputs/direct_history_v6'")
    text=text.replace("v2/'taoca/predictions'/path.name","ROOT/'outputs/atlas_v3/taoca/predictions'/path.name")
    text=text.replace('FNO_TAOCA_V3','TEMP_NEW_TAOCA').replace('FNO_TAOCA_V2','FNO_TAOCA_V3').replace('TEMP_NEW_TAOCA','FNO_TAOCA_V6')
    (ROOT/'scripts/compare_direct_history_v6.py').write_text(text,encoding='utf-8')
    text=(ROOT/'scripts/auto_analyze_atlas_v3.py').read_text(encoding='utf-8')
    text=text.replace("BASE=ROOT/'outputs/atlas_v3'","BASE=ROOT/'outputs/direct_history_v6'")
    text=text.replace("ROOT/'outputs/physics_v2/taoca'","ROOT/'outputs/atlas_v3/taoca'")
    text=text.replace('FNO_TAOCA_V3','TEMP_NEW_TAOCA').replace('FNO_TAOCA_V2','FNO_TAOCA_V3').replace('TEMP_NEW_TAOCA','FNO_TAOCA_V6')
    text=text.replace('TAOCA V3训练结果自动分析','TAOCA V6原始CFD历史直接采样训练结果自动分析')
    (ROOT/'scripts/analyze_direct_history_v6.py').write_text(text,encoding='utf-8')
if __name__=='__main__':main()
