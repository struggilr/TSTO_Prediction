"""Versioned reports; explicitly distinguish fine tuning from equal-budget ablation."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    text=(ROOT/'scripts/compare_direct_history_v6.py').read_text(encoding='utf-8').replace("outputs/direct_history_v6","outputs/physical_v7").replace('FNO_TAOCA_V6','FNO_TAOCA_V7')
    text=text.replace("for key in ['epochs','seed','batch_size','lr']:","for key in ['seed','batch_size']:")
    text=text.replace('Single seed, all overlapping instances on three-route common legal support; repeated use of this test set is exploratory, not independent generalization evidence.','Single seed; TAOCA V7 has different loss and additional fine-tuning budget from a V3 checkpoint. This is an enhanced-system comparison, not a pure representation/equal-budget ablation. All overlapping instances on common legal support; repeatedly viewed test set remains exploratory.')
    text=text.replace("(root/'comparison.json').write_text", "result['all_five_global_channels_better']=bool(all(x<y for x,y in zip(result['regions']['all']['nrmse']['FNO_TAOCA_V7'],result['regions']['all']['nrmse']['FNO_Cartesian'])));result['pressure_density_shock_better']=bool(all(x<y for x,y in zip(result['regions']['shock']['nrmse']['FNO_TAOCA_V7'][:2],result['regions']['shock']['nrmse']['FNO_Cartesian'][:2])));(root/'comparison.json').write_text")
    (ROOT/'scripts/compare_physical_v7.py').write_text(text,encoding='utf-8')
    text=(ROOT/'scripts/analyze_direct_history_v6.py').read_text(encoding='utf-8').replace('outputs/direct_history_v6','outputs/physical_v7').replace('FNO_TAOCA_V6','FNO_TAOCA_V7').replace('V6原始CFD历史直接采样','V7物理网格回映射监督')
    text=text.replace("row[split]['loss']","row[split]['loss']")
    text=text.replace("axs[0].legend(fontsize=7);save(fig,'training_curves')","axs[0].legend(fontsize=7);fig.suptitle('Method-specific objectives; values are not directly comparable across versions',fontsize=9);save(fig,'training_curves')")
    text=text.replace("训练曲线用于检查优化及验证集选择；","V7改变损失并增加微调预算，不能将收益全部归于网格变换；各版本训练目标数值不可直接比较。训练曲线用于检查各自优化及验证集选择；")
    (ROOT/'scripts/analyze_physical_v7.py').write_text(text,encoding='utf-8')
if __name__=='__main__':main()
