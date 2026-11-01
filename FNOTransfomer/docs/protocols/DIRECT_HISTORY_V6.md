# V6：历史场直接采样的受控对照

## 本轮唯一主动改动

V3历史场路径：原始CFD → 原有nearest Cartesian数据 → TAOCA历史图。

V6历史场路径：原始CFD节点变量及原始三角形 → P1直接采样 → 相同TAOCA历史图。

两者使用相同原始快照、训练集归一化、35/35/140/70图分辨率、440列、123200采样点、340349参数网络、共享冻结轨迹、20步历史预测10步未来、时序划分、随机种子42、优化器、学习率调度和100轮预算。

仅使用窗口[s-history,s)的直接采样场作为输入。保持原有输入掩码与源网格有效性掩码的交集，不把无源供体的位置外推补齐。目标物理场、目标有效区、最后观测物理场先验、真实轨迹标签均与V3相同。预测未来几何仍来自历史轨迹预测器，不读取真实未来几何。

本轮不是完整替换所有重采样环节：监督目标仍是原有Cartesian数据，物理场先验也未改变。如此才能先隔离历史输入采样的影响。此前直接采样真实场的往返收益不能直接当作本轮预测收益。

## 自动流程与输出

运行入口：run_direct_history_v6_local.py。

1. scripts/prepare_direct_history_v6.py：读取500帧已核对时间的原始CFD源网格缓存，生成datasets/taoca_direct_history_v6；核查有限值、供体重心权重、维度与无效位置零值，记录数据与算法校验值。
2. scripts/test_direct_history_v6.py：核查目标、几何与先验逐项不变；污染缓存中的未来场后，历史输入必须完全不变。
3. train_direct_history_v6.py --smoke：真实数据3次优化、验证及10步测试回映射。
4. train_direct_history_v6.py --resume：从同一随机初始化正式训练100轮，按验证损失保存best.pt。
5. scripts/compare_direct_history_v6.py：冻结Cartesian、旧TAOCA V3、新TAOCA V6在共同合法支持域上比较全部540个重叠预测实例，核查参考场和预测轨迹一致。
6. scripts/analyze_direct_history_v6.py：自动输出曲线、分区/分工况误差、2700张逐通道云图、典型误差图及analysis/分析报告.md。

任何准备、测试或训练步骤失败都停止流水线，错误见outputs/direct_history_v6/pipeline_status.json及对应日志，不跳过失败门槛继续训练。

## 不变的基线与解释边界

Cartesian检查点不变，V3代码与模型不覆盖。新增物理项与网络模块留到后续独立消融，本轮不添加。

同一物理训练目标保证可比，但监督目标本身的nearest表示误差仍可能限制最终精度。若后续更换共同目标为原始CFD的其他重构，则Cartesian也应重新训练。单次种子和已反复查看的测试集只支持探索性结论，不能预先宣称TAOCA必然更优或具有普遍加速比。
