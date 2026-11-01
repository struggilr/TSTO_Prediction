# TAOCA Python 包索引

`taoca/` 是本项目的几何、数据、模型和评价实现。文件名带版本号的组件用于保留实验可追溯性；是否适用于某个训练入口，以该入口导入关系及相应实验协议为准。

| 模块 | 职责 |
|---|---|
| `tecplot.py`, `prepare.py`, `data.py`, `joint.py`, `physics_data.py` | FE 数据读取、时序窗口、数据准备和样本接口 |
| `geometry.py`, `domain_charts.py`, `allocation_v3.py` | TAOCA 图册几何修复和固定预算分配 |
| `inverse_complete.py`, `inverse_validated.py`, `chart_transfer_v3.py` | donor 搜索、合法逆映射和跨图回映射 |
| `model.py`, `fno_blocks.py`, `temporal_net.py` | 早期 FNO/Transformer 基础实现 |
| `spatial_physics.py`, `allocated_model_v3.py`, `staged.py` | 空间时序骨干、不同图高模型和冻结运动学代理 |
| `physics_losses.py`, `physical_supervision_v7.py` | 物理坐标损失、区域加权与评价算子监督 |
| `metric_features_v8.py`, `metric_model_v8.py`, `context_ablation_v8.py` | V8 几何度量和跨图上下文诊断实验 |
| `evaluate.py`, `physics_evaluate.py`, `allocated_evaluate_v3.py`, `joint_evaluate.py` | 版本化预测和回映射评价 |

严格守恒重映射、Fluent 原生 overset connectivity 和求解器级 GCL 证明不属于当前包已实现的能力。
