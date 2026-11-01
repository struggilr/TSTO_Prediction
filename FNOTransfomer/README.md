# FNOTransfomer · 二维 TSTO 双体流场预测

本目录是可运行源码项目。完整的论文式研究说明见 [`docs/研究论文草稿.md`](docs/研究论文草稿.md)；协议、部署手册和历史交付记录已分类整理在 [`docs/README.md`](docs/README.md)。当前代码保留多轮实验以便复查，正式结果与未完成版本有明确区分。

## 目录结构

```text
taoca/       数据读取、网格图册、预测模型、损失、回映射和评价
scripts/     数据准备、审计、诊断、版本比较、打包等工具
docs/
  protocols/ 实验协议及 V2–V8 版本设计记录
  deployment/ Linux/双 RTX 5090 部署说明
  reports/   数据交付及历史状态记录（历史状态不代表当前运行状态）
datasets/    本地数据与缓存；不纳入 Git
outputs/     本地训练权重、指标、日志和图件；不纳入 Git
```

训练入口留在项目根目录，是为了保持现有相对路径和运行协议：`train_compare.py` 是早期“已知未来几何”比较；`train_joint.py` 是历史流场与轨迹联合预测实验；`train_physics_v2.py`、`train_atlas_v3.py`、`train_direct_history_v6.py`、`train_physical_v7.py` 和 `train_metric_v8.py` 对应后续版本。**文件名较新不代表实验已完成或表现更优**；运行状态以每个实验 `outputs/<version>/status.json`、`pipeline_status.json` 和最终 `comparison.json` 为准。

## 正式数据与主要结果

正式 TSTO 数据为五条二维时序：Ma 5.0、5.5、6.0、6.5、7.0，共 500 帧。训练、验证、测试按时间先后切分，避免随机打乱未来帧。物理变量顺序为压力、密度、温度、X 速度、Y 速度。局部数据和模型权重不随代码发布。

已经完成且可复核的主对照是 Cartesian 与 TAOCA V3：共享模型参数（340,349）、总采样点预算（123,200/帧）、历史轨迹预测器和相同的主要训练配置。V3 将四图径向点数分配为 `35/35/140/70`，Cartesian 为 `280×440`。结果没有支持“TAOCA 五通道全面优于 Cartesian”：近壁五通道改善明显，但全域压力/密度和高压力梯度代理区域压力/密度仍退化。具体数据和解释见研究论文草稿的结果表。

## 本地准备和运行

Python 依赖见 [`requirements.txt`](requirements.txt)。基础原始数据准备（需提供本机的 Tecplot DAT 文件）：

```bash
python scripts/prepare_all.py --source E:/fenli --workers 3
python scripts/audit_paired.py
```

各训练版本依赖不同的数据缓存和几何算子，不能只运行一个 `train_*.py` 就假设所有版本都能启动。请先阅读相应协议，例如 [`docs/protocols/PHYSICS_V2.md`](docs/protocols/PHYSICS_V2.md)、[`docs/protocols/ATLAS_V3.md`](docs/protocols/ATLAS_V3.md) 或 [`docs/protocols/PHYSICAL_V7.md`](docs/protocols/PHYSICAL_V7.md)，再执行对应 `run_*_local.py` 流水线。旧双 RTX 5090 文档只对应曾经打包数据的版本，不是当前源码仓库的直接启动步骤。源码和文档索引见 [`docs/README.md`](docs/README.md)、[`scripts/README.md`](scripts/README.md) 与 [`taoca/README.md`](taoca/README.md)。

公开仓库只放源码、配置、协议、说明和论文草稿；`datasets/`、`outputs/`、HDF5/NPZ/NPY 数据、模型权重、日志和生成图片均由 `.gitignore` 排除。复现实验需要用户另行提供原始数据，并按对应协议重建数据集。

## 已知边界

- 已有精度结论主要来自五条相关 Mach 时间序列、单随机种子和被多次查看的时间外推测试段；属于探索性证据，不是独立跨几何泛化证明。
- 测试预测实例存在时间重叠，540 个窗口步并非 540 个独立样本。
- TAOCA 图册使用 P1 三角插值及 donor 融合，不是严格守恒重映射；也没有完成 Fluent 求解器级 GCL 验证。
- V7/V8 等版本改变训练目标或额外训练预算，不能当作纯网格表示消融。
- 当前共同合法支持上的 NRMSE 不等价于物理面积积分误差；必须连同覆盖率解读。
- 该项目不能据现有结果声称 TAOCA 在全部通道、激波区域或运行效率上普遍优于 Cartesian。
