# FNOTransfomer — 最佳已完成版本（TAOCA V7）

此目录仅保留当前归档中完成了正式训练与测试、且五通道等权平均 NRMSE 最低的 TAOCA 增强版本 V7，以及它运行所需的共享模型组件。历史试验代码已从 GitHub 发布目录移除；版本误差汇总见仓库根目录 [`README.md`](../README.md) 的最后一节。

## 目录

- `taoca/`：V7 使用的网络、时序数据、四图采样/融合、物理损失和合法逆映射模块。模块文件名中的 V3 表示 V7 所依赖的公共图册骨干/接口实现，不代表保留了 V3 训练方案。
- `training/`：唯一训练入口 `train_taoca_v7.py`。
- `scripts/`：V7 训练数据准备、算子校验、结果对照、分析和一键执行脚本。
- `configs/`：运动学代理参数与热力学常数。

数据、模型权重、运行缓存、训练日志和图件均不在 GitHub 中。

## V7 结果概览

在该版本的共同合法支持域（99.9183%）与 540 个重叠预测实例上，五通道 NRMSE 为：压力 0.24639、密度 0.25446、温度 0.25954、X 速度 0.17220、Y 速度 0.31421。与 Cartesian 相比，温度和两个速度通道有所改善；压力、密度仍略差。近壁五通道改善，但高梯度代理区压力和密度仍更差。V7 在 V3 检查点上继续训练 60 轮，且改变了监督目标；它不是等预算的纯网格表示消融，不能据此宣称 TAOCA 全面优于 Cartesian。

## 重现要求

完整 V7 继续训练需要研究归档提供以下输入，不包含在仓库：

1. `datasets/joint_nearest/` 与 `datasets/fenli_paired/` 数据及对应 manifest；
2. V3 TAOCA 初始化权重：`outputs/atlas_v3/taoca/best.pt`；
3. Cartesian 验证基线：`outputs/physics_v2/cartesian/best.pt`、`run.json`；
4. V7 的算子文件需由脚本重新生成。

将上述文件按路径放置在项目根目录后，安装本环境匹配的 CUDA PyTorch，再执行：

```bash
pip install -r requirements.txt
bash scripts/run_taoca_v7.sh
```

本脚本按原 V7 流程进行继续训练和评估；因为检查点未公开，干净克隆不能从随机初始化复现该已报告结果。正式论文复现还应保留原归档中的数据版本和权重哈希。
