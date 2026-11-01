# 文档索引

## 研究论文

- [`研究论文草稿.md`](研究论文草稿.md)：基于本地代码和已保存评估记录撰写的完整中文论文草稿；结果范围、探索性结论和未验证事项均有注明。此稿不是投稿定稿。

## 实验协议与版本记录

- [`protocols/PHYSICS_V2.md`](protocols/PHYSICS_V2.md)：共同空间时序骨干、损失、几何修复与比较协议。
- [`protocols/ATLAS_V3.md`](protocols/ATLAS_V3.md)：TAOCA 四图径向预算重新分配。
- [`protocols/DIRECT_HISTORY_V6.md`](protocols/DIRECT_HISTORY_V6.md)：直接从原始 CFD 网格采样历史输入的消融。
- [`protocols/PHYSICAL_V7.md`](protocols/PHYSICAL_V7.md)：最终逆映射一致性微调；不是纯网格表示对照。
- [`protocols/METRIC_V8.md`](protocols/METRIC_V8.md)：局部几何度量条件候选及验证边界。
- [`protocols/JOINT_PROTOCOL.md`](protocols/JOINT_PROTOCOL.md)：历史场与运动轨迹联合预测的协议。
- [`protocols/STAGED_PROTOCOL.md`](protocols/STAGED_PROTOCOL.md)：冻结历史轨迹代理的分阶段预测协议。
- [`protocols/OPTIMIZATION_CAMPAIGN.md`](protocols/OPTIMIZATION_CAMPAIGN.md)：连续优化目标、实验门槛及失败结果处理。

## 部署

- [`deployment/LINUX_5090.md`](deployment/LINUX_5090.md)：旧数据打包版本的双 RTX 5090 指南，保留审计；不适用于当前源码-only 仓库。
- [`deployment/LINUX_JOINT.md`](deployment/LINUX_JOINT.md)：旧数据打包版本的联合预测指南，保留审计；不适用于当前源码-only 仓库。

## 历史交付记录

- [`reports/数据与训练接口交付报告.md`](reports/数据与训练接口交付报告.md)：500 帧配对数据与早期接口交付记录。
- [`reports/训练状态与最新入口_历史快照_2026-10-20.md`](reports/训练状态与最新入口_历史快照_2026-10-20.md)：保存当时的状态快照；当前训练状态必须以 `outputs/` 中机器生成的状态文件为准。

> V2–V8 协议文件用于追溯不同实验版本，不表示这些版本都已完成正式评估。`outputs/` 与数据文件保存在本地且不随仓库发布；引用这些本地结果的旧实验图不会在 GitHub 上显示。

