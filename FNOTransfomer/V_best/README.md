# V_best：TAOCA V9

此目录仅保留官方统一比较中选出的 TAOCA V9 代码，不包含模型权重、HDF5 数据、训练缓存和大体量结果。官方选择根据九路线同一共同支持域、五通道全域 train-std NRMSE 算术平均，V9=0.245286，为最低分。该选择是单种子事后选择，不能据此声称五通道都优于 Cartesian：同表中压力、密度退化，温度、速度X、速度Y改善。

## 代码组织

- `train_local_v9.py`：V8 初始化、增加局部残差分支并训练 V9。
- `train_physics_v2.py`：基础物理监督训练定义和原 Cartesian 训练函数。
- `run_local_v9_h.py`、`resume_local_v9_h.py`：H 盘正式流程入口/恢复入口。
- `taoca/`：网络、数据读取、图册、融合/逆映射、物理监督与评价依赖。
- `scripts/`：V9 审计、测试与分析工具。
- `V9_PROTOCOL.md`：V9 方法与证据边界。

## 复现所需外部文件

需从原项目提供数据集 `datasets/joint_nearest`、`datasets/fenli_paired`，共享轨迹模型 `outputs/shared_trajectory/model.json`，V8 TAOCA 和 Cartesian 检查点、V7 物理算子缓存与元数据。大文件故未复制到本目录或 GitHub。

官方结果表和各版本说明见上级 `other_version/README.md`。本地完整归档路径为 `E:\1-AllCode\cfd_to_npy_datasets\FNOTransfomer_TAOCA_整理版`。
