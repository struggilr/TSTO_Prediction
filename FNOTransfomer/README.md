<!-- detailed-design-navigation -->
> **详细代码设计（2026-10-21）**：[TAOCA 多图册表示与多轮预测实验](../../docs/modules/09-taoca.md) · [仓库总览](../../README.md)。新增文档按当前源码说明结构、数据、接口与限制；下方保留既有模块说明。

# FNOTransfomer_TAOCA

二维双体 TSTO：比较固定 Cartesian 表示与 TAOCA 四图册表示。正式数据来自 `E:/fenli` 的五条 Mach 时序，共 500 帧；不使用早期十帧开发数据。

## 实验定义

- A：原始非结构网格 → 固定 Cartesian 栅格 → FNO/Transformer → 原始节点。
- B：原始非结构网格 → 两个 O 图、outer、gap → 同一个 FNO/Transformer → 原始节点。
- 两组均采用原始单元三角化后的 P1 插值，不将 Cartesian 场再次扭曲为 TAOCA。A 是无 TAOCA 的规则栅格基线，并非直接在非结构节点上运行 FNO。
- 相同空间采样点预算：A 为 `192×256`，B 为 `4×48×256`，各 49,152 点。重叠图有重复覆盖，因此相同点数不代表相同物理分辨率。
- 模型保留 V2.0 的 `FNOEncoder`、`TemporalPredictor`，重新实现可变尺寸共享解码器、跨图平均潜变量交换与图角色编码。两组参数数目相同。这是适配后的受控比较，不是原始 V2.0 全功能复现。
- **已知目标时刻几何条件下的流场预测**：两组均读取未来几何坐标和有效掩膜，不读取未来物理场作为模型输入。本实验不验证物体运动预测。
- outer 锚点和 O 图边界分段仅由每条轨迹第一帧确定；后续读取当前目标几何。固定包络不能包住后期物体时，相关 donor 被固体碰撞检测剔除，不能称为全时域严格外环；详见逐帧审计。

## 数据与划分

五条序列 Mach 5.0、5.5、6.0、6.5、7.0，帧数依次 101、101、101、96、101。导出的 DAT 中 `SOLUTIONTIME` 重复为零，采用文件名的六位小数时间排序，保留最后一帧。

每条序列索引 `[0,60)` 为训练、`[60,80)` 为验证、`[80,N)` 为测试目标。模型读 20 帧，预测 10 帧。每个目标窗口完整位于自身划分内；验证/测试可读取过去的真实历史，因此这是滑动窗口预测，不能解释为跨整段测试区的自主滚动预测。训练窗口 155 个，验证候选 55 个，测试候选 55 个；验证用不重叠块，测试逐物理帧去重。

仅用训练帧原始节点统计共同的五变量均值和标准差，顺序为压力、密度、温度、X 速度、Y 速度。统计与误差均为节点权重，尚非面积权重。

`datasets/fenli_paired/Ma*/0000.h5` 每帧同时包含两种表示、原始场和逆映射：

```
source/{x,y,triangles,fields}
cartesian/cartesian/{fields,mask,xy,J,inv_ids,inv_bary,inv_weight}
taoca/{smj,xmj,outer,gap}/{fields,mask,xy,J,inv_ids,inv_bary,inv_weight}
```

无效插值点置零且携带显式 mask。逆映射不填补漏点；TAOCA 以几何权重除以 donor 最大边长平方后融合。源节点接口坐标精确合并；若多区域重复节点的场值不同，保留第一区域值。P1 三角化与融合不保证积分守恒。

## 本地重新预处理与审计

```
python scripts/prepare_all.py --source E:/fenli --workers 3
python scripts/audit_paired.py
```

支持逐帧断点续跑，只有完整 500 帧才生成正式 manifest。旧 `datasets/fenli_v2_parallel` 使用错误的按区号壁面识别，不可用于正式训练；`datasets/source`、`datasets/taoca` 仅为十帧旧原型。旧 `scripts/prepare_datasets.py`、`benchmark.py` 等不属于正式训练依赖。

## 上传与服务器运行（Linux，两张 NVIDIA GPU）

双 RTX 5090 请优先阅读 [LINUX_5090.md](LINUX_5090.md)，只需上传上一级目录的 `FNOTransfomer_TAOCA_server.zip`。双卡脚本现默认 batch=4、workers=2、FP32，先执行CUDA和真实数据检查。下方直接运行Python的默认batch仍为1。

上传本目录的 `taoca/`、`scripts/`、`train_compare.py`、`requirements.txt`、`run_dual_gpu.sh` 和 `datasets/fenli_paired/`。不需要 E 盘、原始 DAT、旧工程或 Fluent。建议放在服务器本地 SSD。

先按服务器 CUDA 驱动安装相应 PyTorch，再安装依赖并分别运行一步检查：

```bash
python -m pip install -r requirements.txt
python scripts/verify_package.py
CUDA_VISIBLE_DEVICES=0 python train_compare.py --route cartesian --smoke --workers 0 --output outputs/smoke_cartesian
CUDA_VISIBLE_DEVICES=1 python train_compare.py --route taoca --smoke --workers 0 --output outputs/smoke_taoca
bash run_dual_gpu.sh
```

该脚本同时在 GPU 0 跑 A、GPU 1 跑 B，最多 100 epoch，验证连续 20 epoch 不改善提前停止，自动测试并生成 `outputs/comparison.json`。单独执行：

```bash
CUDA_VISIBLE_DEVICES=0 python -u train_compare.py --route cartesian --device cuda --epochs 100
CUDA_VISIBLE_DEVICES=1 python -u train_compare.py --route taoca --device cuda --epochs 100
python scripts/compare_runs.py
```

输入设备在各进程中均为 `cuda:0`；`CUDA_VISIBLE_DEVICES` 已完成物理卡选择。默认 batch=1，FP32。显存不足时两组同时降低 `--width`、`--latent`，不能只改一组；必要时先做 smoke。模型和 manifest 哈希、参数量、GPU、训练时间、峰值显存都会记录。计时包含训练数据读取及优化，不含验证；预处理时间另外报告。不同 GPU/共享磁盘会影响同时计时，论文应交换 GPU 并重复随机种子。

## 如何看结果

`evaluation.json` 在两组共同原始节点支持上报告总预测 RMSE、真实场往返误差、预测与真实映射回场的差异；这些 RMSE 不是可直接相加的分解。另列每组自身覆盖率，防止只在容易区域打分。NRMSE 分母是共同训练标准差，不是变量均值。

`audit.json` 提供所有帧的几何无效数、节点覆盖和往返误差。更低训练 MSE 不等于更低物理误差，因为两种表示的采样分布不同。允许实验得出 TAOCA 没有优势的结论。

当前任务只准备数据和训练接口，并做最小运行检查；正式模型收敛、预测精度与耗时优势必须以服务器训练输出为证。500 帧属于五条相关轨迹的时间划分，不足以证明跨 Mach 或跨几何泛化，也未证明严格守恒、求解器 GCL 或长期自主滚动稳定性。
