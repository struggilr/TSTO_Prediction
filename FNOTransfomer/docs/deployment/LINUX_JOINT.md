# 历史部署说明：流场与轨迹联合预测（旧打包版本）

> **仅供追溯，不是当前 GitHub 源码版的启动指南。** 本文依赖未发布的 `joint_nearest` / 配对数据缓存与旧 zip 打包结构。公开仓库只含源码，不含 HDF5/NPZ 数据或权重。

只上传FNOTransfomer_TAOCA_joint_server.zip。旧FNOTransfomer_TAOCA_server.zip不是本实验。

```bash
unzip FNOTransfomer_TAOCA_joint_server.zip
cd FNOTransfomer_TAOCA
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python scripts/verify_package.py
bash run_joint_dual_gpu.sh
python scripts/compare_joint.py
```

GPU0 A原始最近邻网格，GPU1 B TAOCA；默认batch2，workers0，100轮，20→10步；无未来真实位置、mask、网格输入。代码和数据逻辑见JOINT_PROTOCOL.md。模型依据观测的历史位姿与流场联合预测未来。第一帧物体形状已知。

真实Windows smoke已完成两路训练、验证、预测及回映射，但不代表收敛精度。查看outputs/joint_cartesian与joint_taoca。B的预测几何回映射CPU耗时较长，训练后测试阶段可能等待数分钟或更久。训练目前固定轮数且保存最佳验证模型。全域误差对缺测用训练均值填充，必须和覆盖率一起报告；不能只选B覆盖到的区域声称精度优势。当前是时间外推窗口实验，尚未验证长时间自主滚动或跨Mach泛化。

两路网格总点数一致123200；A一张280×440，B四张70×440。模型结构相同，但形状/图数导致计算过程不同。未修改原V2.0工程，也没有将其含真实未来mask的旧训练入口用于本实验。
