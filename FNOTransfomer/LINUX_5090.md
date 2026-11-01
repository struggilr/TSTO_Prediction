# 双 RTX 5090 运行说明

上传且仅需上传上一级目录的 `FNOTransfomer_TAOCA_server.zip`。包含500帧配对HDF5、归一化统计、划分、审计、训练/评价代码。无需原始 E:/fenli、旧V2.0项目或Windows虚拟环境。解压后建议预留50GB磁盘空间，使用本地SSD。

```bash
unzip FNOTransfomer_TAOCA_server.zip
cd FNOTransfomer_TAOCA
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python scripts/verify_package.py
bash run_dual_gpu.sh
```

使用支持RTX 5090/Blackwell的CUDA PyTorch构建（上述cu128为安装方案）；主机驱动也必须兼容。无需复制Windows CUDA环境。脚本实际执行两卡矩阵乘法与FFT，随后分别完成真实数据训练/验证/回映射小测试，再并行正式训练。无法访问服务器，因此本地Windows通过不能替代该检查。失败时查看outputs/server_smoke_*.log；不要将检查结果作为正式预测精度。

默认每卡一项实验：GPU0 Cartesian，GPU1 TAOCA；batch size=4，DataLoader workers=2/进程，FP32，最多100 epochs，验证连续20轮未改进提前停止。模型两路完全相同：width32、modes12、latent128、FNO3层、Transformer2层，2,959,365参数；历史20帧预测10帧。49152采样点/路径，未改变数据或拆分；155训练窗口。batch4是保守起点，不承诺最大吞吐，不因5090显存较大就增加模型或改变空间分辨率。空间分辨率变更必须重建数据。

可选更大批量（两路必须一致，先经脚本小测试）：

```bash
BATCH_SIZE=8 WORKERS=2 EPOCHS=100 bash run_dual_gpu.sh
```

每个worker读取30帧窗口，CPU内存及SSD可能成为瓶颈；内存紧张时设WORKERS=0。未在实际5090上测出最优批量。保留FP32以避免FNO FFT在非2幂尺寸上的半精度限制。脚本会覆盖同名outputs训练结果，重复实验先备份。

结果：outputs/cartesian与outputs/taoca中的best.pt、history.json、evaluation.json、snapshots；汇总outputs/comparison.json；日志outputs/cartesian.log及taoca.log。准确率主要比较共同物理覆盖区域的回映射误差，同时查看覆盖率与真实场往返误差。共享磁盘的双卡并行耗时受I/O竞争影响，论文计时需要补充单独运行和交换GPU重复实验。
