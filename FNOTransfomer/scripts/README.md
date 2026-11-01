# Scripts 索引

工具脚本按文件名前缀分类，暂时维持同一目录，以保留 `ROOT = project/scripts/..` 等现有路径约定。

| 前缀 | 作用 | 常用示例 |
|---|---|---|
| `prepare_*` | 从本地原始 CFD 数据生成 paired、joint 或版本化缓存 | `prepare_all.py`、`prepare_joint.py` |
| `validate_*`、`audit_*` | 检查输入数据、映射权重、几何或物理一致性 | `audit_paired.py`、`audit_thermodynamics.py` |
| `check_*`、`test_*` | 运行接口契约、实现门槛或局部训练检查 | `check_cuda.py`、`test_physics_v2.py` |
| `compare_*`、`recompare_*` | 统一支持域上比较已完成的路线或版本 | `compare_physics_v2.py`、`recompare_completed_versions.py` |
| `analyze_*`、`diagnose_*`、`summarize_*` | 生成误差分区、激波剖面和诊断报告 | `analyze_physical_v7.py`、`diagnose_shock_profiles_v5.py` |
| `build_*`、`select_*` | 构建隔离入口或依据预先指定的验证协议选择候选 | `select_allocation_v3.py` |
| `package_*`、`verify_package.py` | 旧版服务器打包和校验工具 | 请先检查对应协议中的数据打包假设 |
| `run_*` | 旧实验编排或 worker 入口 | 正式本机入口优先使用项目根目录的 `run_*_local.py` |

注意事项：

- 名称含 `test` 的脚本是项目实现检查/诊断，不表示存在一个已通过的完整自动化测试套件。
- 版本后缀 V3–V8 的脚本用于复现各自实验，不应任意混合不同版本的缓存、检查点和配置。
- 脚本会创建或读取 `datasets/`、`outputs/`；运行前先阅读 [`../README.md`](../README.md) 和对应 [`../docs/README.md`](../docs/README.md)。
