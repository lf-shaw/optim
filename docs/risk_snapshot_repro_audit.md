# 风险快照接口：真实 repro 对齐回归

本记录验证新增 `Tuda2DataSource.create_risk_model` 和带标签风险对象没有改变既有单期
入口的数值输入，测试日期为 2026-09-10。

## 样本与版本

- 基线：Git tag `v3.2.4`，仅导出源码到隔离目录，在独立 Python 进程运行。
- 对照：当前实现，使用相同 default 环境、依赖及固定 `backend="auto"`。
- 三份真实包均为 2026-08-14、5208 只股票，保留原目标、所有约束及 8% 风险预算。
- `tmp/pp.tar.gz` 为旧格式，不修改包、不迁移 schema；当前加载器拒绝，因此未纳入。

风险表来自包内真实数组，不连接 tuda2 存储。回放适配器保留真实股票、因子、风险数值，
模拟未物理存储的常数 country，并将协方差列倒序以检查按行因子对齐。不能将本测试称为
实际 tuda2 存储访问验证，也不能称为真实多日回测。

## 已执行检查

1. 原有数据源入口重建的数据与包内原数据逐项内容摘要一致：股票、因子名称及类型、alpha、
   基准、期初持仓、tradable、暴露、协方差、特异风险，以及存在时的额外属性。
2. 基线与当前版本的原入口输入摘要一致，实际求解结果进行权重、目标、TE 对照。
3. 当前新快照入口按同序、倒序、固定种子 20260910 随机顺序组装；恢复股票顺序后，
   对输入数组执行逐项精确比较。每隔一只取子集时，风险行及标签正确裁剪。
4. 请求缺失股票时报错；快照复用期间不重复调用风险取数。
5. 新快照入口与当前原入口分别实际求解，权重和目标值进行对照。

## 结果

| 包 | 因子数 | 两版本原入口输入 | 最大权重差 | 目标值差 | TE 差 | 新快照与原入口权重差 |
|---|---:|---|---:|---:|---:|---:|
| pp_20260814.tar.gz | 42 | 完全一致 | 0 | 0 | 0 | 0 |
| pp_20260814_csi1000.tar.gz | 47 | 完全一致 | 0 | 0 | 0 | 0 |
| pp_20260814_csi2000.tar.gz | 47 | 完全一致 | 0 | 0 | 0 | 0 |

所有求解均为 optimal，实际后端为 clarabel_qdldl。单次 wall-clock 约 0.21–0.36 秒，
未做足够重复或控制缓存顺序，因此这些时间不用于宣称性能改善。

数据源测试另覆盖：原单期及多期入口不调用新单日工厂，每种风险数据仍只批量获取一次；
跨行业分类变更日的协方差行列处理保持原约定。完整测试为 388 passed、1 skipped。

## 复跑

使用 `benchmarks/risk_snapshot_repro_audit.py`。通过 PYTHONPATH 指向基线源码或当前仓库，
基线不传 `--snapshot`，当前传该选项。输出目录必须不存在，避免覆盖已有审计结果。

```bash
PYTHONPATH=/path/to/v3.2.4-source python benchmarks/risk_snapshot_repro_audit.py \
  --output tmp/risk_audit_baseline_new \
  tmp/pp_20260814.tar.gz tmp/pp_20260814_csi1000.tar.gz tmp/pp_20260814_csi2000.tar.gz

PYTHONPATH=. python benchmarks/risk_snapshot_repro_audit.py --snapshot \
  --output tmp/risk_audit_current_new \
  tmp/pp_20260814.tar.gz tmp/pp_20260814_csi1000.tar.gz tmp/pp_20260814_csi2000.tar.gz
```

本次摘要分别位于 `tmp/risk_snapshot_audit_v324/summary.json` 和
`tmp/risk_snapshot_audit_current/summary.json`，包含实际导入路径，避免误用同一个版本。
对应小型 NPZ 仅用于本地跨进程权重比较，不需要随报告传输，也不纳入发布包。
