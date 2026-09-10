# 同后端 LP 多日性能与持仓收益缺失排查

补充：本文是早期内部调用实验，不是完整多期吞吐基准；旧代码快照也不是用户随后确认的
2.0.2（对应提交 `152b433`）。后续只测新版的完整入口结果见
[当前版本多期性能审计](current_range_performance_audit.md)，不要将本文 0.160 秒换算成生产吞吐。

测试日期：2026-09-10。结论：本地样本没有复现生产机“旧版 3.7 次/秒、新版 2.5 次/秒”的
整体退化；不能据此否认生产观测。逐日数据物化不是本地主要成本，模型编译、审计哈希仍值得
优化；不可行路径的新版本成本确实更高。本轮不修改性能关键实现或降低校验精度。

## 对照范围

- 生产参数是纯 LP：主动权重 0.004，总主动 1.8，换手 0.05；风格 ±0.3、size ±0.2、行业
  ±0.01。无风险预算、无成分股持仓合计下限。两边固定 MOSEK。
- 本地使用 v5 的 35 个风险日期，2025-05-06 至 2025-06-24，约 5146–5153 只股票。
  本地基准为中证 1000，**不是生产中的沪深 300**。
- 旧模块从 Git `f5013b1^` 中只读加载 `linopt.py` 与 `solver.py`，没有恢复废弃代码。
  当前模块直接使用工作区源码，核心为 Python 源文件，并非生产安装的 Cython wheel。
  MOSEK 版本为 11.2.3，两个路径使用同一环境、默认线程设置。
- 测量旧 `linopt.optimize(enable_parameter_validation=False)`（对应旧多期循环内调用），
  对照当前 `_solve_prevalidated`，加上当前内存数据源逐日物化时间；不把它冒充两套外部取数入口
  的完整端到端回测。当前全区间静态准备单独计时，不包含 Excel 解析和实验表格构造。
- 每日两个路径收到相同初始持仓、alpha、基准和敞口。首日免换手，随后使用当前结果推进；
  收益设为显式零收益，隔离收益数据差异。双方不可行时实验继续持有上一状态，不自动放宽。
  新旧求解次序逐日交替，双方预热。旧版清理前原始权重用于独立复核；不把两个清理策略后的
  持仓误差当作求解误差。原始模板、基准和约束不被修改。

## 两轮 35 日结果

下表为无 profiler 插桩的 wall-clock 均值，每期秒数；逐日物化另计。

| 日期分组 | 期数 | 新版第 1 轮 | 旧版第 1 轮 | 新版第 2 轮 | 旧版第 2 轮 |
|---|---:|---:|---:|---:|---:|
| 全部 | 35 | 0.1758 | 0.2115 | 0.1782 | 0.2094 |
| 可行 | 29 | 0.1587 | 0.2064 | 0.1603 | 0.2045 |
| 不可行 | 6 | 0.2584 | 0.2362 | 0.2643 | 0.2330 |

两边状态完全一致。不可行日期为 6 月 16、17、18、19、20、23 日。第 2 轮旧解通过当前约束
独立复算，最大残差 `1.32e-9`，与新解的原始 alpha 目标差最大 `1.43e-7`。这不是权重相似度
保证。实验向下一日资产域投影时未遗漏非零持仓。

当前全区间静态准备约 0.24–0.27 秒；平均逐期数据物化约 0.0031–0.0035 秒。第 2 轮可行期：

| 新版阶段 | 均值/期 |
|---|---:|
| 数据源物化及本实验持仓组装 | 0.0034 s |
| 编译及上层准备（结果中的 prepare_s） | 0.0457 s |
| MOSEK 原生模型建立 | 0.0076 s |
| MOSEK 原生求解 | 0.1019 s |
| 独立验收 | 0.0031 s |

当前结果的 `compile_s` 在此路径为 0，不代表没有编译；编译已计入 `prepare_s`。原生耗时分项
也不是完整公共调用的全部成本，尤其不可行结果还需构造原生证书、贡献映射等对象，应以外部
wall-clock 为准。

单独 cProfile 指向 `_configure_variables`、`semantic_hash` 和 `canonical_hash` 等热点。
这些是可审计模型编译成本，不是 pandas 对齐成本。profiler 会显著放大 Python 小函数调用时间，
因此不把 profiling 秒数混入上表，也不据此声称移除某个函数能等额提速。

## 为什么仍需要生产样本

1. 中证 300 与中证 1000 会改变主动权重、总主动权重约束的几何结构及内点迭代数。
2. 新版 MOSEK 对含 L1 边界的原生 primal tolerance 使用更严格的 `1e-10`；旧 Fusion 默认
   未显式配置同样精度。两者业务约束相同，不表示原生参数与 formulation 相同。
3. 旧 linopt 输出会丢弃绝对值低于 `10e-5`（即 `1e-4`）的权重；新版清理及验收规则不同。
   两套独立链式回测不保证从第二期开始仍在解同一个问题。
4. 旧 LP 通常只取暴露数据；新版同时构造完整风险数据并计算业务指标。这可能影响准备阶段
   的 I/O，但本地逐期物化仅毫秒级，不能直接解释生产约 0.13 秒/期的差距。

下一步需要生产同日期的 risk / benchmark / alpha / tradable，或几个慢 LP 的 repro，另附
新旧安装版本、MOSEK 版本和计时是否含取数/预检。不要为了复现性能而先改变缺失收益策略。

## 收益缺失为何旧版不报错

旧 `linopt.multioptimize` 对缺失收益直接执行 `fillna(0)`；新版复合收益使用
`prod(skipna=False)`，只要区间内某天缺失，该股票区间收益就为 NaN，随后按实际持仓绝对权重
检查允许缺失比例。旧版不报错不等于没有缺失数据。

本次增强：

- 异常包含上一调仓日、失败调仓日、缺失比例、容差和最大缺失股票；完整明细为
  `SequenceDataError.evidence` DataFrame，区分无股票标签与有标签但值缺失。
- `partial_result` 保留已完成步骤；`previous_weight` 与 `holding_return` 保留失败前状态。
- `optimize_range` / `solve_sequence` 可指定 `failure_dump_dir`，仅漂移数据异常时写入新文件，
  默认不写磁盘，不自动诊断或再次求解。异常仍抛出，路径见 `dump_path`；写入失败见 `dump_error`。
- `exc.dump("failure.json.gz")` 可手动导出。文件不含风险矩阵，不能直接传给 `load_repro`。
  已记录的是区间复合收益，具体缺失日仍须按股票及区间查询日度源数据。
- 修正缺失持仓质量按绝对值合计，避免多空相互抵消掩盖缺失。零持仓股票不进入故障明细。

```python
try:
    sequence = optimizer.optimize_range(
        data_source=source, schedule=schedule, benchmark="000300.SH",
        constraints=constraints, objective=MaximizeAlpha(), alpha_spec=AlphaSpec(),
        show_progress=True, failure_dump_dir="tmp/sequence_failures",
    )
except SequenceDataError as exc:
    failure = exc
    print(exc.evidence)
    print(exc.dump_path)
    partial = exc.partial_result
```

请勿在查看原因前一律填零或提高容差：停牌、新上市、退市、数据缺口及错误日期参数的处理
依据不同，需要根据缺失的股票与日期核实。

## 实验文件

- 脚本：`benchmarks/legacy_mosek_frontend_audit.py`
- 第 1 轮：`tmp/legacy_mosek_frontend_audit_full_20260910/summary.json`
- 第 2 轮：`tmp/legacy_mosek_frontend_audit_full_repeat_20260910/summary.json`
- 每轮目录包含单独 profiling 输出；不导出权重，不修改原始 Excel/CSV。

```bash
python benchmarks/legacy_mosek_frontend_audit.py --dates 35 \
  --output tmp/legacy_mosek_frontend_audit_new
```

脚本依赖仓库历史和已有 v5 实验读取工具，不是独立生产发布包。输出目录必须为新目录。
