# 生产多期数据导出与开发机重放

该工具用于把一次 `Tuda2DataSource` 多期请求转换成与数据库无关的可移植数据包，便于定位
生产与开发环境的准备、编译和求解性能差异。它不是用户手册的一部分，也不改变普通优化接口。

## 生产机导出

在已经构造 `source`、`schedule`、`constraints` 和 `optimizer` 的 Notebook 中运行：

```python
from optim import AlphaSpec, MaximizeAlpha
from optim.devtools.sequence_data_package import export_tuda2_sequence

package_path = export_tuda2_sequence(
    "production_csi300_202607_202608.zip",
    data_source=source,
    schedule=schedule,
    benchmark="000300.SH",
    constraints=constraints,
    objective=MaximizeAlpha(),
    alpha_spec=AlphaSpec(),
    solver_policy=optimizer.policy,
)
print(package_path)
```

参数应与实际 `optimize_range` 调用一致。若实际调用设置了下列参数，导出时也必须原样传入：

```python
sequence_policy=sequence_policy,
initial_weight=initial_weight,
benchmark_policy=benchmark_policy,
tradable_universe=tradable_universe,
extra_attribute_columns=extra_attribute_columns,
independent_initial_weights=independent_initial_weights,
holding_period_returns=holding_period_returns,
```

导出过程会重新执行一次 tuda2 批量取数和全区间预检，但不会调用求解器。默认不覆盖已有文件。
需要 `pyarrow`，可安装 `optim[devtools]`；Parquet 使用 zstd 压缩，ZIP 不会再次压缩
Parquet。大于 4 GiB 时支持 ZIP64。

包中包含以下敏感输入，应只在授权的内部环境传输：

- schedule 中实际消费的 alpha、tradable 及显式额外属性；
- 已对齐的风险敞口、协方差、特异风险和基准权重；
- 链式漂移使用的区间收益、显式初始持仓或独立模式逐日期持仓；
- 目标、约束、序列策略、求解策略、风险来源版本和依赖版本。

包中不包含逐日优化结果权重、求解器 workspace、license、环境变量、数据库地址或日志。
数据使用白名单 JSON 与 Parquet，不使用 pickle；每个成员都有 SHA-256 和长度校验。

## 开发机重放

命令行可直接重放并生成不含权重的逐期性能表：

```bash
python -m optim.devtools.sequence_data_package \
  production_csi300_202607_202608.zip \
  --backend auto \
  --show-progress \
  --performance-csv tmp/production_replay_auto.csv
```

将 `--backend auto` 改成 `mosek` 可做同数据对照。不传该参数时使用包中保存的策略。
加载和校验数据包发生在求解计时之前；输出的 `wall_s` 从完整 `optimize_range` 重放开始，包含
内存数据源预检、逐期物化、模型编译、求解、验收、持仓漂移和结果组装，不包含 Parquet 读取。

也可以在 Python 中分别控制加载和求解：

```python
from optim.devtools.sequence_data_package import (
    load_sequence_package,
    performance_frame,
)

case = load_sequence_package("production_csi300_202607_202608.zip")
result = case.solve(backend="auto", show_progress=True)
performance_frame(result).to_csv("tmp/production_replay_auto.csv", index=False)
```

默认拒绝超过 8 GiB 解压大小的数据包；确有需要时，命令行可显式提高
`--max-uncompressed-gib`。加载过程逐成员校验且只向临时目录写入，退出后自动清理。

## 对比原则

同一数据包应至少顺序测试两轮 auto 和 MOSEK，并交换运行顺序，首轮视为预热。不要并行运行，
也不要用 profiler 下的墙钟速度与普通运行比较。生产原始运行与开发机重放的 `prepare_s` 差异
可用于判断问题来自数据结构还是生产 wheel/运行环境；数据库 I/O 不在逐期 `total_s` 中。
