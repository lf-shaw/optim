# 当前 optim 模块数据校验与序列逻辑审计

本文审计当前 `optim/linopt.py`、`optim/opt.py`、`optim/solver.py` 以及其依赖的
`carry.utils.reindex/scale_by_dt`。目的是保留合理的业务语义，并明确新版不能照搬的行为。

## 1. `linopt` 的现有行为

### 1.1 Universe 与优化日期

`linopt.multioptimize()` 首先调用 `__validate_universe()`：

- 要求非空 `DataFrame`；
- index 必须为两层，名称严格等于 `['dt', 'sid']`；
- 必须包含 `alpha`、`member`、`tradable`；
- 只对 dtype **恰好为** `float64` 的列检查 NaN。

优化日期取自：

```python
opt_dts = universe.index.get_level_values("dt").unique()
```

因此 universe 不只是资产池，也是调仓日程和 alpha 的载体。这一设计合理，特别适合
月度/不规则调仓策略，应在新版保留。

现有检查的缺口：

- 没有验证 `(dt, sid)` 是否唯一；
- 没有在入口明确验证按日期/证券排序和日期单调；
- 没有检查 `inf/-inf`；
- 非 `float64` 数值列可能绕过 NaN 检查；
- 没有逐日验证资产集合非空、alpha 覆盖和权重列语义；
- `member` 虽然强制存在，但没有直接参与求解约束或资产过滤；
- 单日 `optimize(dt, universe, ...)` 假定调用方传入的 universe 只有该日，但没有验证。

### 1.2 `carry.utils.reindex` 的实际语义

`linopt` 大量使用：

```python
carry.utils.reindex(data, index=universe.index)
```

`ffill` 默认是 `False`，因此它按照 `(dt, sid)` 精确匹配，不会自动拿前一日期补数据。
这是正确的同日语义，应在新版保留并在公共契约中明确，而不是改成隐式 as-of 对齐。

`carry.utils.reindex` 的文档要求输入按 `[dt, sid]` 严格排序；现有 `linopt` 入口没有在每个
输入上显式检查这一点。新版应先标准化/检查索引，再对齐。

### 1.3 Benchmark 校验与覆盖率

benchmark 要求：

- 非空、两层 `(dt, sid)` index；
- 包含 `weight`；
- `weight` 不含 NaN。

如果调用方传指数代码，当前实现按 universe 的优化日期调用：

```python
tuda2.get_index_weight(benchmark, dts=opt_dts, type="daily")
```

即请求同日数据，没有 forward-fill。

随后 `__align_benchmark_with_universe()` 把 benchmark reindex 到 universe，并计算：

```python
avg_weight = aligned_benchmark.weight.sum() / number_of_dates
```

如果 `avg_weight < 1 - weight_tolerance` 才报错，最后把缺失填成 0。

这不是逐日覆盖率，而是全区间的平均覆盖质量。例如某日覆盖 99%、另一天超过或接近
100%，可能整体通过；单日严重缺口也可能被长区间平均稀释。现有实现还没有验证：

- benchmark 每日原始权重和是否接近 1；
- 权重是否非负、有限、是否有重复 sid；
- 每日缺失 benchmark 权重及最大值；
- 接受小额缺失后是否应在 universe 内归一化。

当前行为是“缺失填 0、不归一化”。此时 benchmark 在优化宇宙中的总权重可能小于 1，
会使主动权重和不为 0，并影响主动敞口和 TE。新版必须把接受阈值、处理动作和归一化因子
做成显式策略，并至少逐日校验。

### 1.4 初始持仓校验

`init_portfolio` 必须是：

- `Series`；
- index 名称为 `sid`；
- Series 名称为 `weight`。

它被 reindex 到 universe 首日资产，缺失填 0；如果对齐后权重和与 1 的差超过
`weight_tolerance` 则报错。这可以发现持仓中有较大权重资产不在首日 universe 的情况。

缺口包括：

- 不检查负权重、NaN/inf 和重复 sid；
- `turnover_limit` 参数传入校验函数但没有被使用；
- 只验证首日，不形成后续持仓映射/缺失质量的结构化报告；
- 允许的小额丢失没有记录具体资产，也没有明确是否归一化。

### 1.5 风格、行业与额外属性

只有设置对应约束时才读取风险敞口。tuda2 调用使用风险模型原始日期数据，再通过
`reindex(..., ffill=False)` 对齐到 universe。

现有代码检查约束 key、tuple 格式和上下限顺序，但对对齐后新增的 NaN 没有统一再次
检查。`all` 与 override 的展开逻辑分别散落在单期和多期函数中。

行业通过 `industry` 分类列 `get_dummies` 展开。因子元数据和最终矩阵列没有形成一个可供
诊断使用的约束登记表。

### 1.6 交易空间、清单与单股边界

当前 `linopt.optimize()` 的主要规则：

- `tradable == 0` 时把绝对权重上限设成 0；
- blacklist 同样把权重上限设成 0；
- freeze 将上下限固定为初始权重；
- cap/only-sell 把上限设为初始权重；
- customized weight 设置固定值或区间；
- active upper bound 会为部分特殊资产扩大到足以容纳被强制的绝对权重。

需要注意：将 `tradable == 0` 直接解释成“必须清仓”与真实停牌资产通常“不能交易”的
语义可能相反。新版必须区分：

```text
not_buyable / not_sellable / frozen / excluded
```

并从初始持仓推导上下限，不能只使用一个 `tradable` 布尔值。

现有清单在全区间 sid 并集上校验，未保证清单证券存在于每个具体优化日。若证券只在其他
日期出现，部分日可能在 `sids.index()` 处失败。customized float 分支还引用了错误变量，
需要由迁移测试覆盖。

### 1.7 连续持仓自然漂移

`linopt.multioptimize()` 在多期时调用 tuda2 日收益率，从首日至末日遍历每个交易日：

```python
init = init * (1 + daily_return)
init = init / init.sum()
```

只在 `dt in opt_dts` 时求解。也就是说，即使 universe 只给月度调仓日，持仓也会在中间
交易日按收益逐日自然漂移。这一原则正确，应作为新版连续多期优化的强制语义。

当前不足：

- 收益缺失直接填 0；
- 没有记录缺失收益对应的持仓质量；
- 收益锚点被写死为 close/close；该锚点与风险模型 close 快照一致，但没有和 t+1 open
  执行层、planned/realized turnover 形成明确协议；
- 如果求解失败，会把 `init_portfolio = None`，下一次优化直接丢失真实持仓和换手率约束。

最后一点是严重的回测语义错误。某日优化失败并不代表真实持仓消失；新版必须继续漂移
上一个实际持仓，除非调用方明确提供其他交易执行结果。

### 1.8 不可行与自动放松

当前行为可能：

- 逐步放宽 turnover；
- 完全丢弃 turnover 再求；
- 跳过当日；
- 捕获宽泛异常并只输出 warning。

放宽后的问题和原问题没有结构化区分，也没有具体不可行原因。新版诊断只能报告需要的
最小松弛，不能默认修改原问题。若用户选择放松，应生成新的 request/attempt，并保留完整
审计记录。

### 1.9 输出后处理

单日结果使用：

```python
abs(weight) >= 10e-5
```

实际阈值是 `1e-4`（1 bp），不是期望的 `1e-5`（0.1 bp）。多日汇总随后调用
`scale_by_dt` 重新归一化。删除和归一化都发生在 solver 验证之后，可能重新改变预算、
换手率、敞口和 TE，却没有再次验证。

新版应把 raw solution 与 cleaned solution 分开，记录累计删除质量，并对最终返回权重重新
执行全部约束检查。

## 2. `opt` 的现有行为

### 2.1 日期和风险数据

`opt.multioptimize()` 默认从 benchmark 日期生成优化日，而不是 universe；随后按这些日期
裁剪 universe。它使用旧的 `get_rm_data` 接口读取 exposure/cov/spec_risk，并硬编码
CNE5 风格和行业列表。

新版应统一由 universe/schedule 指定调仓日期，并通过当前 tuda2：

```python
get_risk_model("exposure" / "cov" / "spec_risk", dts=dates, model=...)
```

读取指定模型，因子名称来自风险模型元数据，不能硬编码。

### 2.2 合并和 benchmark 覆盖

`opt` 将 alpha、exposure、benchmark、spec_risk reindex 到 universe，benchmark 缺失填 0，
随后对整个合并表 `dropna()`。这样会删除 alpha 或风险数据缺失资产，但删除后没有重新
计算每日 benchmark 覆盖权重，也没有报告被删除的 benchmark 成分。

因此旧 `opt` 的输入数据可能在不知情时改变优化资产池和有效 benchmark。新版禁止这种
无审计的 `dropna()`；每种缺失必须按字段和日期报告，并由显式策略决定 error/drop asset。

### 2.3 连续持仓漂移

当 `use_pre_portfolio=True` 时，`opt` 用相邻优化日期价格的 `pct_change()` 漂移上一期
权重并归一化。对月度日期而言，它直接得到两次调仓日期之间的累计价格变化，原则上也考虑
了自然漂移。

但它只使用 `close`，读取的 `adj_factor` 没有用于计算，缺失收益会导致持仓被 `dropna()`
删除。新版应统一使用可解释的复权收益源，并检查缺失收益持仓质量。

当 `use_pre_portfolio=False` 时，初始持仓和 turnover 都被忽略。新版不再用布尔参数混合
独立优化与连续回测，而使用明确的 `independent` 与 `chained` 序列类型。

## 3. 新版应保留和修正的语义

| 主题 | 保留 | 修正 |
|---|---|---|
| 调仓日 | universe `(dt,sid)` 的日期 | 验证唯一、排序、逐日非空 |
| alpha | 可放在 universe 中 | 是否必需由 objective 决定 |
| 数据日期 | 精确同日 reindex | 任一必需数据缺失即明确 error/skip，不前值替代 |
| benchmark | 与 universe 对齐 | 逐日检查原始总权重、覆盖质量和处理动作 |
| 初始持仓 | 对齐首日 universe | 检查非负/有限/重复，报告缺失质量 |
| 连续持仓 | 调仓间自然漂移 | 统一收益协议；失败日继续持有而非清空 |
| 不可行 | 可选择跳日 | 先返回结构化原因，不静默放松 |
| 输出清理 | 忽略极小权重 | 使用 `1e-5`，记录删除质量并重新验证 |
