# 统一组合优化器实施状态

更新日期：2026-08-31。本文用于周期性检查实现是否偏离已经冻结的架构和产品 policy。

面向代码审阅、严格按当前源码行为整理的分层、调用链、数学转换与已知边界见
[`current_implementation_architecture.md`](current_implementation_architecture.md)。本文继续只维护
里程碑状态和性能 gate。

## 已实现

| 层 | 当前实现 | 验证 |
|---|---|---|
| 公共 contract | 不可变 data/problem/result/policy/certificate/fingerprint | contract tests |
| 单期实盘接口 | `optimize(data=..., ...)`；黑名单、冻结、只买/只卖、定制权重均为 problem-local | asset-trade tests |
| 静态校验 | shape、有限值、风险单位/日期、PSD、bounds、目标依赖、因子名 | 聚合 validation tests |
| LP | canonical sparse model → direct HiGHS | 单元与真实 v5 |
| QP | low-rank + diagonal factor variables → official PIQP 0.6.4+ compact | 单元与真实 v5 |
| factor-QCQP | PIQP 参数 QP frontier；同日 workspace 内 continuation；可选 HiGHS LP 严格筛选 | 单元与真实 v5 2% TE |
| fallback | MOSEK license 可用时优先，否则官方 Clarabel 0.11.1 QDLDL | 强制失败路由测试 |
| 独立验收 | 所有 backend 后统一复算约束、TE、目标和清理后可行性 | backend tests |
| 不可行诊断 | 显式 deep Phase-I、线性最小换手率、最小 TE | diagnostic tests |
| 多期 | close-to-close 自然漂移、previous/fixed theta、显式换手恢复 | sequence tests |
| 手动数据 | 严格 `(dt,sid)`、同日 risk/benchmark/alpha、显式覆盖策略 | data-source tests |
| 长序列 manifest | `PreparedPortfolioRun` 先逐日聚合预检、求解时仅物化当前日 | range tests |
| tuda2 | 延迟导入；风险/基准/收益均全区间一次读取并复用；C2C 收益区间复合 | fake-adapter I/O tests |
| 二进制核心 | 公共 Python facade + `py.typed`；数值 `_core` 选择性 Cython 编译且无 stub/source 分发 | 架构、wheel 与全量行为回归 |
| AI 语料 | wheel 内置 `LIBRARY.toml` 及 `references/api_overview.md`、`recipes.md`、`gotchas.md` | TOML、版本和 wheel 内容检查 |

正常求解失败返回结构化 `OptimizationResult`，用于批量统计、fallback 审计和按
fingerprint 选择单日 deep 诊断；需要异常式取权重时调用 `result.require_weights()`。
输入/schema/非法模型仍在 backend 建立前抛错。

## 本轮漂移检查与修正

1. 文档曾遗留“benchmark 缺口在阈值内默认归一化”，已修正为：任何缺口默认报错；只有
   显式选择 `renormalize_within_tolerance` 才允许阈值内归一化，超过阈值始终报错。
2. Clarabel 运行依赖明确使用官方 `clarabel==0.11.1`。历史自编译 MKL wheel 只属于后端
   benchmark 证据，不是生产要求；PIQP 强制使用已合入上游修复的官方 `piqp>=0.6.4`。
3. factor-QCQP 公共风险单位保持年化 decimal。frontier 内部乘 10,000 只用于延续已测 theta
   数值尺度，certificate 已按相同尺度还原，不改变原模型或公开 TE。
4. PIQP 每日重新建立 workspace；只有同一日 theta continuation 更新 q。修复后的 wheel 不做
   无条件 cold retry，仅在 update 生命周期失败时冷重建一次。
5. 旧实现把 `tradable == 0` 当作清仓，且部分名单越界后静默忽略。新版统一为单期
   `AssetTradeConstraints`：non-tradable/frozen 固定于期初权重，blacklist 强制为零，
   not-buyable/not-sellable 为相对期初权重的单边约束；未知证券和清单重叠默认在 backend
   建立前报错。交易指令为满足现实可交易性可对该证券局部覆盖 active-weight bound，覆盖前
   区间和指令来源记录在 canonical registry，不影响其他证券。
6. tuda2 是本地 FizzDB 快速 I/O，风险模型、基准和日收益按完整优化区间各一次取足，预检后
   直接复用；因子名称元数据按 data source 缓存。组合优化的数据规模由日期数、样本空间和
   固定因子数决定，可在运行前估算，因此核心接口不增加 chunk/LRU 分支和重复 I/O。
7. `_core` 依赖方向已冻结为单向：上层只通过 `_core` 根入口构造数值 payload、options 和
   solve handle；core 不得导入 `portfolio_types/api/validation/diagnostics`。固定 policy 每个
   optimizer 转换一次，多期每日只跨边界调用一次，theta continuation 不跨层往返。
8. 多期问题在首个求解前完成全区间静态预检；逐日热路径直接编译经过预检的模板，不再重复
   扫描风险矩阵和约束数组。链式模式每日唯一变化的期初权重由受控 C2C 漂移生成。
9. 新架构发行版本从 `3.0.0` 开始；包内版本、wheel 元数据和 AI catalog 由自动测试保持一致。

## 单期实盘入口

底层接口仍是不可变的 `solve(PortfolioProblem(...))`；便利入口不会保留名单状态：

```python
result = optimizer.optimize(
    data=today_data,
    objective=MaximizeAlpha(),
    constraints=constraints,
    blacklist=["sid_to_exit"],
    frozen=["suspended_sid"],
    not_buyable=["sell_only_sid"],
    not_sellable=["buy_only_sid"],
    weight_overrides={"special_sid": (0.001, 0.003)},
)
```

名单约束同时适用于 LP、QP 和 factor-QCQP，因为它们在 solver 路由前编译为同一个 canonical
资产域。`optimize_range` 不接受 `asset_trade`：操作性名单依赖当日真实持仓和交易状态，主要
服务单期实盘请求，不在多期研究 facade 中提前表达或广播。

依赖 tuda2 的同日入口仍由 optimizer facade 调用：

```python
result = optimizer.optimize(
    data_source=Tuda2DataSource(risk_model="datayes"),
    date=date,
    universe=universe,
    benchmark_sid="000852.SH",
    initial_weight=initial_weight,
    objective=MaximizeAlpha(),
    constraints=constraints,
    alpha_spec=AlphaSpec(),
)
```

它只读取指定日期，不取持仓漂移收益，并通过 `InMemoryDataSource.build_problem()` 只物化一次
dense 风险矩阵，再进入完全相同的单期核心。多期同样使用
`optimizer.optimize_range(data_source=..., schedule=..., ...)`，普通用户不需要逐日构造
`PortfolioProblem`；低层 `solve_sequence(problems, ...)` 只服务每日业务模型确实不同的场景。

## 前端性能检查

通用递归 canonical fingerprint 曾使 5200 assets × 47 factors 的合成 factor-QCQP
`prepare()` 达到约 0.93 秒，其中约 0.87 秒为变量/约束登记逐字段散列。固定 schema 批量编码
后，同一开发机单次约 0.108 秒（约 8.6 倍改善），且 hash 仍覆盖完整 canonical 数值和登记
元数据。可复现入口：

```bash
PYTHONNOUSERSITE=1 python benchmarks/frontend_prepare_synthetic.py
```

加入单期资产交易指令后再次回看，逐资产审计 registry 曾成为新的主要 Python 开销：每天
为 5,200 只股票创建两组来源容器和空 metadata，再由 `ConstraintRecord` 复制一次。当前
resolver 改用共享不可变来源 tuple，普通资产 metadata 使用 `None`，compiler 按来源组合
intern 不可变审计 mapping；只有黑名单、冻结等例外证券持有独立 metadata。数学模型、审计
字段和 canonical fingerprint 覆盖范围不变。

同一开发机 5200×47 合成 factor-QCQP 的最新 7 次 warm `prepare()` median 约 0.0408 秒
（范围 0.0368–0.0642 秒），相比逐资产 metadata 优化前约 0.065 秒下降约 37%。该数字受
开发机调度波动影响，应持续以生产机 p50/p95 gate 为准。

真实 v5 35 日独立冷启动（绝对权重 `[0,1]`、active cap 1%、风格 ±0.6、行业 ±0.05）：

| 场景 | 路由 | 总耗时中位数 | 均值 | P95 | 状态 |
|---|---|---:|---:|---:|---|
| LP | HiGHS | 0.0982 s | 0.1090 s | 0.1433 s | 35/35 optimal |
| 风险惩罚 QP | PIQP | 0.0945 s | 0.0998 s | 0.1264 s | 35/35 optimal |
| 2% factor-QCQP | PIQP frontier | 0.2577 s | 0.2788 s | 0.3617 s | 35/35 optimal |
| 2% factor-QCQP + 显式 LP 筛选 | HiGHS/PIQP | 0.0989 s | 0.1554 s | 0.4073 s | 31 LP 接受、4 PIQP |

有 factor variables 时，风格/行业约束已改为直接的稀疏 factor bounds，避免在 KKT 中重复
第二份 dense exposure block；QP alpha 使用平移不变的 benchmark-centered scaling reference。
详见 `v5_unified_optimizer_real_benchmark.md`。这些开发机结果不替代生产机多日 gate。

## 尚未完成

1. canonical sparse structure/template 跨日复用。当前只做了单次 compilation 内的 registry
   flyweight；尚未引入跨日 cache。未来只允许复用 Python/CSC 结构，不恢复 PIQP 跨日
   workspace。
2. 二阶段近似 alpha 最优集合内最小化持仓距离：仅保留设计位置，第一版默认关闭。
3. 旧 `opt/linopt` 兼容包装器和迁移回归；旧模块目前仍可延迟导入，但尚未改走新核心。
4. 生产机 250/2500 日前端时间、内存增长、输出策略与 fallback P99 gate；全量风险数据内存
   容量也应在运行前 gate，不在优化日循环中通过额外 I/O 补救。

全量 I/O 优先吞吐。粗略上界可用 `n_dates × n_assets × n_factors × dtype_size` 估算 exposure
主体，再为 pandas index、行业展开、benchmark 和收益保留余量；生产资源不足应在启动前明确
失败或缩短任务区间，而不是让 2500 次优化在日循环中频繁访问 tuda2。
