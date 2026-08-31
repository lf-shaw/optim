# 当前 API 使用与审计范例（v5）

更新日期：2026-08-31。

这不是 v5 文件格式的正式用户手册。它只回答一个审阅问题：把现有 v5 数据转换成
`PortfolioProblem` 后，当前统一 API 怎样完成单期、链式求解、结果检查和不可行诊断，前端
开销是否吞掉了后端优化收益。

可运行脚本为 [`examples/v5_current_api_audit.py`](../examples/v5_current_api_audit.py)。脚本的
v5 Excel/CSV 读取仍复用历史 benchmark adapter；进入 `PortfolioProblem` 后不再调用旧求解器。

## 1. 快速运行

仅依赖 `tmp/v5` 三份数据，使用显式零收益推进 5 日短链：

```bash
python examples/v5_current_api_audit.py \
  --max-dates 5 \
  --single-repeats 5 \
  --output /tmp/v5_current_api_zero_5d.json
```

零收益并不是生产收益假设，它只是让

$$
x_{t,\mathrm{pretrade}}=x_{t-1,\mathrm{target}}
$$

从而在不增加第四份数据的情况下审计链式状态。若有 C2C 价格数据，可以显式传入：

```bash
python examples/v5_current_api_audit.py \
  --max-dates 35 \
  --single-repeats 3 \
  --price-hdf /home/shao/workspace/repos/XEngine/xserver/data.h5 \
  --diagnose-chain-failure \
  --output /tmp/v5_current_api_c2c_35d.json
```

该历史 HDF 的 `px` 是 fixed format，范例会一次读取完整表；这只是本地审计输入，不代表
tuda2 生产取数方式。生产路径由 `Tuda2DataSource` 一次获取完整区间的日度收益，再在内存中
复合各调仓区间。

## 2. 单期用户怎样调用

当数据层已经给出严格同日、位置对齐的 `PortfolioData` 时，实盘单次优化入口是：

```python
optimizer = PortfolioOptimizer(
    SolverPolicy(lp_prescreen=False),
)

result = optimizer.optimize(
    data=portfolio_data,
    objective=MaximizeAlpha(),
    constraints=constraints,
    blacklist=("000001.SZ",),
    frozen=("000002.SZ",),
    not_buyable=("000003.SZ",),
)
```

黑名单、冻结名单和不可买卖名单只进入本次不可变问题，不保存在 `PortfolioOptimizer` 中。
若调用方已经构造完整问题，也可以使用更底层的：

```python
result = optimizer.solve(problem)
```

结果应先检查状态，而不是直接假定存在权重：

```python
if result.status.has_solution:
    target_weight = result.require_weights()
    print(result.backend)
    print(result.metrics.tracking_error)
    print(result.metrics.turnover_l1)
    print(result.timings.total_s)
else:
    print(result.status, result.message)
    for attempt in result.route:
        print(
            attempt.backend,
            attempt.status,
            attempt.reason,
            attempt.native_status,
            attempt.solve_s,
        )
```

`result.metrics`、`result.violations` 和 `result.certificate` 均由公共层独立复算或组织，不能只看
求解器原生状态。`result.timings` 可直接区分准备、后端 setup、后端 solve 和验收耗时。

## 3. 链式用户怎样调用

v5 范例已经把每天转换为一个 `PortfolioProblem`，因此直接调用：

```python
sequence = optimizer.solve_sequence(
    problems,
    holding_period_returns=returns_by_current_rebalance_date,
    sequence_policy=SequencePolicy(
        mode="chained",
        on_failure="stop",
        theta_seed="auto",
        holding_missing_mass_tolerance=1e-5,
        renormalize_missing_holdings=True,
        output_weights="sparse",
    ),
)
```

其中收益映射的 key 是当前调仓日，value 是“上一个调仓日至当前调仓日”的逐资产复合 C2C
收益。序列层先计算：

$$
x_{t,\mathrm{pretrade},i}
=
\frac{x_{t-1,\mathrm{target},i}(1+r_{t-1,t,i})}
{\sum_j x_{t-1,\mathrm{target},j}(1+r_{t-1,t,j})},
$$

再把它作为第 $t$ 日换手率约束的期初权重。`theta_seed="auto"` 在链式模式传播上一成功日的
theta，但不跨日复用 PIQP workspace。

如果数据已经按 `PortfolioSchedule`、`FactorRiskFrames` 和每日 benchmark frames 组织，建议
调用更高层的 `optimizer.optimize_range(...)`。它先做全区间严格同日预检，再进入同一个
`solve_sequence` 状态机。tuda2 用户则使用 `Tuda2DataSource.optimize_range(...)`，外部 I/O
仍然只发生在进入逐日求解循环之前。

结果读取示例：

```python
for step in sequence.steps:
    print(
        step.date,
        step.result.status,
        step.result.timings.total_s,
        step.result.metrics.turnover_l1,
        step.theta_seed,
    )

weights = sequence.result_for_date("2025-05-12").require_weights()
final_actual_weight = sequence.final_weight
```

## 4. 不可行诊断怎样调用

诊断不会自动运行。必须保留产生失败结果的准确问题，并显式请求：

```python
result = optimizer.solve(problem)

if not result.status.has_solution:
    report = optimizer.diagnose(
        problem,
        prior_result=result,
        level="deep",
    )
    print(report.summary_text)
    print(report.turnover_linear_lower_bound)
    print(report.minimum_tracking_error)
    for item in report.relaxations[:10]:
        print(item.group, item.key, item.side, item.amount)
```

`prior_result` 与 `problem` 的 fingerprint 必须完全一致。链式失败时，诊断问题的
`initial_weight` 必须是该日 `step.pretrade_weight`，不能拿原始占位问题诊断。范例的
`--diagnose-chain-failure` 已演示如何重建该问题。

如果只想看失败状态和 route，不应运行 deep 诊断。真实 v5 的 2025-06-16 deep 诊断额外花费
约 5.81 秒，明显高于正常求解，所以保持手动触发是正确选择。

## 5. 本机实测

口径为 5,146 左右资产、47 因子、年化 TE 2%、单股主动权重 1%、L1 换手率 5%、风格
$\pm0.6$、行业 $\pm0.05$；第一日为基准权重最大的 500 只归一化。风险单位仍按 DataYes
定义显式转换。以下都是开发机 wall-clock，不含首次 Excel 解析。

### 5.1 单期

选择 2025-05-06，热身后 fresh solve 三次：

| 指标 | 结果 |
|---|---:|
| 状态 | 3/3 optimal |
| 后端 | factor-QCQP → PIQP frontier |
| wall mean | 0.3203 s |
| wall median | 0.3234 s |
| prepare | 0.0338 s（最后一次） |
| backend solve | 0.2705 s（最后一次） |
| TE | 1.999472% |
| L1 turnover | 5.000000% |
| certificate gap | $3.088\times10^{-6}$ |
| 清理后非零权重 | 499 |

单期公开调用的 wall 与 `result.timings.total_s` 只相差约 0.6 ms，公共结果封装没有出现隐藏的
大额 Python 开销。

### 5.2 5 日短链

使用实际 C2C 收益且默认关闭 LP prescreen：

| 指标 | 结果 |
|---|---:|
| 状态 | 5/5 optimal |
| 整段 wall | 1.2121 s |
| 逐日 mean | 0.2390 s |
| 逐日 median | 0.2390 s |
| 最大持仓自然漂移 L1 | 1.4160% |

显式开启 LP prescreen 后，3/5 日由 HiGHS 证书直接返回，逐日均值为 0.1942 秒，短样本下降
约 18.8%；但同一首日仍需进入 frontier，单期耗时反而从约 0.33 秒增加到约 0.37 秒。这个
结果支持“按策略显式开启并统计通过率”，不支持修改当前默认关闭策略。

### 5.3 35 日链式正常路径

采用实际 C2C 收益、`on_failure="stop"`，在 2025-06-16 停止。停止前有 28 个成功日：

| 指标 | 成功日结果 |
|---|---:|
| total mean | 0.5129 s |
| total median | 0.4621 s |
| P95 | 0.9116 s |
| max | 1.4243 s |
| prepare mean | 0.0495 s |
| backend solve mean | 0.4442 s |
| PIQP 子问题 mean / median / max | 5.71 / 6 / 11 |
| 最大 TE | 1.999861% |
| 最大 L1 turnover | 5.000015% |
| 最大持仓自然漂移 L1 | 1.787567% |

prepare 约占成功日总耗时 9.7%，后端 solve 约占 86.6%。因此当前前端还没有吞掉此前争取到的
求解器性能；主要耗时仍在风险活跃日的参数 QP。按成功日均值机械外推 2,500 次约 21.4 分钟，
但生产估算仍应使用生产机的 P50/P95/P99，不能用这台开发机替代。

冷启动首次解析 Excel 约 20.1 秒；命中 v5 adapter cache 后约 0.06--0.10 秒。35 日问题转换约
0.67 秒，HDF 整表读取约 0.46--0.57 秒。这些属于数据适配器/I/O，不在逐日
`OptimizationResult.timings` 内。

## 6. 2025-06-16 暴露并已修复的 fallback 问题

最初审计时，该日的 route 为：

```text
PIQP frontier: MAX_ITER / numerical failure
MOSEK: PrimalInfeasible
Clarabel: NumericalError
```

deep 诊断结果为：

- 公共线性域不可行；
- 保持其余线性约束时，最小 L1 换手率为 16.1687%；
- 配置上限只有 5%，至少需要增加 11.1687 个百分点；
- Phase-I 最大冲突为 `benchmark_member_weight:members` 下界，需要放宽 5.58435%；
- 因线性域已经不可行，不再计算 minimum TE。

这符合指数成分调整日的预期。旧逻辑的公共最终状态却是 `numerical_error`，因为最后一次
Clarabel 尝试覆盖了 route 中 MOSEK 的 `infeasible` 证据，且 `result.message` 为空。这带来
两个真实问题：

1. 用户只看最终状态会误判问题性质，必须展开 `result.route` 或运行 deep 诊断；
2. 当前换手恢复只在最终 `status == INFEASIBLE` 时触发，因此该日不会自动进入已经显式授权的
   turnover recovery。

现已改为：MOSEK 成功运行并报告 `INFEASIBLE` 或 `UNBOUNDED` 时立即终止 fallback；只有
MOSEK 未安装、无 license 或自身求解失败时才进入 Clarabel。不能简单把所有 numerical
failure 都改成 infeasible，因为没有 MOSEK/诊断证据的新问题仍可能是真实数值故障。

相同 29 日链式数据复验后，2025-06-16 的公共结果为：

```text
status = infeasible
message = mosek 报告 canonical 问题不可行；原生状态：...PrimalInfeasible
route = [PIQP numerical failure, MOSEK infeasible]
```

Clarabel 不再运行，失败日总耗时从约 2.63 秒降至约 2.48 秒；更重要的是最终状态现在可以触发
用户显式授权的 `TurnoverRecoveryPolicy`。

使用同一数据并显式设置 `--turnover-recovery-max 0.20` 后，29 个交易日全部得到最优解，且仅
2025-06-16 触发一次恢复：

```text
configured turnover limit = 0.050000
minimum feasible turnover = 0.161697
effective turnover limit  = 0.161697
realized turnover         = 0.161697
tracking error            = 0.019972
final backend             = factor_qcqp_piqp
```

这说明结构性冲突被正确识别并限制在指数调整日，而不是继续污染后续日期。恢复后的正式求解约
0.55 秒；异常日还会额外执行一次精确最小换手率诊断，因此整段 wall time 会包含未计入最终
`OptimizationResult.timings` 的诊断开销。该恢复仍默认关闭，只有用户显式配置最大允许换手率时
才会启用。

将 `on_failure` 改为 `hold` 并跑完整 35 日后，6 月 16 日至 6 月 24 日连续 7 日失败，总 wall
约 34.5 秒。这再次证明默认 `stop` 是合理的；盲目承接持仓会把一次指数调整冲突扩散到后续
多期。正确恢复应依赖显式 `TurnoverRecoveryPolicy(max_turnover=...)` 和可信的不可行分类。

## 7. 当前评价

- 单期接口已经足够直接，单日交易名单的生命周期清晰。
- 链式接口正确执行 C2C 自然漂移和 theta 传播；实际漂移量也能从 `pretrade_weight` 审计。
- 计时字段可以清楚区分前端与后端，当前真实 v5 前端占比约一成，不是主要瓶颈。
- `route`、fingerprint 和 deep diagnosis 的证据结构可用，诊断默认关闭是正确性能选择。
- 多后端 fallback 现已保留 MOSEK 的确定不可行/无界状态；MOSEK 不可用或自身失败时仍由
  Clarabel 提供免费兜底。
