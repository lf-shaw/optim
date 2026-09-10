# optim — Recipes

> 以下示例展示稳定公共接口，不绑定开发期 v5 文件格式。风险输入均使用年化小数单位。

基准参数统一为 `benchmark`，不再接受 `benchmark_sid`：

- `benchmark="000852.SH"`：数据源一次性获取请求日期的指数权重。
- 单期 `benchmark=pd.Series({"000001.SZ": 0.4, "600000.SH": 0.6})`：按股票标签
  对齐，视为指定 `date` 的权重，不调用指数权重接口。
- 多期 `benchmark=weights`：weights 必须是 `(dt, sid)` MultiIndex Series，提供每个
  调仓日期的权重。缺日期报错，不广播单日权重、不前向填充。

自定义权重仍执行相同的覆盖率和归一化校验。使用已绑定基准的 `PortfolioData` 或
`InMemoryDataSource` 时不再传 benchmark，避免出现两个基准来源。

---

## R1. 手工表格或已对齐数组求解单期

手工表格推荐以下入口；`universe.index` 是股票，列为 alpha/tradable，基准与期初持仓
使用股票索引 Series。risk 为按同一个 universe 股票顺序构造的风险对象；如果只有原始表，
先用 `make_factor_risk_model(date=..., assets=universe.index, exposure=...,
factor_covariance=..., specific_volatility=..., factor_types=...)` 装配。

单期 universe、benchmark Series、initial_weight 也可保留 `(dt, sid)` 索引；三者分别
校验只能含一个与 date 一致的时间戳，不允许混入其他日期。检查通过后自动脱去日期层，
与 sid 单层输入可混用；适用于手工数据工厂及 tuda2 驱动的 optimize 入口。

```python
from optim import make_portfolio_data, AlphaSpec, PortfolioOptimizer, MaximizeAlpha

data = make_portfolio_data(
    date="2026-08-31", universe=universe,
    benchmark=benchmark, initial_weight=initial_weight, risk_model=risk,
    alpha_spec=AlphaSpec(units="standardized_score"),
)
result = PortfolioOptimizer().optimize(
    data=data, objective=MaximizeAlpha(), constraints=constraints,
)
```

缺 tradable 列默认全可交易，缺 alpha 列仅用于允许无 alpha 的目标；风险单位为年化小数。
Notebook 用 `make_portfolio_data?` 查看参数；`dataclasses.fields(PortfolioData)` 查看字段，
`dataclasses.replace(data, alpha=new_alpha)` 派生数据，不原地修改已求解对象的数组。

如果输入已经严格对齐，可继续直接构造数组对象：

```python
import numpy as np
import pandas as pd

from optim import (
    AlphaSpec,
    ExposureBounds,
    FactorRiskModel,
    LowerBound,
    MaximizeAlpha,
    PortfolioConstraints,
    PortfolioData,
    PortfolioOptimizer,
    PortfolioProblem,
    SymmetricBound,
    TrackingErrorLimit,
    TurnoverLimit,
    WeightBounds,
)

assets = pd.Index(asset_ids, name="sid")
risk = FactorRiskModel(
    asof=pd.Timestamp("2026-08-31"),
    exposure=exposure,                    # (n_assets, n_factors)
    covariance=factor_covariance,         # 年化小数协方差
    specific_volatility=specific_vol,     # 年化小数波动率
    factor_names=tuple(factor_names),
    factor_types=tuple(factor_types),
)
data = PortfolioData(
    date=pd.Timestamp("2026-08-31"),
    assets=assets,
    alpha=np.asarray(alpha, dtype=float),
    alpha_spec=AlphaSpec(units="standardized_score", scale=1.0),
    benchmark=np.asarray(benchmark_weight, dtype=float),
    initial_weight=np.asarray(pretrade_weight, dtype=float),
    tradable=np.asarray(tradable, dtype=bool),
    risk_model=risk,
)
constraints = PortfolioConstraints(
    asset_weight=WeightBounds(0.0, 1.0),
    active_weight=SymmetricBound(0.01),
    total_active=1.8,
    turnover=TurnoverLimit(0.05),
    benchmark_member_weight=LowerBound(0.81),
    style=ExposureBounds(default=(-0.60, 0.60)),
    industry=ExposureBounds(default=(-0.05, 0.05)),
    tracking_error=TrackingErrorLimit(0.02),
)
problem = PortfolioProblem(data, MaximizeAlpha(), constraints)
result = PortfolioOptimizer().solve(problem)
weights = result.require_weights()
```

---

## R2. 单期实盘黑名单、冻结和单边交易约束

```python
from optim import MaximizeAlpha, PortfolioOptimizer

optimizer = PortfolioOptimizer()
result = optimizer.optimize(
    data=today_data,
    objective=MaximizeAlpha(),
    constraints=constraints,
    blacklist=["delist_candidate"],
    frozen=["suspended_sid"],
    not_buyable=["sell_only_sid"],
    not_sellable=["buy_only_sid"],
    weight_overrides={
        "special_sid": (0.001, 0.003),
    },
)
```

这些名单只属于本次请求，不会保存在 `optimizer` 中。

---

## R3. 通过 tuda2 求解单个交易日

```python
from optim import AlphaSpec, MaximizeAlpha, PortfolioOptimizer
from optim.integrations.tuda2 import Tuda2DataSource

result = PortfolioOptimizer().optimize(
    data_source=Tuda2DataSource(
        risk_model="datayes",
        benchmark_weight_type="daily",
    ),
    date=trade_date,
    universe=today_universe,      # sid 索引，含 alpha 列
    benchmark="000852.SH",
    initial_weight=pretrade_weight,
    objective=MaximizeAlpha(),
    constraints=constraints,
    alpha_spec=AlphaSpec(),
)
```

该入口只取指定日期，不读取持仓漂移收益。

---

## R4. 通过 tuda2 做链式多期优化

```python
import pandas as pd

from optim import (
    AlphaSpec,
    MaximizeAlpha,
    PortfolioOptimizer,
    PortfolioSchedule,
    SequencePolicy,
)
from optim.integrations.tuda2 import Tuda2DataSource

# universe 的索引名称必须严格为 (dt, sid)，每行含 alpha；可选 tradable。
schedule = PortfolioSchedule(
    universe=universe.sort_index(),
    alpha_column="alpha",
    tradable_column="tradable",
)

sequence = PortfolioOptimizer().optimize_range(
    data_source=Tuda2DataSource(risk_model="datayes"),
    schedule=schedule,
    benchmark="000852.SH",
    objective=MaximizeAlpha(),
    constraints=constraints,
    alpha_spec=AlphaSpec(),
    initial_weight=first_day_weight,
    sequence_policy=SequencePolicy(
        mode="chained",
        on_failure="stop",
        output_weights="sparse",
    ),
)

for step in sequence.steps:
    print(step.date, step.result.status, step.result.timings.total_s)
```

tuda2 的风险、基准和日收益按完整区间批量读取；每日 turnover 相对自然漂移后的
`step.pretrade_weight` 计算。

---

## R5. 独立冷启动多期优化

```python
from optim import SequencePolicy

sequence = optimizer.optimize_range(
    data_source=data_source,
    schedule=schedule,
    objective=objective,
    constraints=constraints,
    alpha_spec=alpha_spec,
    initial_weight=first_initial_weight,
    independent_initial_weights={
        date: weight_by_date[date]
        for date in schedule.dates
    },
    sequence_policy=SequencePolicy(
        mode="independent",
        output_weights="none",
    ),
)
```

独立模式不使用前一日优化结果，也不需要 close-to-close 持仓漂移收益。

---

## R6. 最小化跟踪误差且不提供 alpha

```python
from dataclasses import replace

from optim import MinimizeTrackingError, PortfolioOptimizer, PortfolioProblem

data_without_alpha = replace(today_data, alpha=None, alpha_spec=None)
problem = PortfolioProblem(
    data=data_without_alpha,
    objective=MinimizeTrackingError(alpha_floor=None),
    constraints=constraints_without_te_cap,
)
result = PortfolioOptimizer().solve(problem)
print(result.metrics.tracking_error)
```

---

## R7. 显式允许指数调整日恢复换手率

```python
from optim import SequencePolicy, TurnoverRecoveryPolicy

policy = SequencePolicy(
    on_failure="stop",
    turnover_recovery=TurnoverRecoveryPolicy(
        max_turnover=0.20,
        buffer=1e-5,
    ),
)

sequence = optimizer.optimize_range(
    data_source=data_source,
    schedule=schedule,
    objective=objective,
    constraints=constraints,
    alpha_spec=alpha_spec,
    initial_weight=initial_weight,
    sequence_policy=policy,
)
```

恢复只在原问题不可行、配置了 turnover 且诊断边界位于授权范围内时发生。通过
`step.recovered_turnover`、`minimum_feasible_turnover` 和 `effective_turnover_limit` 审计。

---

## R8. 诊断一个失败的准确问题

```python
result = optimizer.solve(problem)
if not result.status.has_solution:
    print(result.status.value, result.message)
    for attempt in result.route:
        print(attempt.backend, attempt.status.value, attempt.native_status)

    report = optimizer.diagnose(
        result,
        level="deep",
    )
    print(report.summary_text)
    report.dump("diagnosis.json.gz", indent=None)
    for relaxation in report.relaxations[:10]:
        print(
            relaxation.group,
            relaxation.key,
            relaxation.side,
            relaxation.amount,
        )
```

不要把 `optimize(...)` 产生的结果与后来重新组装、数据或名单已变化的问题混用；fingerprint
不同会被拒绝。

成功结果（`status.has_solution=True`）会在校验、编译前拒绝诊断，包括通过 `prior_result`
传入的成功结果。成功时读取 `result.metrics`，不要为检查已有指标运行额外 LP/QP。
上述导出包含全部 Phase-I 松弛、原生证据的组计数摘要和集中字段说明，默认不覆盖已有文件。
只有需要原始乘子时使用 `report.dump("full.json.gz", evidence="full", indent=None)`。

### 同进程对照与跨进程复现

原生贡献无需经 dump 再读取：

如果接收的是文件，先 `report = InfeasibilityReport.load("full.json.gz")`，仅支持 v2。
摘要报告的 `contributors_complete` 为 False（存在省略贡献时），可查看
`contributor_summaries`，但调用 contributors_frame 会明确报错，需发送方重新提供 full。
缺少版本、v1 和未知版本拒绝加载。加载仅恢复诊断对象，不恢复 PortfolioProblem。

```python
frame = report.contributors_frame(groups=("asset_bound", "turnover"))
one_certificate = frame.loc[frame["certificate_index"] == 0]
one_asset = one_certificate.loc[one_certificate["key"] == "000001.SZ"]
# 仅深入排查原生坐标时，使用 include_metadata=True。
```

各证书单独阅读，不能把不同后端的行拼成一个证明；表格仅便利查询，不是 IIS 或冲突排名。
缺少原生证据时返回固定列空表，仍可阅读 Phase-I 等诊断结果。

后端对照不修改业务问题：

```python
from optim import SolverPolicy

automatic = PortfolioOptimizer(SolverPolicy(backend="auto")).solve(problem)
direct = PortfolioOptimizer(SolverPolicy(backend="mosek")).solve(problem)
# 或 backend="clarabel"；无 license / 数值失败不会转入别的后端。
```

自动路线使用原设计；显式后端跳过预筛与回退，仍独立验收。
highs 只接受 LP，piqp 只接受 QP（不是通用锥求解器）；不支持的类型在准备阶段拒绝。
比较时核对 fingerprint、目标和约束残差，不能要求权重逐项相同。
复现包会保存 backend 策略；诊断辅助模型默认自动路由，不继承原问题的后端选择。
可显式 `optimizer.diagnose(result, backend="mosek")` 或 `backend="clarabel"` 进行对照，
不会回退，原结果证书来源不变。piqp 不支持必需的 Phase-I LP；highs 不能求最小风险 QP。
HiGHS/MOSEK/Clarabel 均提供统一复算的辅助 LP 数值对偶下界；查看 attempts.metadata 中
dual_bound_status、dual_bound_reason、dual_bound_method。无法处理的无穷残差方向等情况
返回 None，不把候选目标当下界。有限盒界可由原约束及已验收候选的目标子水平集推导，
仅在显式诊断时执行；数值解和松弛分配不要求跨后端逐项相同。

松弛条目可直接读取 `item.description`，例如“下界从 2.0000% 降低至 1.5000%”。
`item.relaxed_bound` 给出计算后的新边界，JSON 导出也包含该值和单位。
`amount` 始终是幅度而不是新边界；多个条目构成一个同时放松方案，不能逐项理解为必要最小改动。

不需要导出也可以调试。`with_constraints()` 支持多个字段，但定位原因时推荐单项变更：

```python
original = result.problem
assert original is not None
candidate = original.with_constraints(turnover=None)
trial = optimizer.solve(candidate)
```

原问题不变，风险数组不复制。可选约束用 `None` 移除；嵌套对象整体替换，不隐式合并。
`with_data(initial_weight=...)` 和 `with_objective(...)` 同样派生新问题。正常求解仍进行
校验并生成新 fingerprint。移除换手后可行，说明换手参与冲突，不代表它是唯一原因。
这与同时松弛多条边界的加权 Phase-I 不是同一个实验。

三类导出按用途选择，不自动增加诊断计算：

- 阅读诊断：`report.dump("diagnosis.json.gz")`，原生贡献按组汇总。
- 核查完整证据：`report.dump("evidence.json.gz", evidence="full")`，保留全部原生坐标。
- 复跑原模型：`export_repro(...)`，携带输入数组、策略和已有结果，诊断报告可选。

```python
from optim import PortfolioOptimizer, export_repro, load_repro

# 发送方；report 可省略，不会自动诊断。
export_repro("case.zip", result=result, report=report)
# 序列失败时显式传 problem=sequence_result.stopped_problem，result 为失败那一期结果。

# 接收方；加载不取数，不求解。
case = load_repro("case.zip")
print(case.version_differences)
baseline = case.solve()
local = PortfolioOptimizer(case.policy)
trial = local.solve(case.problem.with_constraints(turnover=None))
if not baseline.status.has_solution:
    new_report = local.diagnose(baseline)  # 始终显式启动
```

先比较原始 `case.original_result` 与 baseline 的状态、指标及路由，再做单项实验。
原结果、原报告是 JSON 审计快照，不是运行时对象。包保存该期真实
期初持仓，但不保存跨期 workspace 内存，不承诺重现依赖进程历史的故障。
不同 license、版本和硬件可能改变实际路由和数值解。复现包含 alpha、持仓、风险数据，
只向可信接收方传输；不含 license、环境变量和源码。默认加载上限为解压后 1 GiB，
SHA256 仅校验完整性，不认证来源。加载后数组只读，修改数值请通过 with_data 传入新数组。

---

## R9. 显式容忍极小基准覆盖缺口

```python
from optim import BenchmarkCoveragePolicy

coverage = BenchmarkCoveragePolicy(
    action="renormalize_within_tolerance",
    missing_mass_tolerance=1e-5,
)

result = optimizer.optimize(
    data_source=data_source,
    date=date,
    universe=universe,
    benchmark="000852.SH",
    initial_weight=initial_weight,
    objective=objective,
    constraints=constraints,
    alpha_spec=alpha_spec,
    benchmark_policy=coverage,
)

print(result.alignment.benchmark_missing_mass)
print(result.alignment.benchmark_renormalization_factor)
```

---

## R10. 显式开启 Factor-QCQP 的 LP 预筛

```python
from optim import PortfolioOptimizer, SolverPolicy

optimizer = PortfolioOptimizer(
    SolverPolicy(lp_prescreen=True)
)
result = optimizer.solve(problem_with_te_limit)
```

只有已经在相似历史样本上测得较高 LP 通过率时才建议开启；否则会额外增加一次 LP 成本。
