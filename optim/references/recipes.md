# optim — Recipes

> 以下示例展示稳定公共接口，不绑定开发期 v5 文件格式。风险输入均使用年化小数单位。

---

## R1. 使用已经对齐的数组求解单期 Factor-QCQP

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
    benchmark_sid="000852.SH",
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
    benchmark_sid="000852.SH",
    objective=MaximizeAlpha(),
    constraints=constraints,
    alpha_spec=AlphaSpec(),
    initial_weight=first_day_weight,
    sequence_policy=SequencePolicy(
        mode="chained",
        theta_seed="auto",
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
        theta_seed="fixed",
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
        problem,
        prior_result=result,
        level="deep",
    )
    print(report.summary_text)
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
    benchmark_sid="000852.SH",
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
