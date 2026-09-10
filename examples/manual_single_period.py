"""手工表格单期示例：可逐个 # %% 单元复制到 Notebook，无外部数据或 tuda2 依赖。"""

# %% 构造股票表；股票顺序仅在此声明，交易状态缺省时全部可交易。
from dataclasses import fields, replace
from time import perf_counter

import pandas as pd

from optim import (
    AlphaSpec,
    MaximizeAlpha,
    PortfolioConstraints,
    PortfolioData,
    PortfolioOptimizer,
    TrackingErrorLimit,
    TurnoverLimit,
    make_factor_risk_model,
    make_portfolio_data,
)

date = "2026-08-14"
universe = pd.DataFrame(
    {"alpha": [0.7, 0.1, -0.3], "tradable": [True, True, True]},
    index=pd.Index(["B", "A", "C"], name="sid"),
)
benchmark = pd.Series({"A": 0.4, "B": 0.6})
initial_weight = benchmark.copy()

# %% 如果已有按 universe 顺序构造的风险对象，可跳过此单元直接使用它。
# 暴露表：股票 × 因子；以下故意使用不同股票顺序，展示按标签对齐。
exposure = pd.DataFrame({"SIZE": [1.0, -1.0, 0.0]}, index=["A", "C", "B"])
factor_covariance = pd.DataFrame([[0.04]], index=["SIZE"], columns=["SIZE"])
specific_volatility = pd.Series({"C": 0.3, "B": 0.2, "A": 0.1})

started = perf_counter()
risk = make_factor_risk_model(
    date=date,
    assets=universe.index,
    exposure=exposure,
    factor_covariance=factor_covariance,  # 年化小数协方差，不再乘交易日数。
    specific_volatility=specific_volatility,  # 年化小数标准差，不是方差。
    factor_types={"SIZE": "style"},
)
data = make_portfolio_data(
    date=date,
    universe=universe,
    benchmark=benchmark,
    initial_weight=initial_weight,
    risk_model=risk,
    alpha_spec=AlphaSpec(units="standardized_score"),
)
print(f"数据装配：{perf_counter() - started:.6f} s")

# %% 单期优化；不需要手动创建 PortfolioProblem。
optimizer = PortfolioOptimizer()
result = optimizer.optimize(
    data=data,
    objective=MaximizeAlpha(),
    constraints=PortfolioConstraints(
        turnover=TurnoverLimit(0.05),
        tracking_error=TrackingErrorLimit(0.02),
    ),
)
print(result.status, result.backend)
print(result.require_weights())
print(result.timings)  # 不含前面装配数据的时间；小例子耗时不代表生产规模。

# %% 查看字段与派生修改；不使用 asdict 深复制整个风险模型。
print([field.name for field in fields(PortfolioData)])
print(data.assets)
changed_data = replace(data, alpha=data.alpha * 0.9)
# Notebook 中还可以运行 make_portfolio_data? 或在左括号后按 Shift+Tab。

# %% 仅在某次求解失败且需要调查时手动执行，不自动诊断成功结果。
# report = optimizer.diagnose(failed_result)
# print(report)
# report.dump("infeasibility.json.gz")
# variant = failed_result.problem.with_constraints(turnover=None)
# comparison = optimizer.solve(variant)
