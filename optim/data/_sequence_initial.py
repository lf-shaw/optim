"""首期建仓约定：预检与实际求解共用同一问题转换。"""

from dataclasses import replace

import numpy as np

from ..portfolio_types import PortfolioProblem, SequencePolicy
from ._tradable import _as_tradable


def _first_period_problem(problem: PortfolioProblem, policy: SequencePolicy) -> PortfolioProblem:
    """链式首期可豁免换手；缺省持仓使用基准中可交易部分归一化初始化。

    不修改原始问题；基准必须匹配资金预算，不推测非单位预算的初始化方式。
    """
    if policy.mode != "chained":
        return problem
    data, constraints = problem.data, problem.constraints
    if data.initial_weight is None:
        if not policy.ignore_first_turnover:
            raise ValueError("chained sequence requires initial_weight unless ignore_first_turnover=True")
        if constraints.asset_trade is not None:
            raise ValueError("asset_trade must be None when initial_weight is omitted")
        if data.benchmark is None:
            raise ValueError("benchmark is required when initial_weight is omitted")
        benchmark = np.asarray(data.benchmark, dtype=float)
        if not np.isclose(benchmark.sum(), constraints.budget, rtol=0.0, atol=1e-8):
            raise ValueError("benchmark initialization must match budget; provide explicit initial_weight")
        tradable = (
            np.ones(len(data.assets), dtype=bool)
            if data.tradable is None else _as_tradable(data.tradable)
        )
        if benchmark.shape != (len(data.assets),) or tradable.shape != benchmark.shape:
            raise ValueError("benchmark and tradable must match assets for initialization")
        if not np.all(np.isfinite(benchmark)) or np.any(benchmark < 0):
            raise ValueError("benchmark initialization requires finite non-negative weights")
        initial = np.where(tradable, benchmark, 0.0)
        mass = float(initial.sum())
        if mass <= 0.0:
            raise ValueError("no positive tradable benchmark weight for initialization; provide explicit initial_weight")
        initial = initial * (constraints.budget / mass)
        # 仅改变初始化持仓，原基准不变；保留冻结策略使不可交易资产固定在零权重。
        data = replace(data, initial_weight=initial)
    if policy.ignore_first_turnover:
        constraints = replace(constraints, turnover=None)
    return replace(problem, data=data, constraints=constraints)
