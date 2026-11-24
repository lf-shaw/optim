"""首期建仓约定：预检与实际求解共用同一问题转换。"""

from dataclasses import replace

import numpy as np

from ..portfolio_types import PortfolioConstraints, PortfolioProblem, SequencePolicy
from ._tradable import _as_tradable


def _first_period_problem(problem: PortfolioProblem, policy: SequencePolicy) -> PortfolioProblem:
    """链式首期可豁免换手；缺省持仓使用基准中可交易部分归一化初始化。

    不修改原始问题；基准必须匹配资金预算，不推测非单位预算的初始化方式。
    """
    data = problem.data
    initial, constraints = _first_period_inputs(
        len(data.assets), data.initial_weight, data.benchmark, data.tradable, problem.constraints, policy,
    )
    if initial is data.initial_weight and constraints is problem.constraints:
        return problem
    if initial is not data.initial_weight:
        data = replace(data, initial_weight=initial)
    return replace(problem, data=data, constraints=constraints)


def _first_period_inputs(
    n_assets: int,
    initial: np.ndarray | None,
    benchmark: np.ndarray | None,
    tradable: np.ndarray | None,
    constraints: PortfolioConstraints,
    policy: SequencePolicy,
) -> tuple[np.ndarray | None, PortfolioConstraints]:
    """预检与求解共用首期持仓初始化及换手豁免规则。"""
    if policy.mode != "chained":
        return initial, constraints
    if initial is None:
        if not policy.ignore_first_turnover:
            raise ValueError("chained sequence requires initial_weight unless ignore_first_turnover=True")
        if constraints.asset_trade is not None:
            raise ValueError("asset_trade must be None when initial_weight is omitted")
        if benchmark is None:
            raise ValueError("benchmark is required when initial_weight is omitted")
        benchmark = np.asarray(benchmark, dtype=float)
        if not np.isclose(benchmark.sum(), constraints.budget, rtol=0.0, atol=1e-8):
            raise ValueError("benchmark initialization must match budget; provide explicit initial_weight")
        tradable = np.ones(n_assets, dtype=bool) if tradable is None else _as_tradable(tradable)
        if benchmark.shape != (n_assets,) or tradable.shape != benchmark.shape:
            raise ValueError("benchmark and tradable must match assets for initialization")
        if not np.all(np.isfinite(benchmark)) or np.any(benchmark < 0):
            raise ValueError("benchmark initialization requires finite non-negative weights")
        initial = np.where(tradable, benchmark, 0.0)
        mass = float(initial.sum())
        if mass <= 0.0:
            raise ValueError("no positive tradable benchmark weight for initialization; provide explicit initial_weight")
        initial *= constraints.budget / mass
    if policy.ignore_first_turnover and constraints.turnover is not None:
        constraints = replace(constraints, turnover=None)
    return initial, constraints
