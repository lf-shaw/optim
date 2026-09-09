"""确定性的 close-to-close 多期组合状态机。

序列层刻意位于求解器后端之外：它推进真实组合状态、执行显式授权的失败
策略，并在每个日期调用同一个无状态单期优化器。收益率只用于将上期目标组合估值到下一个
收盘；当前契约不存在 next-open 执行模式。
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from .diagnostics import InfeasibilityReport
from .portfolio_types import (
    OptimizationResult,
    PortfolioProblem,
    ProblemFingerprint,
    SequencePolicy,
    SolveStatus,
    TurnoverLimit,
)


@dataclass(frozen=True)
class SequenceStep:
    """一个尝试调仓日的可审计输入/输出状态。

    ``pretrade_weight`` 是换手率约束实际使用的自然漂移持仓，不一定等于上期目标权重。只有
    调用方显式开启换手率恢复时，恢复相关字段才有值。

    Attributes
    ----------
    date : pandas.Timestamp
        当前尝试的调仓日期。
    result : OptimizationResult
        当前日期最终的标准化结果，可能来自原问题或授权恢复问题。
    pretrade_weight : pandas.Series | None
        用于当前换手率计算的实际交易前权重；按输出策略可能为空。
    recovered_turnover : bool
        是否通过显式放宽当前日期换手率得到可用解。
    recovery_report : InfeasibilityReport | None
        支撑换手率恢复决策的深度诊断报告。
    configured_turnover_limit : float | None
        原问题配置的换手率上限。
    minimum_feasible_turnover : float | None
        诊断得到的线性下界或完整凸问题可行边界。
    effective_turnover_limit : float | None
        恢复求解实际采用的换手率上限。
    derived_from : ProblemFingerprint | None
        原始不可行问题的 fingerprint，用于证明恢复问题的派生关系。
    """

    date: pd.Timestamp
    result: OptimizationResult
    pretrade_weight: pd.Series | None
    recovered_turnover: bool = False
    recovery_report: InfeasibilityReport | None = None
    configured_turnover_limit: float | None = None
    minimum_feasible_turnover: float | None = None
    effective_turnover_limit: float | None = None
    derived_from: ProblemFingerprint | None = None


@dataclass(frozen=True)
class PortfolioSequenceResult:
    """有序多期尝试及最终实际组合状态。

    Attributes
    ----------
    steps : tuple[SequenceStep, ...]
        按日期顺序记录的所有已尝试步骤。
    stopped_date : pandas.Timestamp | None
        因策略停止的日期；完整运行时为 ``None``。
    final_weight : pandas.Series | None
        最后一个已推进状态下的实际持仓；从未建立状态时为空。
    policy : SequencePolicy
        本次运行使用的多期策略。
    schedule_prepare_s : float
        整段 schedule 预检耗时，单位为秒。
    stopped_problem : PortfolioProblem | None
        ``on_failure='stop'`` 时导致序列停止的准确动态问题，可直接用于显式诊断。完整运行或
        ``hold`` 时为 ``None``。逐日结果不会保留问题，以免长回测持有全部风险模型。
    """

    steps: tuple[SequenceStep, ...]
    stopped_date: pd.Timestamp | None
    final_weight: pd.Series | None
    policy: SequencePolicy
    schedule_prepare_s: float = 0.0
    stopped_problem: PortfolioProblem | None = field(
        default=None, repr=False, compare=False
    )

    def result_for_date(self, date: Any) -> OptimizationResult:
        """返回指定已尝试日期的优化结果。

        Parameters
        ----------
        date : Any
            可转换为 :class:`pandas.Timestamp` 的日期。

        Returns
        -------
        OptimizationResult
            对应日期的结果。

        Raises
        ------
        KeyError
            日期不在已尝试步骤中。
        """

        target = pd.Timestamp(date)
        for step in self.steps:
            if step.date == target:
                return step.result
        raise KeyError(target)


class SequenceDataError(ValueError):
    """多期日期、收益或持仓状态无法按既定语义安全推进时抛出。"""

    pass


def solve_sequence(
    optimizer: Any,
    problems: Iterable[PortfolioProblem],
    *,
    holding_period_returns: Mapping[Any, pd.Series | np.ndarray] | None = None,
    sequence_policy: SequencePolicy | None = None,
) -> PortfolioSequenceResult:
    """按确定性规则推进实际持仓并求解有序日期。

    在首个昂贵求解前检查所有静态问题及必需的持仓收益日期。链式模式会用实际市值漂移组合
    替换每日占位期初权重；成功解成为下一期目标，``on_failure='hold'`` 承接当前交易前组合，
    默认 ``stop`` 则终止序列。

    Parameters
    ----------
    optimizer : PortfolioOptimizer
        无状态单期优化器。
    problems : Iterable[PortfolioProblem] | PreparedPortfolioRun
        严格递增日期的问题序列，或已聚合预检的惰性物化清单。
    holding_period_returns : Mapping[Any, pandas.Series | numpy.ndarray] | None
        以当前调仓日为键的上一调仓日至当日 close-to-close 复合收益。Series 按上一期持仓
        资产标签对齐；ndarray 必须按上一期持仓顺序。
    sequence_policy : SequencePolicy | None
        持仓推进、失败、恢复和输出策略。

    Returns
    -------
    PortfolioSequenceResult
        逐日步骤、停止日期及最终实际持仓。

    Raises
    ------
    SequenceDataError
        日期无序/重复、链式首日无期初持仓、收益日期缺失或持仓漂移无法安全对齐。
    PortfolioValidationError
        任一静态问题未通过预检。
    """

    policy = SequencePolicy() if sequence_policy is None else sequence_policy
    from .data.memory import PreparedPortfolioRun

    schedule_prepare_s = 0.0
    if isinstance(problems, PreparedPortfolioRun):
        problems.validation.raise_for_errors()
        dates = [pd.Timestamp(date) for date in problems.dates]
        schedule_prepare_s = problems.prepare_s

        def problem_at(position: int) -> PortfolioProblem:
            return problems.problem_at(dates[position])

        prevalidated = True
    else:
        items = tuple(problems)
        dates = [pd.Timestamp(item.data.date) for item in items]

        def problem_at(position: int) -> PortfolioProblem:
            return items[position]

        prevalidated = False
    if not dates:
        raise SequenceDataError("portfolio sequence must contain at least one problem")
    if dates != sorted(dates) or len(set(dates)) != len(dates):
        raise SequenceDataError(
            "sequence problem dates must be unique and strictly increasing"
        )

    # 在首个昂贵求解前校验所有静态日期和收益日期。后续链式期初权重虽为动态值，调用方仍
    # 提供 shape 正确的占位值，以便提前完成其余检查。
    if not prevalidated:
        for position in range(len(dates)):
            report = optimizer.validate(problem_at(position))
            report.raise_for_errors()
    first_template = problem_at(0)
    if policy.mode == "chained" and first_template.data.initial_weight is None:
        raise SequenceDataError("chained sequence requires first-day initial weights")
    normalized_returns: dict[pd.Timestamp, pd.Series | np.ndarray] = {}
    if holding_period_returns is not None:
        normalized_returns = {
            pd.Timestamp(date): value for date, value in holding_period_returns.items()
        }
    if policy.mode == "chained":
        missing_dates = [date for date in dates[1:] if date not in normalized_returns]
        if missing_dates:
            raise SequenceDataError(
                "missing close-to-close holding-period returns for "
                + ", ".join(date.strftime("%Y-%m-%d") for date in missing_dates[:10])
            )

    steps: list[SequenceStep] = []
    actual_weight: pd.Series | None = None
    stopped_date: pd.Timestamp | None = None
    stopped_problem: PortfolioProblem | None = None

    for position in range(len(dates)):
        template = first_template if position == 0 else problem_at(position)
        date = dates[position]
        if policy.mode == "chained":
            if position == 0:
                initial = np.asarray(template.data.initial_weight, dtype=float)
                pretrade = pd.Series(
                    initial,
                    index=pd.Index(template.data.assets, name="sid"),
                    name="weight",
                )
            else:
                assert actual_weight is not None
                pretrade = _mark_to_market(
                    actual_weight,
                    normalized_returns[date],
                    pd.Index(template.data.assets, name="sid"),
                    policy,
                )
            problem = replace(
                template,
                data=replace(
                    template.data, initial_weight=pretrade.to_numpy(copy=False)
                ),
            )
        else:
            problem = template
            pretrade = (
                None
                if template.data.initial_weight is None
                else pd.Series(
                    np.asarray(template.data.initial_weight, dtype=float),
                    index=pd.Index(template.data.assets, name="sid"),
                    name="weight",
                )
            )

        # 全部模板已经在进入循环前完成静态校验。链式模式此处只替换由受控漂移产生的
        # initial_weight，因此直接编译并求解，避免每个日期重复扫描风险矩阵和约束数组。
        result = optimizer._solve_prevalidated(problem)
        recovered = False
        recovery_report = None
        configured_turnover = None
        minimum_feasible_turnover = None
        effective_turnover = None
        derived_from = None
        if (
            result.status is SolveStatus.INFEASIBLE
            and policy.turnover_recovery is not None
            and problem.constraints.turnover is not None
        ):
            derived_from = result.fingerprint
            (
                result,
                recovered,
                recovery_report,
                configured_turnover,
                minimum_feasible_turnover,
                effective_turnover,
            ) = _recover_turnover(
                optimizer,
                problem,
                result,
                policy,
            )

        if result.status.has_solution:
            solved_weight = result.require_weights()
            actual_weight = solved_weight.copy()
        elif policy.mode == "chained" and policy.on_failure == "hold":
            actual_weight = pretrade.copy() if pretrade is not None else actual_weight
        else:
            stopped_date = date
            if policy.on_failure == "stop":
                stopped_problem = result.problem or problem

        # 单期结果保留准确问题以便直接诊断；序列只在顶层保留一个 stop 问题，避免 2500 日
        # 回测因每步结果反向持有风险矩阵而显著增加内存。
        stored_result = replace(_apply_output_policy(result, policy), problem=None)
        steps.append(
            SequenceStep(
                date=date,
                result=stored_result,
                pretrade_weight=_store_weight(pretrade, policy),
                recovered_turnover=recovered,
                recovery_report=recovery_report,
                configured_turnover_limit=configured_turnover,
                minimum_feasible_turnover=minimum_feasible_turnover,
                effective_turnover_limit=effective_turnover,
                derived_from=derived_from,
            )
        )
        if stopped_date is not None:
            break

    return PortfolioSequenceResult(
        steps=tuple(steps),
        stopped_date=stopped_date,
        final_weight=None if actual_weight is None else actual_weight.copy(),
        policy=policy,
        schedule_prepare_s=schedule_prepare_s,
        stopped_problem=stopped_problem,
    )


def _recover_turnover(
    optimizer: Any,
    problem: PortfolioProblem,
    original_result: OptimizationResult,
    sequence_policy: SequencePolicy,
) -> tuple[
    OptimizationResult,
    bool,
    InfeasibilityReport,
    float,
    float | None,
    float | None,
]:
    """仅在显式授权的换手率区间内尝试恢复。

    深度诊断先计算其余线性域下的最小换手率。没有 TE 约束时该值精确；存在 TE 约束时它只是
    下界，因此本函数会尝试用户授权上限，并对具有单调性的换手率上限放宽做二分。所有返回候选
    仍须通过普通独立验收；本路径从不把基准自身换手率当作组合最小换手率。
    """

    assert sequence_policy.turnover_recovery is not None
    assert problem.constraints.turnover is not None
    recovery = sequence_policy.turnover_recovery
    configured = problem.constraints.turnover.l1_limit
    report = optimizer.diagnose(problem, prior_result=original_result, level="deep")
    linear_lower = report.turnover_linear_lower_bound
    if (
        linear_lower is None
        or linear_lower > recovery.max_turnover + recovery.buffer
        or recovery.max_turnover <= configured
    ):
        return original_result, False, report, configured, linear_lower, None

    convention = problem.constraints.turnover.convention
    attempts = list(report.attempts)

    def solve_limit(limit: float) -> OptimizationResult:
        candidate = replace(
            problem,
            constraints=replace(
                problem.constraints,
                turnover=TurnoverLimit(limit, convention=convention),
            ),
        )
        candidate_result = optimizer.solve(candidate)
        attempts.extend(candidate_result.route)
        return candidate_result

    # 对 LP/QP，这是其余线性可行域的精确最小值；对 factor-QCQP，它只是下界，但通常足够
    # 接近完整凸边界，可以一次恢复求解成功。
    lower = max(configured, linear_lower)
    first_limit = min(recovery.max_turnover, lower + recovery.buffer)
    first_result = solve_limit(first_limit)
    if first_result.status.has_solution:
        convex_minimum = (
            first_limit
            if problem.constraints.tracking_error is not None
            else linear_lower
        )
        report = _with_convex_turnover(report, convex_minimum, attempts)
        return first_result, True, report, configured, convex_minimum, first_limit

    # 没有 TE 约束时，线性最小值就是精确值；在 Tmin+buffer 处完整求解失败，不能证明继续
    # 放宽换手率是正确修复，因此保留为未恢复的求解器/模型失败。
    if (
        problem.constraints.tracking_error is None
        or first_limit >= recovery.max_turnover
    ):
        return (
            first_result,
            False,
            replace(report, attempts=tuple(attempts)),
            configured,
            linear_lower,
            first_limit,
        )

    maximum_result = solve_limit(recovery.max_turnover)
    if not maximum_result.status.has_solution:
        return (
            maximum_result,
            False,
            replace(report, attempts=tuple(attempts)),
            configured,
            linear_lower,
            recovery.max_turnover,
        )

    # 仅放宽单一上限时可行性具有单调性。二分只发生在显式开启的异常路径，普通日期不增加
    # 求解调用。失败中点保守地归入下侧，最终返回的上侧点始终经过独立验收。
    low = first_limit
    high = recovery.max_turnover
    high_result = maximum_result
    search_tolerance = max(recovery.buffer, 1e-6)
    for _ in range(24):
        if high - low <= search_tolerance:
            break
        midpoint = 0.5 * (low + high)
        midpoint_result = solve_limit(midpoint)
        if midpoint_result.status.has_solution:
            high = midpoint
            high_result = midpoint_result
        else:
            low = midpoint
    report = _with_convex_turnover(report, high, attempts)
    return high_result, True, report, configured, high, high


def _with_convex_turnover(
    report: InfeasibilityReport,
    minimum: float,
    attempts: list,
) -> InfeasibilityReport:
    addition = f"完整含风险约束模型的可行换手率上界为 {minimum:.4%}。"
    return replace(
        report,
        turnover_convex_minimum=minimum,
        summary_text=report.summary_text + addition,
        attempts=tuple(attempts),
    )


def _mark_to_market(
    previous_target: pd.Series,
    holding_return: pd.Series | np.ndarray,
    current_assets: pd.Index,
    policy: SequencePolicy,
) -> pd.Series:
    """使用 C2C 收益漂移上期目标权重，并对齐当前资产域。

    缺失观测按实际持仓质量计量；除非配置容差明确允许，否则不会静默填补。新资产域删除持仓
    证券时还必须显式允许重新归一化。
    """

    if isinstance(holding_return, pd.Series):
        if holding_return.index.has_duplicates:
            raise SequenceDataError(
                "holding-period return Series contains duplicate assets"
            )
        aligned_return = holding_return.reindex(previous_target.index)
        missing = aligned_return.isna().to_numpy()
        missing_mass = float(previous_target.to_numpy()[missing].sum())
        if missing_mass > policy.holding_missing_mass_tolerance:
            raise SequenceDataError(
                f"missing close-to-close returns cover {missing_mass:.6%} of holdings"
            )
        aligned_return = aligned_return.fillna(0.0).to_numpy(float)
    else:
        aligned_return = np.asarray(holding_return, dtype=float)
        if aligned_return.shape != (len(previous_target),):
            raise SequenceDataError(
                "array holding-period returns must follow the previous holding coordinate"
            )
    if not np.all(np.isfinite(aligned_return)):
        raise SequenceDataError("holding-period returns contain NaN or infinity")
    gross = 1.0 + aligned_return
    if np.any(gross < 0.0):
        raise SequenceDataError("holding-period return below -100% is invalid")
    drifted = previous_target.to_numpy(float) * gross
    total = float(drifted.sum())
    if total <= 0.0:
        raise SequenceDataError("mark-to-market holdings have no positive value")
    drifted /= total
    drifted_series = pd.Series(drifted, index=previous_target.index, name="weight")
    aligned = drifted_series.reindex(current_assets)
    missing_mass = float(
        drifted_series.loc[~drifted_series.index.isin(current_assets)].sum()
    )
    if missing_mass > policy.holding_missing_mass_tolerance:
        raise SequenceDataError(
            f"current optimization assets omit {missing_mass:.6%} of drifted holdings"
        )
    aligned = aligned.fillna(0.0)
    retained = float(aligned.sum())
    if not np.isclose(retained, 1.0, rtol=0.0, atol=1e-12):
        if not policy.renormalize_missing_holdings:
            raise SequenceDataError(
                "current assets omit holdings; explicit renormalize_missing_holdings is disabled"
            )
        aligned /= retained
    return aligned.rename("weight")


def _apply_output_policy(
    result: OptimizationResult,
    policy: SequencePolicy,
) -> OptimizationResult:
    if result.weights is None or policy.output_weights == "all":
        return result
    if policy.output_weights == "none":
        return replace(result, weights=None)
    # 数值清理属于单期优化器：它会提升并独立复验修改后的向量。序列输出层只删除精确零，
    # 绝不能改变已经验收的解。
    sparse = result.weights[result.weights != 0.0]
    return replace(result, weights=sparse)


def _store_weight(
    weight: pd.Series | None,
    policy: SequencePolicy,
) -> pd.Series | None:
    if weight is None or policy.output_weights == "none":
        return None
    if policy.output_weights == "all":
        return weight.copy()
    return weight[weight != 0.0].copy()
