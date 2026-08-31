"""已加载组合数据的严格同日组装器。

该边界使用 pandas 对象强制执行带标签的 ``(dt, sid)`` 对齐。物化后的
:class:`PortfolioProblem` 使用统一资产顺序和稠密 NumPy 数组进入数值热路径。本模块不进行
日期替代或隐式 forward-fill。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from ..portfolio_types import (
    AlphaSpec,
    DataProvenance,
    PortfolioConstraints,
    PortfolioData,
    PortfolioObjective,
    PortfolioProblem,
)
from ..validation import (
    ValidationIssue,
    ValidationReport,
    ValidationSeverity,
    validate_problem,
)
from .alignment import BenchmarkCoveragePolicy, DataAlignmentError, align_benchmark
from .contracts import FactorRiskFrames, PortfolioSchedule, _exact_xs, _require_dt_sid


class InMemoryDataSource:
    """从已加载的批量 frame 构造严格同日核心问题。

    对象只保存源 frame，不保存逐日求解器模型。基准覆盖由显式策略控制，风险、alpha 和基准
    日期必须准确匹配每个优化日期。

    Parameters
    ----------
    risk_data : FactorRiskFrames
        已加载、使用核心年化小数单位的批量风险模型。
    benchmark : pandas.Series | pandas.DataFrame
        以严格 ``(dt, sid)`` MultiIndex 存储的逐日基准权重；DataFrame 必须只有一列。
    benchmark_policy : BenchmarkCoveragePolicy | None
        基准缺口处理策略；``None`` 使用默认严格报错策略。

    Attributes
    ----------
    risk_data : FactorRiskFrames
        批量风险模型引用。
    benchmark : pandas.Series | pandas.DataFrame
        批量基准权重引用。
    benchmark_policy : BenchmarkCoveragePolicy
        实际使用的显式覆盖策略。
    """

    def __init__(
        self,
        *,
        risk_data: FactorRiskFrames,
        benchmark: pd.Series | pd.DataFrame,
        benchmark_policy: BenchmarkCoveragePolicy | None = None,
    ) -> None:
        _require_dt_sid(benchmark.index, "benchmark")
        if isinstance(benchmark, pd.DataFrame) and benchmark.shape[1] != 1:
            raise DataAlignmentError("benchmark must contain exactly one weight column")
        self.risk_data = risk_data
        self.benchmark = benchmark
        self.benchmark_policy = (
            BenchmarkCoveragePolicy() if benchmark_policy is None else benchmark_policy
        )

    def build_problems(
        self,
        schedule: PortfolioSchedule,
        *,
        objective: PortfolioObjective,
        constraints: PortfolioConstraints,
        alpha_spec: AlphaSpec | None,
        initial_weight: pd.Series,
        independent_initial_weights: Mapping[pd.Timestamp, pd.Series] | None = None,
        extra_attribute_columns: tuple[str, ...] = (),
    ) -> tuple[PortfolioProblem, ...]:
        """物化并预对齐所有请求的优化日期。

        链式模式中首日之后的期初权重只是 shape 正确的占位符；序列引擎会在实际求解前用自然
        漂移后的真实持仓替换。独立模式可提供准确的逐日期期初权重映射。

        Parameters
        ----------
        schedule : PortfolioSchedule
            调仓日期、资产、alpha 和可选属性计划。
        objective : PortfolioObjective
            各日期共享的目标类型。
        constraints : PortfolioConstraints
            各日期共享的约束配置。
        alpha_spec : AlphaSpec | None
            alpha 的单位与尺度。
        initial_weight : pandas.Series
            首日或默认期初权重，以 sid 为索引。
        independent_initial_weights : Mapping[pandas.Timestamp, pandas.Series] | None
            独立模式下严格逐日期的期初权重。
        extra_attribute_columns : tuple[str, ...]
            需要物化为额外逐资产属性的 schedule 列。

        Returns
        -------
        tuple[PortfolioProblem, ...]
            按日期排序、已通过静态校验的单期问题。

        Raises
        ------
        PortfolioValidationError
            任一日期存在输入、单位或静态模型错误。
        DataAlignmentError
            日期或资产标签不能严格对齐。
        """

        prepared = self.prepare_run(
            schedule,
            objective=objective,
            constraints=constraints,
            alpha_spec=alpha_spec,
            initial_weight=initial_weight,
            independent_initial_weights=independent_initial_weights,
            extra_attribute_columns=extra_attribute_columns,
        )
        prepared.validation.raise_for_errors()
        return tuple(prepared.problem_at(date) for date in prepared.dates)

    def build_problem(
        self,
        schedule: PortfolioSchedule,
        *,
        objective: PortfolioObjective,
        constraints: PortfolioConstraints,
        alpha_spec: AlphaSpec | None,
        initial_weight: pd.Series,
        extra_attribute_columns: tuple[str, ...] = (),
    ) -> PortfolioProblem:
        """为低延迟单期路径只物化一次严格同日问题。

        校验仍由优化器负责。与 ``build_problems()`` 不同，本方法不会先做整段预检再重复物化
        同一份稠密风险矩阵。

        Parameters
        ----------
        schedule : PortfolioSchedule
            必须恰好包含一个日期的计划。
        objective, constraints, alpha_spec
            目标、约束和 alpha 单位配置。
        initial_weight : pandas.Series
            该日交易前实际权重，以 sid 为索引。
        extra_attribute_columns : tuple[str, ...]
            需要物化的额外属性列。

        Returns
        -------
        PortfolioProblem
            尚未重复静态校验的单日问题。

        Raises
        ------
        DataAlignmentError
            schedule 日期数不为 1，或任一输入无法严格对齐。
        """

        dates = schedule.dates
        if len(dates) != 1:
            raise DataAlignmentError(
                f"build_problem requires exactly one schedule date; observed {len(dates)}"
            )
        return self.materialize_problem(
            schedule,
            pd.Timestamp(dates[0]),
            0,
            objective=objective,
            constraints=constraints,
            alpha_spec=alpha_spec,
            initial_weight=initial_weight,
            independent_initial_weights=None,
            extra_attribute_columns=tuple(extra_attribute_columns),
        )

    def materialize_problem(
        self,
        schedule: PortfolioSchedule,
        date: pd.Timestamp,
        position: int,
        *,
        objective: PortfolioObjective,
        constraints: PortfolioConstraints,
        alpha_spec: AlphaSpec | None,
        initial_weight: pd.Series,
        independent_initial_weights: Mapping[pd.Timestamp, pd.Series] | None = None,
        extra_attribute_columns: tuple[str, ...] = (),
    ) -> PortfolioProblem:
        """按日期及其全局序列位置物化一个单期问题。

        ``position`` 用于在预检时区分链式首日和后续日期；后续日期需要 shape 正确的期初组合
        占位符，实际求解前由序列引擎替换为漂移持仓。

        Parameters
        ----------
        schedule : PortfolioSchedule
            完整调仓计划。
        date : pandas.Timestamp
            必须存在于计划及所有数据源中的准确日期。
        position : int
            该日期在完整有序序列中的从零开始位置。
        objective, constraints, alpha_spec, initial_weight
            本次物化使用的目标、约束、alpha 单位和首日期初权重。
        independent_initial_weights : Mapping[pandas.Timestamp, pandas.Series] | None
            独立模式的逐日期期初权重。
        extra_attribute_columns : tuple[str, ...]
            需要物化的额外属性列。

        Returns
        -------
        PortfolioProblem
            使用统一资产位置数组的单期问题。
        """

        return self._materialize_problem(
            schedule,
            pd.Timestamp(date),
            int(position),
            objective=objective,
            constraints=constraints,
            alpha_spec=alpha_spec,
            initial_weight=initial_weight,
            independent_initial_weights=independent_initial_weights,
            extra_attribute_columns=tuple(extra_attribute_columns),
        )

    def prepare_run(
        self,
        schedule: PortfolioSchedule,
        *,
        objective: PortfolioObjective,
        constraints: PortfolioConstraints,
        alpha_spec: AlphaSpec | None,
        initial_weight: pd.Series,
        independent_initial_weights: Mapping[pd.Timestamp, pd.Series] | None = None,
        extra_attribute_columns: tuple[str, ...] = (),
    ) -> "PreparedPortfolioRun":
        """预检全部日期，但不保留逐日稠密风险数组。

        每个日期都会临时物化并静态校验，使独立数据错误在任何后端工作前一次聚合。临时数组
        随即释放，求解时由 ``problem_at`` 再物化指定日期。由于源 frame 已驻留内存，这属于
        有意的“先校验、后计算”，而非额外外部 I/O。

        Parameters
        ----------
        schedule, objective, constraints, alpha_spec, initial_weight
            完整序列的数据和业务配置。
        independent_initial_weights : Mapping[pandas.Timestamp, pandas.Series] | None
            独立模式的逐日初始组合。
        extra_attribute_columns : tuple[str, ...]
            需要物化的额外属性列。

        Returns
        -------
        PreparedPortfolioRun
            轻量惰性物化清单及全区间聚合校验报告。
        """

        started = time.perf_counter()
        normalized_independent = None
        if independent_initial_weights is not None:
            normalized_independent = {
                pd.Timestamp(date): value
                for date, value in independent_initial_weights.items()
            }
        run = PreparedPortfolioRun(
            data_source=self,
            schedule=schedule,
            objective=objective,
            constraints=constraints,
            alpha_spec=alpha_spec,
            initial_weight=initial_weight,
            independent_initial_weights=normalized_independent,
            extra_attribute_columns=tuple(extra_attribute_columns),
            validation=ValidationReport(),
            prepare_s=0.0,
        )
        issues: list[ValidationIssue] = []
        for date in run.dates:
            try:
                problem = run.problem_at(date)
            except (DataAlignmentError, TypeError, ValueError) as exc:
                issues.append(
                    ValidationIssue(
                        field="schedule.data_alignment",
                        code="alignment_error",
                        severity=ValidationSeverity.ERROR,
                        message=str(exc),
                        date=date,
                        context={"exception_type": type(exc).__name__},
                    )
                )
                continue
            issues.extend(validate_problem(problem).issues)
        return PreparedPortfolioRun(
            data_source=self,
            schedule=schedule,
            objective=objective,
            constraints=constraints,
            alpha_spec=alpha_spec,
            initial_weight=initial_weight,
            independent_initial_weights=normalized_independent,
            extra_attribute_columns=tuple(extra_attribute_columns),
            validation=ValidationReport(tuple(issues)),
            prepare_s=time.perf_counter() - started,
        )

    def _materialize_problem(
        self,
        schedule: PortfolioSchedule,
        date: pd.Timestamp,
        position: int,
        *,
        objective: PortfolioObjective,
        constraints: PortfolioConstraints,
        alpha_spec: AlphaSpec | None,
        initial_weight: pd.Series,
        independent_initial_weights: Mapping[pd.Timestamp, pd.Series] | None,
        extra_attribute_columns: tuple[str, ...],
    ) -> PortfolioProblem:
        day = schedule.day(date)
        assets = pd.Index(day.index, name="sid")
        if assets.has_duplicates:
            raise DataAlignmentError(f"universe contains duplicate sid on {date.date()}")
        if len(assets) == 0:
            raise DataAlignmentError(f"universe is empty on {date.date()}")

        benchmark_day = _exact_xs(self.benchmark, date, "benchmark")
        if isinstance(benchmark_day, pd.DataFrame):
            benchmark_day = benchmark_day.iloc[:, 0]
        aligned_benchmark = align_benchmark(
            benchmark_day,
            assets,
            self.benchmark_policy,
        )

        selected_initial = initial_weight
        if independent_initial_weights is not None:
            if date not in independent_initial_weights:
                raise DataAlignmentError(
                    f"independent initial weights have no exact data for {date.date()}"
                )
            selected_initial = independent_initial_weights[date]
        elif position > 0:
            # 序列预检只需要 shape 正确的占位权重；链式引擎会在实际求解前替换为自然漂移持仓。
            selected_initial = pd.Series(aligned_benchmark.values, index=assets)
        aligned_initial = _align_initial(selected_initial, assets, constraints.budget, date)

        alpha = None
        if schedule.alpha_column in day:
            alpha = pd.to_numeric(day[schedule.alpha_column], errors="coerce").to_numpy(float)
        tradable = (
            day[schedule.tradable_column].to_numpy(bool)
            if schedule.tradable_column in day
            else np.ones(len(day), dtype=bool)
        )
        extra = {}
        for column in extra_attribute_columns:
            if column not in day:
                raise DataAlignmentError(
                    f"universe has no requested attribute {column!r} on {date.date()}"
                )
            extra[column] = pd.to_numeric(day[column], errors="coerce").to_numpy(float)

        risk_model = self.risk_data.materialize(date, assets)
        metadata = {
            "benchmark_missing_mass": aligned_benchmark.missing_mass,
            "benchmark_missing_assets": aligned_benchmark.missing_assets,
            "benchmark_maximum_missing_weight": aligned_benchmark.maximum_missing_weight,
            "benchmark_renormalization_factor": aligned_benchmark.renormalization_factor,
            "benchmark_source_date": date,
            "alpha_source_date": date if alpha is not None else None,
        }
        data = PortfolioData(
            date=date,
            assets=assets,
            alpha=alpha,
            alpha_spec=alpha_spec,
            benchmark=aligned_benchmark.values,
            initial_weight=aligned_initial,
            tradable=tradable,
            risk_model=risk_model,
            extra_attributes=extra,
            provenance=DataProvenance(
                source="memory",
                source_date=date,
                metadata=metadata,
            ),
        )
        return PortfolioProblem(data=data, objective=objective, constraints=constraints)


@dataclass(frozen=True)
class PreparedPortfolioRun:
    """已校验、按日惰性物化的轻量多期清单。

    清单只保存批量 frame 引用和不可变运行配置，不为每个日期保留一套约
    5,000 资产乘因子的 NumPy 数组。

    Attributes
    ----------
    data_source : InMemoryDataSource
        已加载的风险模型和基准数据源。
    schedule : PortfolioSchedule
        调仓日期、资产、alpha 和属性计划。
    objective : PortfolioObjective
        各日期共享的目标。
    constraints : PortfolioConstraints
        各日期共享的静态约束。
    alpha_spec : AlphaSpec | None
        alpha 单位和尺度。
    initial_weight : pandas.Series
        链式首日或默认期初权重。
    independent_initial_weights : Mapping[pandas.Timestamp, pandas.Series] | None
        独立模式的逐日期期初权重。
    extra_attribute_columns : tuple[str, ...]
        物化为额外属性的 schedule 列。
    validation : ValidationReport
        全日期预检的聚合报告。
    prepare_s : float
        全区间预检 wall-clock 秒数。
    """

    data_source: InMemoryDataSource
    schedule: PortfolioSchedule
    objective: PortfolioObjective
    constraints: PortfolioConstraints
    alpha_spec: AlphaSpec | None
    initial_weight: pd.Series
    independent_initial_weights: Mapping[pd.Timestamp, pd.Series] | None
    extra_attribute_columns: tuple[str, ...]
    validation: ValidationReport
    prepare_s: float

    @property
    def dates(self) -> pd.DatetimeIndex:
        """返回已准备序列的有序唯一调仓日期。"""

        return self.schedule.dates

    def problem_at(self, date: pd.Timestamp) -> PortfolioProblem:
        """按需物化一个准确日期的单期问题。

        Parameters
        ----------
        date : pandas.Timestamp
            必须存在于已准备 schedule 中的日期。

        Returns
        -------
        PortfolioProblem
            使用该日准确数据和全局序列位置构造的问题。

        Raises
        ------
        KeyError
            日期不属于已准备 schedule。
        DataAlignmentError
            该日数据无法严格按标签对齐。
        """

        date = pd.Timestamp(date)
        try:
            location = int(self.dates.get_loc(date))
        except KeyError as exc:
            raise KeyError(f"date {date.date()} is not in the prepared schedule") from exc
        return self.data_source.materialize_problem(
            self.schedule,
            date,
            location,
            objective=self.objective,
            constraints=self.constraints,
            alpha_spec=self.alpha_spec,
            initial_weight=self.initial_weight,
            independent_initial_weights=self.independent_initial_weights,
            extra_attribute_columns=self.extra_attribute_columns,
        )


def _align_initial(
    initial: pd.Series,
    assets: pd.Index,
    budget: float,
    date: pd.Timestamp,
) -> np.ndarray:
    if not isinstance(initial, pd.Series):
        raise TypeError("initial_weight must be a pandas Series indexed by sid")
    if initial.index.has_duplicates:
        raise DataAlignmentError("initial weights contain duplicate assets")
    omitted = initial.loc[~initial.index.isin(assets)]
    omitted_mass = float(np.abs(omitted.to_numpy(float)).sum())
    if omitted_mass > 1e-12:
        raise DataAlignmentError(
            f"optimization universe omits {omitted_mass:.6%} initial holding mass on "
            f"{date.date()}"
        )
    values = initial.reindex(assets, fill_value=0.0).to_numpy(float)
    if not np.all(np.isfinite(values)):
        raise DataAlignmentError("initial weights contain NaN or infinity")
    total = float(values.sum())
    if not np.isclose(total, budget, rtol=0.0, atol=1e-8):
        raise DataAlignmentError(
            f"initial weights must sum to budget {budget:.12g}; observed {total:.12g}"
        )
    return values
