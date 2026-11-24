"""已加载组合数据的严格同日组装器。

该边界使用 pandas 对象强制执行带标签的 ``(dt, sid)`` 对齐。物化后的
:class:`PortfolioProblem` 使用统一资产顺序和稠密 NumPy 数组进入数值热路径。本模块不进行
日期替代或隐式 forward-fill。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from functools import cached_property
from typing import Mapping

import numpy as np
import pandas as pd

from .._progress import _current_progress

from ..portfolio_types import (
    AlphaSpec,
    DataProvenance,
    PortfolioConstraints,
    PortfolioData,
    PortfolioObjective,
    PortfolioProblem,
    SequencePolicy,
)
from ..validation import (
    ValidationIssue,
    ValidationReport,
    ValidationSeverity,
    _AlignedFactorInputs,
    _validate_inputs,
)
from .alignment import (
    AlignedBenchmark,
    BenchmarkCoveragePolicy,
    DataAlignmentError,
    RiskCoveragePolicy,
    _finish_benchmark,
    align_benchmark,
)
from ._reindex import _reindex_rows, _sid_isin
from ._tradable import _as_tradable
from ._sequence_initial import _first_period_inputs, _first_period_problem
from ._date_slices import _DateSlices
from .contracts import FactorRiskFrames, PortfolioSchedule, _FactorColumns, _require_dt_sid


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
        self._prepared_schedule = None
        self._benchmark_cache = {}
        self._risk_gaps = {}
        self._aligned_risk: _AlignedRiskRows | None = None
        self._prepared_columns: _ScheduleColumns | None = None

    @cached_property
    def _benchmark_slices(self):
        return _DateSlices(self.benchmark, "sid", "benchmark")

    def _risk_coverage(self, schedule, policy, *, map_rows=False):
        """以批量坐标检查风险缺口，仅为有缺口的日期组装审计。"""
        target = schedule.universe.index
        risk = self.risk_data
        exposure_dates = risk._exposure_slices._dates
        specific_dates = risk._specific_slices._dates
        _ = schedule._date_slices
        exposure_index = risk.exposure.index
        specific_index = risk.specific_volatility.index
        # 原始带行业分类的敞口表按位置裁剪；共享索引时复用映射。
        if map_rows:
            exposure_rows = exposure_index.get_indexer(target)
            specific_rows = (
                exposure_rows if exposure_index.equals(specific_index)
                else specific_index.get_indexer(target)
            )
            row_maps = exposure_rows, specific_rows
            no_exposure, no_specific = exposure_rows < 0, specific_rows < 0
        else:
            row_maps = None
            no_exposure = ~target.isin(exposure_index)
            no_specific = no_exposure if specific_index is exposure_index else ~target.isin(specific_index)
        missing = no_exposure | no_specific
        if not missing.any():
            return schedule, (), row_maps

        date_codes = target.codes[0]
        totals = np.bincount(date_codes, minlength=len(target.levels[0]))
        gaps = pd.DataFrame(
            {"exposure": no_exposure[missing], "specific": no_specific[missing]},
            index=target[missing],
        )
        issues = []
        for date, rows in gaps.groupby(level="dt", sort=True):
            count = len(rows)
            total = int(totals[target.levels[0].get_loc(date)])
            fraction = count / total
            allowed = count < total and fraction <= policy.missing_asset_fraction_tolerance
            assets = rows.index.get_level_values("sid")
            context = {
                "universe_count": total, "missing_count": count, "missing_fraction": fraction,
                "exposure_missing_count": int(rows["exposure"].sum()),
                "specific_missing_count": int(rows["specific"].sum()),
                "exposure_missing_sample": tuple(map(str, assets[rows["exposure"].to_numpy()][:10])),
                "specific_missing_sample": tuple(map(str, assets[rows["specific"].to_numpy()][:10])),
                "tolerance": policy.missing_asset_fraction_tolerance,
            }
            if allowed:
                reason = f"excluded from the optimization universe on {date.date()}"
            elif date not in exposure_dates or date not in specific_dates:
                reason = f"risk model has no exact data for {date.date()}"
            else:
                reason = f"risk coverage is insufficient on {date.date()}"
            issues.append(ValidationIssue(
                field="schedule.risk_coverage",
                code="missing_risk_assets_dropped" if allowed else "missing_risk_assets",
                severity=ValidationSeverity.WARNING if allowed else ValidationSeverity.ERROR,
                message=(f"{reason}: {count}/{total} ({fraction:.4%}), "
                         f"tolerance {policy.missing_asset_fraction_tolerance:.4%}; "
                         f"exposure_missing={context['exposure_missing_sample']}, "
                         f"specific_missing={context['specific_missing_sample']}"),
                date=date, context=context,
            ))
        errors = tuple(issue for issue in issues if issue.severity is ValidationSeverity.ERROR)
        if errors:
            return schedule, errors, None
        keep = ~missing
        if row_maps is not None:
            exposure_rows, specific_rows = row_maps
            kept_exposure = exposure_rows[keep]
            row_maps = kept_exposure, kept_exposure if exposure_rows is specific_rows else specific_rows[keep]
        return replace(schedule, universe=schedule.universe.iloc[keep]), tuple(issues), row_maps

    def _batch_for_schedule(self, schedule, *, row_maps=None, expand_risk=None):
        """按实际调仓坐标一次裁剪风险与权重；保留裁剪前的逐日覆盖审计。"""
        target = schedule.universe.index
        risk = self.risk_data
        if row_maps is None:
            # 已展开的数值表继续走通用的 carry2/pandas 对齐。
            exposure = _reindex_rows(risk.exposure, target)
            specific = _reindex_rows(risk.specific_volatility, target)
        else:
            # 同一映射已给出缺失坐标，校验通过后仅搬运保留行。
            exposure = risk.exposure.take(row_maps[0])
            specific = risk.specific_volatility.take(row_maps[1])
            exposure.index = specific.index = target
        aligned_risk = replace(risk, exposure=exposure, specific_volatility=specific)
        if expand_risk is not None:
            aligned_risk = expand_risk(aligned_risk)
        raw_benchmark = self.benchmark
        if isinstance(raw_benchmark, pd.DataFrame):
            raw_benchmark = raw_benchmark.iloc[:, 0]
        numeric = pd.to_numeric(raw_benchmark, errors="coerce").astype(float)
        result = InMemoryDataSource(
            risk_data=aligned_risk, benchmark=self.benchmark,
            benchmark_policy=self.benchmark_policy,
        )
        result._prepared_schedule = schedule
        result._benchmark_cache = self._batch_benchmark(schedule, numeric)
        result._aligned_risk = _AlignedRiskRows(aligned_risk)
        return result

    def _batch_benchmark(self, schedule, numeric):
        """一次标记所有样本外基准，按日期代码归约原始权重及缺口。"""
        _ = self._benchmark_slices
        index = numeric.index
        codes = index.codes[0]
        n_dates = len(index.levels[0])
        values = numeric.to_numpy(copy=False)
        date_codes = index.levels[0].get_indexer(schedule.dates)
        target = schedule.universe.index
        positions = target.get_indexer(index)
        inside = positions >= 0
        aligned = np.zeros(len(target), dtype=float)
        aligned[positions[inside]] = values[inside]
        outside = ~inside
        requested = np.zeros(n_dates, dtype=bool)
        requested[date_codes[date_codes >= 0]] = True
        outside &= requested[codes]
        finite = np.isfinite(values)
        totals = np.bincount(codes, weights=values, minlength=n_dates)
        invalid = np.bincount(codes[~finite], minlength=n_dates) > 0
        negative = np.bincount(codes[values < 0], minlength=n_dates) > 0
        missing = np.bincount(codes[outside], weights=values[outside], minlength=n_dates)
        maximum = np.zeros(n_dates, dtype=float)
        np.maximum.at(maximum, codes[outside], values[outside])
        labels = {}
        if outside.any():
            for date, sid in index[outside]:
                labels.setdefault(date, []).append(str(sid))
        raw_positions = self._benchmark_slices._positions
        target_positions = schedule._date_slices._positions
        policy = self.benchmark_policy
        normalized = np.isclose(totals, 1.0, rtol=0.0, atol=policy.weight_sum_tolerance)
        cache = {}
        for date, code in zip(schedule.dates, date_codes):
            try:
                if date not in raw_positions:
                    raise DataAlignmentError(f"benchmark has no exact data for {date.date()}")
                if invalid[code]:
                    raise DataAlignmentError("benchmark contains NaN or infinity")
                if negative[code]:
                    raise DataAlignmentError("benchmark contains negative weights")
                if not normalized[code]:
                    raise DataAlignmentError(
                        f"raw benchmark must sum to one before coverage handling; observed {totals[code]:.12g}"
                    )
                cache[date] = _finish_benchmark(
                    aligned[target_positions[date]], float(missing[code]),
                    tuple(labels.get(date, ())), float(maximum[code]), policy,
                )
            except DataAlignmentError as exc:
                cache[date] = exc
        return cache

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
        risk_coverage_policy: RiskCoveragePolicy | None = None,
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
        risk_coverage_policy : RiskCoveragePolicy | None
            每日样本股票风险缺口比例上限；默认严格。

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
            risk_coverage_policy=risk_coverage_policy,
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
        risk_coverage_policy: RiskCoveragePolicy | None = None,
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
        risk_coverage_policy : RiskCoveragePolicy | None
            样本股票风险缺口容差；默认严格。基准及持仓仍执行各自的覆盖规则。
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
        schedule, issues, _ = self._risk_coverage(schedule, risk_coverage_policy or RiskCoveragePolicy())
        if issues and not ValidationReport(issues).is_valid:
            raise DataAlignmentError(issues[0].message)
        if issues:
            _check_filtered_bounds(constraints)
        problem = self.materialize_problem(
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
        if not issues:
            return problem
        provenance = problem.data.provenance
        return problem.with_data(provenance=replace(
            provenance, metadata={**provenance.metadata, **_risk_gap_metadata(issues[0])},
        ))

    def materialize_problem(
        self,
        schedule: PortfolioSchedule,
        date: pd.Timestamp,
        position: int,
        *,
        objective: PortfolioObjective,
        constraints: PortfolioConstraints,
        alpha_spec: AlphaSpec | None,
        initial_weight: pd.Series | None,
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
        initial_weight: pd.Series | None = None,
        independent_initial_weights: Mapping[pd.Timestamp, pd.Series] | None = None,
        extra_attribute_columns: tuple[str, ...] = (),
        sequence_policy: SequencePolicy | None = None,
        risk_coverage_policy: RiskCoveragePolicy | None = None,
    ) -> "PreparedPortfolioRun":
        """准备全区间静态输入并预检全部日期，不保存逐日求解模型。

        静态输入按实际调仓计划准备一次，并保留逐日基准覆盖审计。预检直接检查已对齐的
        数组切片，单期问题按需构造。数据错误在首个后端工作前聚合。
        准备后的输入不得原地修改；修改输入应重新创建日程和准备清单。

        Parameters
        ----------
        schedule, objective, constraints, alpha_spec, initial_weight
            完整序列的数据和业务配置。
        independent_initial_weights : Mapping[pandas.Timestamp, pandas.Series] | None
            独立模式的逐日初始组合。
        extra_attribute_columns : tuple[str, ...]
            需要物化的额外属性列。
        sequence_policy : SequencePolicy | None
            首期建仓策略；由 optimize_range 传入。None 保持单期模板原有约束。
        risk_coverage_policy : RiskCoveragePolicy | None
            按日计算的风险缺失股票比例上限，默认严格。通过裁剪的日期记录到 validation.warnings。

        Returns
        -------
        PreparedPortfolioRun
            轻量惰性物化清单及全区间聚合校验报告。
        """

        return self._prepare_run(
            schedule, objective=objective, constraints=constraints, alpha_spec=alpha_spec,
            initial_weight=initial_weight, independent_initial_weights=independent_initial_weights,
            extra_attribute_columns=extra_attribute_columns, sequence_policy=sequence_policy,
            risk_coverage_policy=risk_coverage_policy,
        )

    def _prepare_run(
        self, schedule, *, objective, constraints, alpha_spec, initial_weight,
        independent_initial_weights, extra_attribute_columns, sequence_policy,
        risk_coverage_policy, expand_risk=None,
    ) -> "PreparedPortfolioRun":
        """共用预检流程，允许数据适配层在风险坐标裁剪后展开敞口。"""

        started = time.perf_counter()
        progress = _current_progress()
        if progress is not None:
            progress.phase("数据对齐")
        normalized_independent = None
        if independent_initial_weights is not None:
            normalized_independent = {
                pd.Timestamp(date): value
                for date, value in independent_initial_weights.items()
            }
        schedule, coverage_issues, row_maps = self._risk_coverage(
            schedule, risk_coverage_policy or RiskCoveragePolicy(), map_rows=expand_risk is not None,
        )
        coverage_report = ValidationReport(coverage_issues)
        prepared_source = self
        if coverage_report.is_valid:
            if coverage_issues:
                _check_filtered_bounds(constraints)
            prepared_source = self._batch_for_schedule(schedule, row_maps=row_maps, expand_risk=expand_risk)
            prepared_source._risk_gaps = {
                **self._risk_gaps,
                **{issue.date: _risk_gap_metadata(issue) for issue in coverage_issues},
            }
        run = PreparedPortfolioRun(
            data_source=prepared_source,
            schedule=schedule,
            objective=objective,
            constraints=constraints,
            alpha_spec=alpha_spec,
            initial_weight=initial_weight,
            independent_initial_weights=normalized_independent,
            extra_attribute_columns=tuple(extra_attribute_columns),
            validation=coverage_report,
            prepare_s=0.0,
            sequence_policy=sequence_policy,
        )
        if not coverage_report.is_valid:
            return replace(run, prepare_s=time.perf_counter() - started)
        issues = list(coverage_issues)
        prepared_source._prepared_columns = _ScheduleColumns(schedule, extra_attribute_columns)
        if progress is not None:
            progress.phase("静态预检", len(run.dates))
        for position, date in enumerate(run.dates):
            if progress is not None:
                progress.date(date)
            try:
                inputs, effective_constraints = run._inputs(date, position, ordered_risk=False)
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
            else:
                issues.extend(_validate_inputs(
                    date, inputs.assets,
                    alpha_values=inputs.alpha, alpha_spec=alpha_spec,
                    benchmark_values=inputs.benchmark.values, initial_values=inputs.initial,
                    tradable_values=inputs.tradable, extra_attributes=inputs.extra,
                    risk_model=inputs.risk, objective=objective, constraints=effective_constraints,
                ).issues)
            if progress is not None:
                progress.advance()
        return replace(
            run,
            validation=ValidationReport(tuple(issues)),
            prepare_s=time.perf_counter() - started,
        )

    def _prepared_inputs(
        self, schedule, date, position, config, initial, independent, attributes, *, ordered_risk=True,
    ):
        """用已确认的行坐标读取业务数组和当日风险，无需再次按标签对齐。"""
        rows = schedule._date_slices._positions[date]
        assets = schedule._date_slices._index(date)
        benchmark = self._benchmark_cache[date]
        if isinstance(benchmark, Exception):
            raise benchmark
        initial = _daily_initial(initial, independent, date, position, assets, benchmark.values, config.budget)
        assert self._prepared_columns is not None and self._aligned_risk is not None
        alpha, tradable, extra = self._prepared_columns.day(date, rows, attributes)
        layout, risk = self._aligned_risk.day(date, rows, ordered=ordered_risk)
        return _AlignedDay(assets, benchmark, initial, alpha, tradable, extra, layout, risk)

    def _materialize_problem(
        self,
        schedule: PortfolioSchedule,
        date: pd.Timestamp,
        position: int,
        *,
        objective: PortfolioObjective,
        constraints: PortfolioConstraints,
        alpha_spec: AlphaSpec | None,
        initial_weight: pd.Series | None,
        independent_initial_weights: Mapping[pd.Timestamp, pd.Series] | None,
        extra_attribute_columns: tuple[str, ...],
    ) -> PortfolioProblem:
        if self._prepared_schedule is schedule:
            inputs = self._prepared_inputs(
                schedule, date, position, constraints, initial_weight, independent_initial_weights,
                extra_attribute_columns,
            )
            return self._prepared_problem(date, inputs, objective, constraints, alpha_spec)

        day = schedule.day(date)
        assets = pd.Index(day.index, name="sid")
        if assets.has_duplicates:
            raise DataAlignmentError(f"universe contains duplicate sid on {date.date()}")
        if len(assets) == 0:
            raise DataAlignmentError(f"universe is empty on {date.date()}")

        benchmark_day = self._benchmark_slices._day(date)
        if isinstance(benchmark_day, pd.DataFrame):
            benchmark_day = benchmark_day.iloc[:, 0]
        aligned_benchmark = align_benchmark(benchmark_day, assets, self.benchmark_policy)
        aligned_initial = _daily_initial(
            initial_weight, independent_initial_weights, date, position, assets,
            aligned_benchmark.values, constraints.budget,
        )

        alpha = None
        if schedule.alpha_column in day:
            alpha = pd.to_numeric(day[schedule.alpha_column], errors="coerce").to_numpy(float)
        tradable = (
            _as_tradable(day[schedule.tradable_column].to_numpy())
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
        return self._make_problem(
            date, assets, aligned_benchmark, aligned_initial, alpha, tradable, extra,
            risk_model, objective, constraints, alpha_spec,
        )

    def _prepared_problem(self, date, inputs, objective, constraints, alpha_spec):
        risk = inputs.risk
        risk_model = self.risk_data._make_risk_model(
            date, inputs.layout, risk.physical_exposure, risk.covariance, risk.specific_volatility,
        )
        return self._make_problem(
            date, inputs.assets, inputs.benchmark, inputs.initial, inputs.alpha,
            inputs.tradable, inputs.extra, risk_model, objective, constraints, alpha_spec,
        )

    def _make_problem(
        self, date, assets, aligned_benchmark, aligned_initial, alpha, tradable, extra,
        risk_model, objective, constraints, alpha_spec,
    ):
        metadata = {
            "benchmark_missing_mass": aligned_benchmark.missing_mass,
            "benchmark_missing_assets": aligned_benchmark.missing_assets,
            "benchmark_maximum_missing_weight": aligned_benchmark.maximum_missing_weight,
            "benchmark_renormalization_factor": aligned_benchmark.renormalization_factor,
            "benchmark_source_date": date,
            "alpha_source_date": date if alpha is not None else None,
            **self._risk_gaps.get(date, {}),
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

    清单保存按请求准备的批量数据及不可变运行配置，不额外保存一套逐日风险矩阵副本。
    输入数组及索引不得原地修改；修改后应重新准备。

    Attributes
    ----------
    data_source : InMemoryDataSource
        已准备的风险模型、基准数据源及覆盖审计；可能不是调用方原始数据源实例。
    schedule : PortfolioSchedule
        调仓日期、资产、alpha 和属性计划。
    objective : PortfolioObjective
        各日期共享的目标。
    constraints : PortfolioConstraints
        各日期共享的静态约束。
    alpha_spec : AlphaSpec | None
        alpha 单位和尺度。
    initial_weight : pandas.Series | None
        链式首日或默认期初权重；None 仅在首期建仓策略允许时使用基准初始化。
    independent_initial_weights : Mapping[pandas.Timestamp, pandas.Series] | None
        独立模式的逐日期期初权重。
    extra_attribute_columns : tuple[str, ...]
        物化为额外属性的 schedule 列。
    validation : ValidationReport
        全日期预检的聚合报告。
    prepare_s : float
        全区间预检 wall-clock 秒数。
    sequence_policy : SequencePolicy | None
        准备时采用的首期建仓策略，实际求解时不得改变其初始化语义。
    """

    data_source: InMemoryDataSource
    schedule: PortfolioSchedule
    objective: PortfolioObjective
    constraints: PortfolioConstraints
    alpha_spec: AlphaSpec | None
    initial_weight: pd.Series | None
    independent_initial_weights: Mapping[pd.Timestamp, pd.Series] | None
    extra_attribute_columns: tuple[str, ...]
    validation: ValidationReport
    prepare_s: float
    sequence_policy: SequencePolicy | None = None

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
        if self.data_source._prepared_schedule is self.schedule:
            inputs, constraints = self._inputs(date, location)
            return self.data_source._prepared_problem(
                date, inputs, self.objective, constraints, self.alpha_spec,
            )
        problem = self.data_source.materialize_problem(
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
        if location == 0 and self.sequence_policy is not None:
            return _first_period_problem(problem, self.sequence_policy)
        return problem

    def _inputs(self, date, position, *, ordered_risk=True):
        """读取已对齐的数值切片；预检不物化单期对象及来源信息。"""
        config = self.constraints
        inputs = self.data_source._prepared_inputs(
            self.schedule, date, position, config, self.initial_weight, self.independent_initial_weights,
            self.extra_attribute_columns, ordered_risk=ordered_risk,
        )
        if position == 0 and self.sequence_policy is not None:
            initial, config = _first_period_inputs(
                len(inputs.assets), inputs.initial, inputs.benchmark.values, inputs.tradable,
                config, self.sequence_policy,
            )
            if initial is not inputs.initial:
                inputs = replace(inputs, initial=initial)
        return inputs, config


@dataclass(frozen=True)
class _AlignedDay:
    """单日期的数组切片和小型坐标描述，不保留到 PreparedPortfolioRun 中。

    `assets` 定义当天的股票顺序。`benchmark` 含基准权重和覆盖审计，`initial` 是期初权重或
    后续链式调仓的占位权重。`alpha`、`tradable`、`extra` 分别为得分、交易掩码和额外属性。
    `layout` 描述当日因子坐标，`risk` 持有对应的数值切片。
    """

    assets: pd.Index
    benchmark: AlignedBenchmark
    initial: np.ndarray | None
    alpha: np.ndarray | None
    tradable: np.ndarray
    extra: dict[str, np.ndarray]
    layout: _FactorColumns
    risk: _AlignedFactorInputs


class _ScheduleColumns:
    """复用整段常规数值列；异常或对象列在当日转换以保留报错位置。"""

    def __init__(self, schedule, attributes):
        self._universe = schedule.universe
        self._alpha = schedule.alpha_column
        self._numeric = {}
        for name in dict.fromkeys((self._alpha, *attributes)):
            if name in self._universe:
                column = self._universe[name]
                if isinstance(column, pd.Series) and isinstance(column.dtype, np.dtype) and column.dtype.kind in "biuf":
                    self._numeric[name] = column.to_numpy(float)
        self._tradable = None
        name = schedule.tradable_column
        self._flag = self._universe[name] if name in self._universe else None
        if isinstance(self._flag, pd.Series):
            try:
                self._tradable = _as_tradable(self._flag.to_numpy())
            except TypeError:
                # 回到按日校验，以便报告全部异常日期。
                pass

    def day(self, date, rows, attributes):
        def numeric(name):
            cached = self._numeric.get(name)
            if cached is not None:
                return cached[rows]
            return pd.to_numeric(self._universe[name].iloc[rows], errors="coerce").to_numpy(float)

        alpha = numeric(self._alpha) if self._alpha in self._universe else None
        if self._tradable is not None:
            tradable = self._tradable[rows]
        elif self._flag is None:
            length = rows.stop - rows.start if isinstance(rows, slice) else len(rows)
            tradable = np.ones(length, dtype=bool)
        else:
            tradable = _as_tradable(self._flag.iloc[rows].to_numpy())
        extra = {}
        for name in attributes:
            if name not in self._universe:
                raise DataAlignmentError(f"universe has no requested attribute {name!r} on {date.date()}")
            extra[name] = numeric(name)
        return alpha, tradable, extra


class _AlignedRiskRows:
    """风险行已按日程对齐，仅缓存小型协方差矩阵和共享因子坐标。"""

    def __init__(self, risk):
        self._frames = risk
        self._layouts = {}
        self._covariances = {}
        self._specific = risk.specific_volatility

    def day(self, date, rows, *, ordered):
        risk = self._frames
        cached = self._covariances.get(date)
        if cached is None:
            covariances = risk._covariance_slices
            if date not in covariances._positions:
                raise DataAlignmentError(f"factor covariance has no exact data for {date.date()}")
            index = covariances._index(date)
            key = tuple(index)
            layout = self._layouts.get(key)
            if layout is None:
                layout = risk._factor_columns(date, index)
                self._layouts[key] = layout
            covariance = layout.covariance_values(risk.covariance.iloc[covariances._positions[date]])
            self._covariances[date] = layout, covariance
        else:
            layout, covariance = cached
        return layout, _AlignedFactorInputs(
            layout.physical_values(risk.exposure.iloc[rows], ordered=ordered), covariance,
            self._specific_values(rows), layout.names, layout.types,
        )

    def _specific_values(self, rows):
        specific = self._specific
        if isinstance(specific, pd.DataFrame):
            if specific.shape[1] != 1:
                raise DataAlignmentError("specific volatility must have exactly one value column")
            specific = specific.iloc[:, 0]
        return specific.iloc[rows].to_numpy(float)


def _daily_initial(initial, independent, date, position, assets, benchmark, budget):
    """独立或首期持仓按标签对齐；后续链式模板保持原有占位语义。"""
    if independent is not None:
        if date not in independent:
            raise DataAlignmentError(f"independent initial weights have no exact data for {date.date()}")
        initial = independent[date]
    elif position > 0:
        return _check_initial_values(benchmark, budget)
    return None if initial is None else _align_initial(initial, assets, budget, date)


def _risk_gap_metadata(issue: ValidationIssue) -> dict:
    return {
        "risk_missing_asset_count": issue.context["missing_count"],
        "risk_missing_asset_fraction": issue.context["missing_fraction"],
    }


def _check_filtered_bounds(constraints: PortfolioConstraints) -> None:
    bounds = constraints.asset_weight
    if np.ndim(bounds.lower) or np.ndim(bounds.upper):
        raise DataAlignmentError(
            "risk coverage filtering requires scalar asset_weight bounds; "
            "align positional bounds to the filtered universe explicitly"
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
    if initial.index.equals(assets):
        return _check_initial_values(initial.to_numpy(float), budget)
    inside = _sid_isin(initial.index, assets)
    omitted_mass = float(np.abs(initial.loc[~inside].to_numpy(float)).sum())
    if omitted_mass > 1e-12:
        raise DataAlignmentError(
            f"optimization universe omits {omitted_mass:.6%} initial holding mass on "
            f"{date.date()}"
        )
    return _check_initial_values(_reindex_rows(initial, assets, fill_value=0.0).to_numpy(float), budget)


def _check_initial_values(values: np.ndarray, budget: float) -> np.ndarray:
    if not np.all(np.isfinite(values)):
        raise DataAlignmentError("initial weights contain NaN or infinity")
    total = float(values.sum())
    if not np.isclose(total, budget, rtol=0.0, atol=1e-8):
        raise DataAlignmentError(
            f"initial weights must sum to budget {budget:.12g}; observed {total:.12g}"
        )
    return values
