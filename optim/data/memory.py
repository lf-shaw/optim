"""Strict exact-date assembler for already-loaded portfolio data.

Pandas objects are used at this boundary to enforce labelled ``(dt, sid)``
alignment.  A materialized :class:`PortfolioProblem` then uses one shared asset
coordinate and dense NumPy arrays on the numerical hot path.  No date fallback
or implicit forward-fill is performed here.
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
    """Create exact-date core problems from already-loaded batch frames.

    The object retains source frames, not per-day solver models.  Benchmark
    coverage is governed by an explicit policy and risk/alpha/benchmark dates
    must match each requested optimization date exactly.
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
        """Materialize and pre-align every requested optimization date.

        Later chained initial weights are placeholders only; the sequence engine
        replaces them with naturally drifted holdings immediately before solve.
        Independent mode can supply an exact per-date mapping.
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
        """Materialize one exact-date problem once for the low-latency path.

        Validation remains the optimizer's responsibility. Unlike
        ``build_problems()``, this method intentionally does not run a schedule
        preflight and then materialize the same dense risk matrix a second time.
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
        """Materialize one date with its global sequence position.

        ``position`` distinguishes the first chained date from later dates when
        schedule preflight needs a shape-correct placeholder initial portfolio.
        The sequence engine replaces that placeholder with drifted holdings
        immediately before the actual solve.
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
        """Preflight all dates without retaining per-day dense risk arrays.

        Each date is materialized and statically validated so independent data
        errors are aggregated before any backend work begins.  Those temporary
        arrays are discarded; ``problem_at`` materializes the requested date
        again during solving.  This is deliberate validation-before-compute, not
        additional external I/O, because all source frames are already resident.
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
            # Shape-correct placeholder for schedule preflight; the chained
            # engine replaces it with naturally drifted actual holdings.
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
    """Validated lightweight schedule manifest with daily materialization.

    The manifest stores references to batch frames and immutable run settings.
    It does not store a tuple of 5,000-by-factor NumPy arrays for every date.
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
        return self.schedule.dates

    def problem_at(self, date: pd.Timestamp) -> PortfolioProblem:
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
