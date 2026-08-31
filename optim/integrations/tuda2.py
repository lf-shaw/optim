"""Lazy, batched tuda2 adapter for the solver-independent data layer."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from importlib import import_module
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from ..data import (
    BenchmarkCoveragePolicy,
    FactorRiskFrames,
    InMemoryDataSource,
    PortfolioSchedule,
)
from ..portfolio_types import (
    AlphaSpec,
    AssetTradeConstraints,
    DataProvenance,
    PortfolioConstraints,
    PortfolioObjective,
    SequencePolicy,
)


class Tuda2UnavailableError(ImportError):
    pass


@dataclass(frozen=True)
class Tuda2LoadedData:
    risk_data: FactorRiskFrames
    benchmark: pd.DataFrame


@dataclass(frozen=True)
class Tuda2DataSource:
    risk_model: str = "datayes"
    benchmark_weight_type: str = "daily"
    module: Any | None = None

    def __post_init__(self) -> None:
        if self.risk_model not in {"cne5", "datayes"}:
            raise ValueError("risk_model must be 'cne5' or 'datayes'")
        if self.benchmark_weight_type not in {"free", "daily"}:
            raise ValueError("benchmark_weight_type must be 'free' or 'daily'")

    @cached_property
    def _factor_schema(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        module = self._module()
        style = tuple(
            str(name)
            for name in module.get_risk_model_factor_names(
                "style", model_type=self.risk_model
            )
        )
        industry = tuple(
            str(name)
            for name in module.get_risk_model_factor_names(
                "industry", model_type=self.risk_model
            )
        )
        return style, industry

    def load(
        self,
        *,
        dates: pd.DatetimeIndex | list[pd.Timestamp],
        benchmark_sid: str,
    ) -> Tuda2LoadedData:
        """Fetch every requested date in one call per data type.

        tuda2 returns annualized decimal covariance/specific volatility, so no
        numerical unit conversion is performed here.  Industry categories are
        expanded once into the same factor coordinate as covariance.
        """

        module = self._module()
        requested = pd.DatetimeIndex(pd.to_datetime(dates)).sort_values().unique()
        if len(requested) == 0:
            raise ValueError("dates must not be empty")

        exposure_raw = module.get_risk_model(
            "exposure", dts=requested, version=self.risk_model
        )
        covariance = module.get_risk_model(
            "cov", dts=requested, version=self.risk_model
        )
        specific = module.get_risk_model(
            "spec_risk", dts=requested, version=self.risk_model
        )
        style, industry = self._factor_schema
        exposure = _expand_exposure(exposure_raw, style, industry)
        benchmark = module.get_index_weight(
            benchmark_sid,
            dts=requested,
            type=self.benchmark_weight_type,
        )
        factor_types = {
            **{name: "style" for name in style},
            **{name: "industry" for name in industry},
        }
        provenance = DataProvenance(
            source="tuda2",
            version=_module_version(module),
            metadata={
                "risk_model": self.risk_model,
                "requested_dates": tuple(date.isoformat() for date in requested),
            },
        )
        return Tuda2LoadedData(
            risk_data=FactorRiskFrames(
                exposure=exposure,
                covariance=covariance,
                specific_volatility=specific,
                factor_types=factor_types,
                provenance=provenance,
            ),
            benchmark=benchmark,
        )

    def load_close_to_close_returns(
        self,
        *,
        dates: pd.DatetimeIndex | list[pd.Timestamp],
        sids: list[str] | None = None,
    ) -> dict[pd.Timestamp, pd.Series]:
        """Compound daily close-to-close returns over rebalance intervals.

        The result is keyed by the *current* rebalance date: the value for
        ``dates[i]`` compounds daily returns in ``(dates[i-1], dates[i]]``.
        Missing daily observations remain NaN so the sequence engine can
        measure missing actual holding mass rather than silently skipping them.
        """

        module = self._module()
        requested = pd.DatetimeIndex(pd.to_datetime(dates)).sort_values().unique()
        if len(requested) < 2:
            return {}
        daily_returns = module.get_return(
            since=requested[0],
            until=requested[-1],
            sids=sids,
            freq="D",
            shift=False,
            window=1,
            market_side="close",
            price_type="vwap",
            price_window=0,
        )
        if not isinstance(daily_returns, pd.DataFrame):
            raise TypeError("tuda2.get_return must return a pandas DataFrame")
        if not isinstance(daily_returns.index, pd.MultiIndex) or tuple(
            daily_returns.index.names
        ) != (
            "dt",
            "sid",
        ):
            raise ValueError("tuda2 returns must use a (dt, sid) MultiIndex")
        if daily_returns.index.has_duplicates:
            raise ValueError("tuda2 returns contain duplicate (dt, sid) rows")
        value_column = "m0" if "m0" in daily_returns.columns else None
        if value_column is None and daily_returns.shape[1] == 1:
            value_column = str(daily_returns.columns[0])
        if value_column is None:
            raise ValueError("tuda2 daily close returns do not contain one unambiguous value column")
        daily = daily_returns[value_column].unstack("sid").sort_index()
        if sids is not None:
            daily = daily.reindex(columns=pd.Index(sids, name="sid"))
        finite_values = daily.to_numpy(float)
        if np.any(finite_values[np.isfinite(finite_values)] < -1.0):
            raise ValueError("tuda2 daily return below -100% is invalid")
        result: dict[pd.Timestamp, pd.Series] = {}
        for position in range(1, len(requested)):
            previous_date = requested[position - 1]
            current_date = requested[position]
            interval = daily.loc[(daily.index > previous_date) & (daily.index <= current_date)]
            if interval.empty or current_date not in interval.index:
                raise ValueError(
                    f"tuda2 daily close returns do not reach rebalance date {current_date.date()}"
                )
            compounded = (1.0 + interval).prod(axis=0, skipna=False) - 1.0
            compounded.name = "close_to_close_return"
            result[pd.Timestamp(current_date)] = compounded
        return result

    def optimize_range(
        self,
        optimizer: Any,
        *,
        schedule: PortfolioSchedule,
        benchmark_sid: str,
        initial_weight: pd.Series,
        objective: PortfolioObjective,
        constraints: PortfolioConstraints,
        alpha_spec: AlphaSpec | None,
        sequence_policy: SequencePolicy | None = None,
        benchmark_policy: BenchmarkCoveragePolicy | None = None,
        independent_initial_weights: Mapping[pd.Timestamp, pd.Series] | None = None,
        holding_period_returns: Mapping[pd.Timestamp, pd.Series] | None = None,
        tradable_universe: str | None = None,
        extra_attribute_columns: tuple[str, ...] = (),
    ):
        """Fetch each full tuda2 range once and reuse it through all solves.

        Strategy dates/assets/alpha remain defined by the schedule. tuda2
        supplies exact-date risk, benchmark and compounded close-to-close
        returns. No tuda2 I/O occurs in the daily optimization loop.
        """

        effective_schedule = (
            schedule
            if tradable_universe is None
            else self._attach_tradability(schedule, tradable_universe)
        )
        dates = effective_schedule.dates
        loaded = self.load(dates=dates, benchmark_sid=benchmark_sid)
        memory_source = InMemoryDataSource(
            risk_data=loaded.risk_data,
            benchmark=loaded.benchmark,
            benchmark_policy=benchmark_policy,
        )
        prepared = memory_source.prepare_run(
            effective_schedule,
            objective=objective,
            constraints=constraints,
            alpha_spec=alpha_spec,
            initial_weight=initial_weight,
            independent_initial_weights=independent_initial_weights,
            extra_attribute_columns=extra_attribute_columns,
        )
        prepared.validation.raise_for_errors()

        resolved_policy = SequencePolicy() if sequence_policy is None else sequence_policy
        returns = holding_period_returns
        if resolved_policy.mode == "chained" and returns is None:
            sids = (
                effective_schedule.universe.index.get_level_values("sid")
                .unique()
                .astype(str)
                .tolist()
            )
            returns = self.load_close_to_close_returns(dates=dates, sids=sids)
        return optimizer.solve_sequence(
            prepared,
            holding_period_returns=returns,
            sequence_policy=resolved_policy,
        )

    def optimize(
        self,
        optimizer: Any,
        *,
        date: Any,
        universe: pd.DataFrame,
        benchmark_sid: str,
        initial_weight: pd.Series,
        objective: PortfolioObjective,
        constraints: PortfolioConstraints,
        alpha_spec: AlphaSpec | None,
        asset_trade: AssetTradeConstraints | None = None,
        blacklist: Iterable[Any] | None = None,
        frozen: Iterable[Any] | None = None,
        not_buyable: Iterable[Any] | None = None,
        not_sellable: Iterable[Any] | None = None,
        weight_overrides: Mapping[Any, float | tuple[float, float]] | None = None,
        benchmark_policy: BenchmarkCoveragePolicy | None = None,
        tradable_universe: str | None = None,
        extra_attribute_columns: tuple[str, ...] = (),
        theta_seed: float | None = None,
    ):
        """Fetch exact-date tuda2 inputs and solve one live-trading request."""

        target_date = pd.Timestamp(date)
        schedule = _single_date_schedule(target_date, universe)
        if tradable_universe is not None:
            schedule = self._attach_tradability(schedule, tradable_universe)
        loaded = self.load(dates=[target_date], benchmark_sid=benchmark_sid)
        memory_source = InMemoryDataSource(
            risk_data=loaded.risk_data,
            benchmark=loaded.benchmark,
            benchmark_policy=benchmark_policy,
        )
        problem = memory_source.build_problem(
            schedule,
            objective=objective,
            constraints=constraints,
            alpha_spec=alpha_spec,
            initial_weight=initial_weight,
            extra_attribute_columns=extra_attribute_columns,
        )
        return optimizer.optimize(
            data=problem.data,
            objective=objective,
            constraints=constraints,
            asset_trade=asset_trade,
            blacklist=blacklist,
            frozen=frozen,
            not_buyable=not_buyable,
            not_sellable=not_sellable,
            weight_overrides=weight_overrides,
            theta_seed=theta_seed,
        )

    def _attach_tradability(
        self,
        schedule: PortfolioSchedule,
        universe_name: str,
    ) -> PortfolioSchedule:
        frame = self._module().get_universe(
            universe=universe_name,
            dts=schedule.dates,
        )
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("tuda2.get_universe must return a pandas DataFrame")
        if not isinstance(frame.index, pd.MultiIndex) or tuple(frame.index.names) != (
            "dt",
            "sid",
        ):
            raise ValueError("tuda2 universe must use a (dt, sid) MultiIndex")
        if frame.index.has_duplicates:
            raise ValueError("tuda2 universe contains duplicate (dt, sid) rows")
        if "tradable" not in frame.columns:
            raise ValueError("tuda2 universe does not contain the tradable field")
        aligned = frame["tradable"].reindex(schedule.universe.index)
        if aligned.isna().any():
            missing = schedule.universe.index[aligned.isna()].tolist()[:10]
            raise ValueError(
                f"tuda2 tradability has no exact rows for schedule keys {missing}"
            )
        universe = schedule.universe.copy()
        universe[schedule.tradable_column] = aligned.to_numpy(bool)
        return PortfolioSchedule(
            universe,
            alpha_column=schedule.alpha_column,
            tradable_column=schedule.tradable_column,
        )

    def _module(self):
        if self.module is not None:
            return self.module
        try:
            return import_module("tuda2")
        except ImportError as exc:
            raise Tuda2UnavailableError(
                "tuda2 is optional; install the internal tuda2 package to use this adapter"
            ) from exc


def _expand_exposure(
    exposure: pd.DataFrame,
    style_factors: tuple[str, ...],
    industry_factors: tuple[str, ...],
) -> pd.DataFrame:
    if not isinstance(exposure.index, pd.MultiIndex) or tuple(exposure.index.names) != (
        "dt",
        "sid",
    ):
        raise ValueError("tuda2 exposure must use a (dt, sid) MultiIndex")
    missing_style = [name for name in style_factors if name not in exposure.columns]
    if missing_style:
        raise ValueError(f"tuda2 exposure is missing style factors {missing_style[:10]}")
    style = exposure.loc[:, list(style_factors)].astype(float, copy=False)
    if industry_factors:
        if "industry" in exposure.columns:
            labels = exposure["industry"]
            unknown = sorted(
                set(labels.dropna().astype(str).unique()) - set(industry_factors)
            )
            if unknown:
                raise ValueError(f"tuda2 exposure contains unknown industries {unknown[:10]}")
            industry = pd.get_dummies(labels, dtype=float).reindex(
                columns=list(industry_factors), fill_value=0.0
            )
        elif all(name in exposure.columns for name in industry_factors):
            industry = exposure.loc[:, list(industry_factors)].astype(float, copy=False)
        else:
            raise ValueError("tuda2 exposure has neither industry labels nor dummy factors")
        result = pd.concat([style, industry], axis=1, copy=False)
    else:
        result = style
    ordered = list(style_factors) + list(industry_factors)
    result = result.loc[:, ordered]
    values = result.to_numpy(copy=False)
    if not np.all(np.isfinite(values)):
        raise ValueError("tuda2 exposure contains NaN or infinity after expansion")
    return result


def _single_date_schedule(date: pd.Timestamp, universe: pd.DataFrame) -> PortfolioSchedule:
    if not isinstance(universe, pd.DataFrame):
        raise TypeError("universe must be a pandas DataFrame")
    frame = universe.copy()
    if isinstance(frame.index, pd.MultiIndex):
        if tuple(frame.index.names) != ("dt", "sid"):
            raise ValueError("universe MultiIndex names must be exactly ('dt', 'sid')")
        dates = pd.DatetimeIndex(frame.index.get_level_values("dt").unique())
        if len(dates) != 1 or pd.Timestamp(dates[0]) != date:
            raise ValueError("single-period universe must contain exactly the requested date")
    else:
        if frame.index.has_duplicates:
            raise ValueError("single-period universe contains duplicate sid rows")
        sids = pd.Index(frame.index, name="sid")
        frame.index = pd.MultiIndex.from_arrays(
            [np.repeat(date.to_datetime64(), len(sids)), sids],
            names=("dt", "sid"),
        )
    return PortfolioSchedule(frame)


def _module_version(module: Any) -> str | None:
    value = getattr(module, "__version__", None)
    return None if value is None else str(value)
