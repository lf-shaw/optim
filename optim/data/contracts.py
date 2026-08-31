"""Frame-oriented data contracts outside the numerical solver core."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
import pandas as pd

from ..portfolio_types import DataProvenance, FactorRiskModel
from .alignment import DataAlignmentError


def _require_dt_sid(index: pd.Index, field: str) -> pd.MultiIndex:
    if not isinstance(index, pd.MultiIndex) or index.nlevels != 2:
        raise DataAlignmentError(f"{field} must use a two-level (dt, sid) MultiIndex")
    if tuple(index.names) != ("dt", "sid"):
        raise DataAlignmentError(f"{field} index names must be exactly ('dt', 'sid')")
    if index.has_duplicates:
        raise DataAlignmentError(f"{field} contains duplicate (dt, sid) rows")
    return index


def _exact_xs(frame: pd.DataFrame | pd.Series, date: pd.Timestamp, field: str):
    _require_dt_sid(frame.index, field)
    available = pd.DatetimeIndex(frame.index.get_level_values("dt").unique())
    if date not in available:
        raise DataAlignmentError(f"{field} has no exact data for {date.date()}")
    return frame.xs(date, level="dt", drop_level=True)


@dataclass(frozen=True)
class PortfolioSchedule:
    """Strategy-supplied optimization dates, assets, alpha and attributes."""

    universe: pd.DataFrame
    alpha_column: str = "alpha"
    tradable_column: str = "tradable"

    def __post_init__(self) -> None:
        _require_dt_sid(self.universe.index, "universe")
        dates = pd.DatetimeIndex(self.universe.index.get_level_values("dt"))
        if dates.hasnans:
            raise DataAlignmentError("universe contains a missing optimization date")

    @property
    def dates(self) -> pd.DatetimeIndex:
        values = self.universe.index.get_level_values("dt").unique()
        return pd.DatetimeIndex(values).sort_values()

    def day(self, date: pd.Timestamp) -> pd.DataFrame:
        return _exact_xs(self.universe, pd.Timestamp(date), "universe")


@dataclass(frozen=True)
class FactorRiskFrames:
    """Batch factor-model frames in the same units as the solver core.

    Exposure and specific volatility use ``(dt, sid)``.  Covariance uses
    ``(dt, factor)`` rows and same-name factor columns.  Values are annualized
    decimal covariance/volatility; no magnitude-based unit inference occurs.
    """

    exposure: pd.DataFrame
    covariance: pd.DataFrame
    specific_volatility: pd.Series | pd.DataFrame
    factor_types: Mapping[str, str]
    provenance: DataProvenance = field(default_factory=DataProvenance)

    def materialize(self, date: pd.Timestamp, assets: pd.Index) -> FactorRiskModel:
        date = pd.Timestamp(date)
        exposure = _exact_xs(self.exposure, date, "risk exposure")
        specific = _exact_xs(self.specific_volatility, date, "specific volatility")
        covariance = _exact_covariance(self.covariance, date)

        if exposure.index.has_duplicates or specific.index.has_duplicates:
            raise DataAlignmentError(f"risk data contains duplicate sid on {date.date()}")
        missing_exposure = assets.difference(exposure.index)
        missing_specific = assets.difference(specific.index)
        if len(missing_exposure) or len(missing_specific):
            raise DataAlignmentError(
                f"risk model does not cover all optimization assets on {date.date()}: "
                f"exposure_missing={missing_exposure.astype(str).tolist()[:10]}, "
                f"specific_missing={missing_specific.astype(str).tolist()[:10]}"
            )

        factors = pd.Index(covariance.index.astype(str), name="factor")
        if covariance.index.has_duplicates or covariance.columns.has_duplicates:
            raise DataAlignmentError(f"factor covariance contains duplicate factors on {date.date()}")
        if set(covariance.columns.astype(str)) != set(factors):
            raise DataAlignmentError(
                f"factor covariance rows and columns differ on {date.date()}"
            )
        covariance = covariance.copy()
        covariance.index = factors
        covariance.columns = covariance.columns.astype(str)
        covariance = covariance.loc[factors, factors]
        exposure_columns = pd.Index(exposure.columns.astype(str))
        if set(exposure_columns) != set(factors):
            missing = factors.difference(exposure_columns).tolist()
            extra = exposure_columns.difference(factors).tolist()
            raise DataAlignmentError(
                f"exposure/covariance factors differ on {date.date()}: "
                f"missing={missing[:10]}, extra={extra[:10]}"
            )
        exposure = exposure.copy()
        exposure.columns = exposure_columns
        exposure = exposure.reindex(index=assets, columns=factors)

        if isinstance(specific, pd.DataFrame):
            if specific.shape[1] != 1:
                raise DataAlignmentError("specific volatility must have exactly one value column")
            specific = specific.iloc[:, 0]
        specific = specific.reindex(assets)
        factor_types = tuple(self.factor_types.get(str(name), "unknown") for name in factors)
        unknown = [str(name) for name, kind in zip(factors, factor_types) if kind == "unknown"]
        if unknown:
            raise DataAlignmentError(f"factor types are missing for {unknown[:10]}")

        source_date = self.provenance.source_date or date
        provenance = DataProvenance(
            source=self.provenance.source,
            source_date=pd.Timestamp(source_date),
            version=self.provenance.version,
            metadata=self.provenance.metadata,
        )
        return FactorRiskModel(
            asof=date,
            exposure=exposure.to_numpy(float),
            covariance=covariance.to_numpy(float),
            specific_volatility=specific.to_numpy(float),
            factor_names=tuple(str(name) for name in factors),
            factor_types=factor_types,
            provenance=provenance,
        )


def _exact_covariance(frame: pd.DataFrame, date: pd.Timestamp) -> pd.DataFrame:
    if not isinstance(frame.index, pd.MultiIndex) or frame.index.nlevels != 2:
        raise DataAlignmentError("factor covariance must use a (dt, factor) MultiIndex")
    if tuple(frame.index.names) != ("dt", "factor"):
        raise DataAlignmentError(
            "factor covariance index names must be exactly ('dt', 'factor')"
        )
    available = pd.DatetimeIndex(frame.index.get_level_values("dt").unique())
    if date not in available:
        raise DataAlignmentError(f"factor covariance has no exact data for {date.date()}")
    return frame.xs(date, level="dt", drop_level=True)
