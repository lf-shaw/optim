"""数值求解核心之外、面向表格的批量数据契约。"""

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
    """策略提供的优化日期、资产、alpha 与逐资产属性。

    Attributes
    ----------
    universe : pandas.DataFrame
        行索引必须为唯一且层名严格等于 ``(dt, sid)`` 的 MultiIndex。每个日期定义当日
        优化样本空间及可选逐资产属性。
    alpha_column : str
        alpha 类目标读取的 alpha 向量列名。
    tradable_column : str
        标识本次调仓中资产是否可交易的布尔列名。
    """

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
        """返回日程声明的、已排序且唯一的调仓日期。"""

        values = self.universe.index.get_level_values("dt").unique()
        return pd.DatetimeIndex(values).sort_values()

    def day(self, date: pd.Timestamp) -> pd.DataFrame:
        """返回以资产标识为索引的严格同日样本空间切片。

        Parameters
        ----------
        date : pandas.Timestamp
            需要读取的调仓日期。

        Returns
        -------
        pandas.DataFrame
            保持原始列的当日样本空间，索引为 ``sid``。

        Raises
        ------
        DataAlignmentError
            ``date`` 缺失时抛出；本方法不会用前一可用日期替代。
        """

        return _exact_xs(self.universe, pd.Timestamp(date), "universe")


@dataclass(frozen=True)
class FactorRiskFrames:
    """与求解器核心采用相同单位的批量因子风险模型表。

    因子暴露和特异波动率使用 ``(dt, sid)`` 索引；协方差使用 ``(dt, factor)`` 行索引
    与同名因子列。数值为年化小数协方差或波动率，系统不会依据数值大小推断单位。

    Attributes
    ----------
    exposure : pandas.DataFrame
        以唯一 ``(dt, sid)`` 行索引排列的资产乘因子暴露。
    covariance : pandas.DataFrame
        以 ``(dt, factor)`` 为行索引、以同名因子为列的年化小数因子协方差。
    specific_volatility : pandas.Series | pandas.DataFrame
        以 ``(dt, sid)`` 为索引的年化小数特异波动率；DataFrame 必须且只能包含一个
        数值列。
    factor_types : Mapping[str, str]
        因子名到语义类型的映射，通常为 ``"style"`` 或 ``"industry"``。
    provenance : DataProvenance
        传播至每个实例化风险模型的来源与版本元数据。
    """

    exposure: pd.DataFrame
    covariance: pd.DataFrame
    specific_volatility: pd.Series | pd.DataFrame
    factor_types: Mapping[str, str]
    provenance: DataProvenance = field(default_factory=DataProvenance)

    def materialize(self, date: pd.Timestamp, assets: pd.Index) -> FactorRiskModel:
        """不做日期替代，创建一个按资产位置排列的风险模型。

        Parameters
        ----------
        date : pandas.Timestamp
            需要实例化的严格风险模型日期。
        assets : pandas.Index
            因子暴露行与特异波动率必须遵循的资产顺序。

        Returns
        -------
        FactorRiskModel
            使用年化小数单位的严格同日稠密数组。

        Raises
        ------
        DataAlignmentError
            日期、资产或因子缺失，标签重复，因子坐标不一致，或缺少因子类型时抛出。
        """

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
