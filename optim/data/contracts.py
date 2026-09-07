"""数值求解核心之外、面向表格的批量数据契约。"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
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

    因子暴露和特异波动率使用 ``(dt, sid)`` 索引；协方差使用 ``(dt, factor)`` 行索引。
    每个日期的行因子是当日有效因子集合，列可以是覆盖整个批量区间的因子并集；物化时
    以当日行顺序选择同名列得到方阵。数值为年化小数协方差或波动率，系统不会依据
    数值大小推断单位。

    Attributes
    ----------
    exposure : pandas.DataFrame
        以唯一 ``(dt, sid)`` 行索引排列的资产乘因子暴露。
    covariance : pandas.DataFrame
        以 ``(dt, factor)`` 为行索引的年化小数因子协方差。列必须覆盖每个日期的
        行因子，可以额外包含区间内其他日期才有效的因子。
    specific_volatility : pandas.Series | pandas.DataFrame
        以 ``(dt, sid)`` 为索引的年化小数特异波动率；DataFrame 必须且只能包含一个
        数值列。
    factor_types : Mapping[str, str]
        因子名到语义类型的映射，通常为 ``"style"`` 或 ``"industry"``。
    provenance : DataProvenance
        传播至每个实例化风险模型的来源与版本元数据。
    constant_exposures : Mapping[str, float]
        不在 ``exposure`` 中物理存储的常数因子敞口。例如 country 因子对所有资产
        的敞口均为 1，可设为 ``{"country": 1.0}``；按日物化风险模型时才生成该列。
    """

    exposure: pd.DataFrame
    covariance: pd.DataFrame
    specific_volatility: pd.Series | pd.DataFrame
    factor_types: Mapping[str, str]
    provenance: DataProvenance = field(default_factory=DataProvenance)
    constant_exposures: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """冻结常数敞口映射，并在逐日物化前拒绝无效数值。"""

        normalized = {
            str(name): float(value)
            for name, value in self.constant_exposures.items()
        }
        if len(normalized) != len(self.constant_exposures):
            raise ValueError("constant exposure names are not unique after string conversion")
        invalid = [name for name, value in normalized.items() if not np.isfinite(value)]
        if invalid:
            raise ValueError(f"constant exposures are not finite: {invalid[:10]}")
        object.__setattr__(
            self,
            "constant_exposures",
            MappingProxyType(normalized),
        )

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

        if isinstance(specific, pd.DataFrame):
            if specific.shape[1] != 1:
                raise DataAlignmentError("specific volatility must have exactly one value column")
            specific = specific.iloc[:, 0]
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

        # DataYes 在 2019-12-03 调整过行业分类。批量读取跨越该日期时，columns 是
        # 全历史因子并集，而每个日期的行索引才声明当日有效因子。因此严格以行顺序
        # 选择同名列构造当日方阵；额外列不能进入当日风险计算。
        factors = pd.Index(covariance.index.astype(str), name="factor")
        covariance_columns = pd.Index(covariance.columns.astype(str), name="factor")
        if covariance.index.has_duplicates or covariance.columns.has_duplicates:
            raise DataAlignmentError(f"factor covariance contains duplicate factors on {date.date()}")
        if factors.has_duplicates or covariance_columns.has_duplicates:
            raise DataAlignmentError(
                f"factor covariance contains labels that collide after string conversion "
                f"on {date.date()}"
            )
        missing_covariance_columns = factors.difference(covariance_columns)
        if len(missing_covariance_columns):
            raise DataAlignmentError(
                f"factor covariance columns do not cover row factors on {date.date()}: "
                f"missing={missing_covariance_columns.tolist()[:10]}"
            )
        if not covariance_columns.equals(factors) or not covariance.index.equals(factors):
            covariance = covariance.copy()
            covariance.index = factors
            covariance.columns = covariance_columns
            covariance = covariance.loc[:, factors]
        exposure_columns = pd.Index(exposure.columns.astype(str))
        if exposure_columns.has_duplicates:
            raise DataAlignmentError(
                f"exposure contains duplicate factors after string conversion on {date.date()}"
            )
        constant_names = pd.Index(tuple(self.constant_exposures), dtype=object)
        overlap = exposure_columns.intersection(constant_names)
        if len(overlap):
            raise DataAlignmentError(
                f"physical and constant exposures overlap on {overlap.tolist()[:10]}"
            )
        available_factors = exposure_columns.append(constant_names)
        missing_factors = factors.difference(available_factors)
        if len(missing_factors):
            raise DataAlignmentError(
                f"exposure does not cover covariance factors on {date.date()}: "
                f"missing={missing_factors.tolist()[:10]}"
            )

        # tuda2 应保证 exposure/spec risk 共用完全相同的 (dt, sid) 顺序。
        # 两者同序时只计算一次 indexer；若调用方的 universe 本身也同序，则完全
        # 跳过资产行重排。不同序数据仍按标签分别对齐，不会将默认顺序当作正确性前提。
        if exposure.index.equals(specific.index):
            if not exposure.index.equals(assets):
                indexer = exposure.index.get_indexer(assets)
                exposure = exposure.take(indexer)
                specific = specific.take(indexer)
        else:
            exposure = exposure.reindex(assets)
            specific = specific.reindex(assets)

        if not exposure_columns.equals(pd.Index(exposure.columns)):
            exposure = exposure.copy()
            exposure.columns = exposure_columns
        physical_names = [name for name in factors if name in exposure_columns]
        if not self.constant_exposures and tuple(exposure_columns) == tuple(factors):
            exposure_values = exposure.to_numpy(float)
        else:
            exposure_values = np.empty((len(assets), len(factors)), dtype=float)
            if physical_names:
                physical_positions = factors.get_indexer(physical_names)
                exposure_values[:, physical_positions] = exposure.loc[
                    :, physical_names
                ].to_numpy(float)
            for name, value in self.constant_exposures.items():
                exposure_values[:, factors.get_loc(name)] = value
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
            exposure=exposure_values,
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
