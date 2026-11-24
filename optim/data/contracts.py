"""数值求解核心之外、面向表格的批量数据契约。"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from types import MappingProxyType
from typing import Mapping

import numpy as np
import pandas as pd

from ..portfolio_types import DataProvenance, FactorRiskModel
from .alignment import DataAlignmentError
from ._reindex import _reindex_rows
from ._date_slices import _DateSlices


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


def _single_date_values(frame: pd.DataFrame | pd.Series, date: pd.Timestamp, field: str):
    """单期输入仅允许准确一个日期；脱去日期层，不复制数值或静默裁剪多日数据。"""
    if pd.isna(date):
        raise DataAlignmentError("date must not be missing")
    if not isinstance(frame.index, pd.MultiIndex):
        return frame
    index = _require_dt_sid(frame.index, field)
    dates = index.get_level_values("dt").unique()
    if (
        not isinstance(dates, pd.DatetimeIndex)
        or dates.hasnans
        or len(dates) != 1
        or dates[0] != date
    ):
        raise DataAlignmentError(
            f"single-period {field} must contain exactly the requested date {date}"
        )
    result = frame.copy(deep=False)
    result.index = index.droplevel("dt")
    return result


@dataclass(frozen=True)
class PortfolioSchedule:
    """策略提供的优化日期、资产、alpha 与逐资产属性。

    构造后不得原地修改 universe 的数据或索引；需要变更时创建新的日程对象。

    Attributes
    ----------
    universe : pandas.DataFrame
        行索引必须为唯一且层名严格等于 ``(dt, sid)`` 的 MultiIndex。每个日期定义当日
        优化样本空间及可选逐资产属性。
    alpha_column : str
        alpha 类目标读取的 alpha 向量列名。
    tradable_column : str
        标识本次调仓中资产是否可交易的列名。接受 bool 或数值 0/1（1 可交易，0 不可交易），
        不接受缺失值、字符串或其他数值。
    """

    universe: pd.DataFrame
    alpha_column: str = "alpha"
    tradable_column: str = "tradable"

    def __post_init__(self) -> None:
        _require_dt_sid(self.universe.index, "universe")
        dates = pd.DatetimeIndex(self.universe.index.get_level_values("dt"))
        if dates.hasnans:
            raise DataAlignmentError("universe contains a missing optimization date")

    @cached_property
    def dates(self) -> pd.DatetimeIndex:
        """返回日程声明的、已排序且唯一的调仓日期。"""

        return self._date_slices._dates

    @cached_property
    def _date_slices(self):
        return _DateSlices(self.universe, "sid", "universe")

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

        return self._date_slices._day(date)


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

    @cached_property
    def _exposure_slices(self):
        return _DateSlices(self.exposure, "sid", "risk exposure")

    @cached_property
    def _specific_slices(self):
        return _DateSlices(self.specific_volatility, "sid", "specific volatility")

    @cached_property
    def _covariance_slices(self):
        return _DateSlices(self.covariance, "factor", "factor covariance")

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
        exposure = self._exposure_slices._day(date)
        specific = self._specific_slices._day(date)
        covariance = self._covariance_slices._day(date)

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

        layout = self._factor_columns(date, covariance.index)
        if exposure.index.equals(specific.index):
            if not exposure.index.equals(assets):
                indexer = exposure.index.get_indexer(assets)
                exposure = exposure.take(indexer)
                specific = specific.take(indexer)
        else:
            exposure = _reindex_rows(exposure, assets)
            specific = _reindex_rows(specific, assets)
        return self._make_risk_model(
            date, layout, layout.physical_values(exposure), layout.covariance_values(covariance),
            specific.to_numpy(float),
        )

    @cached_property
    def _factor_axes(self):
        """批量表共享的列坐标只转换一次；日期仍以协方差行因子为准。"""
        exposure = pd.Index(self.exposure.columns.astype(str))
        covariance = pd.Index(self.covariance.columns.astype(str))
        constant = pd.Index(tuple(self.constant_exposures), dtype=object)
        return exposure, covariance, constant

    def _factor_columns(self, date: pd.Timestamp, rows: pd.Index) -> _FactorColumns:
        """解析一个因子版本的列位置；保留历史行业切换时的行顺序。"""
        exposure, covariance, constant = self._factor_axes
        factors = pd.Index(rows.astype(str))
        if rows.has_duplicates or self.covariance.columns.has_duplicates:
            raise DataAlignmentError(f"factor covariance contains duplicate factors on {date.date()}")
        if factors.has_duplicates or covariance.has_duplicates:
            raise DataAlignmentError(
                f"factor covariance contains labels that collide after string conversion "
                f"on {date.date()}"
            )
        missing = factors.difference(covariance)
        if len(missing):
            raise DataAlignmentError(
                f"factor covariance columns do not cover row factors on {date.date()}: "
                f"missing={missing.tolist()[:10]}"
            )
        if exposure.has_duplicates:
            raise DataAlignmentError(
                f"exposure contains duplicate factors after string conversion on {date.date()}"
            )
        overlap = exposure.intersection(constant)
        if len(overlap):
            raise DataAlignmentError(
                f"physical and constant exposures overlap on {overlap.tolist()[:10]}"
            )
        missing = factors.difference(exposure.append(constant))
        if len(missing):
            raise DataAlignmentError(
                f"exposure does not cover covariance factors on {date.date()}: "
                f"missing={missing.tolist()[:10]}"
            )
        physical = exposure.get_indexer(factors)
        positions = np.flatnonzero(physical >= 0)
        names = tuple(factors)
        types = tuple(self.factor_types.get(name, "unknown") for name in names)
        unknown = [name for name, kind in zip(names, types) if kind == "unknown"]
        if unknown:
            raise DataAlignmentError(f"factor types are missing for {unknown[:10]}")
        physical_columns = physical[positions]
        return _FactorColumns(
            names=names,
            types=types,
            covariance_columns=None if covariance.equals(factors) else covariance.get_indexer(factors),
            physical_columns=(
                None if np.array_equal(physical_columns, np.arange(len(exposure))) else physical_columns
            ),
            all_physical_columns_used=len(physical_columns) == len(exposure),
            physical_positions=positions,
            constants=tuple(
                (factors.get_loc(name), value)
                for name, value in self.constant_exposures.items() if name in factors
            ),
        )

    def _make_risk_model(self, date, layout, physical_exposure, covariance, specific):
        """仅在需要单期模型时展开常数敞口并附加来源。"""
        exposure = physical_exposure
        if layout.constants:
            exposure = np.empty((len(specific), len(layout.names)), dtype=float)
            exposure[:, layout.physical_positions] = physical_exposure
            for position, value in layout.constants:
                exposure[:, position] = value
        provenance = DataProvenance(
            source=self.provenance.source,
            source_date=pd.Timestamp(self.provenance.source_date or date),
            version=self.provenance.version,
            metadata=self.provenance.metadata,
        )
        return FactorRiskModel(
            asof=date,
            exposure=exposure,
            covariance=covariance,
            specific_volatility=specific,
            factor_names=layout.names,
            factor_types=layout.types,
            provenance=provenance,
        )


@dataclass(frozen=True)
class _FactorColumns:
    """同一有效因子序列共享的列映射。

    `names` 与 `types` 按当日协方差的行顺序对应因子名和类型。
    `covariance_columns` 和 `physical_columns` 分别是原协方差表和原敞口表的取列位置。
    None 表示可以沿用原列顺序。`all_physical_columns_used` 指示预检能否直接扫描原敞口表。
    `physical_positions` 是物理敞口在最终矩阵中的目标列位置，`constants` 保存当日常数因子的
    目标列位置和值。
    """

    names: tuple[str, ...]
    types: tuple[str, ...]
    covariance_columns: np.ndarray | None
    physical_columns: np.ndarray | None
    all_physical_columns_used: bool
    physical_positions: np.ndarray
    constants: tuple[tuple[int, float], ...]

    def covariance_values(self, frame: pd.DataFrame) -> np.ndarray:
        if self.covariance_columns is not None:
            frame = frame.iloc[:, self.covariance_columns]
        return frame.to_numpy(float)

    def physical_values(self, frame: pd.DataFrame, *, ordered: bool = True) -> np.ndarray:
        # 预检只需扫描实际使用的列，全部有效时可以直接扫描原顺序。
        if self.physical_columns is not None and (ordered or not self.all_physical_columns_used):
            frame = frame.iloc[:, self.physical_columns]
        if ordered:
            return frame.to_numpy(float)
        values = frame.to_numpy(copy=False)
        # 保留常见浮点表的存储精度，由物化负责转成求解器使用的 float64。
        # 扩展类型与对象表仍按正常浮点转换校验，保留原有可接受的数值格式。
        return values if values.dtype.kind in "biuf" and values.dtype.itemsize <= 8 else frame.to_numpy(float)


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
