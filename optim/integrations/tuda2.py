"""面向求解器无关数据层的延迟导入、批量 tuda2 适配器。"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from importlib import import_module
from typing import Any, Mapping

import numpy as np
import pandas as pd

from ..data import (
    BenchmarkCoveragePolicy,
    FactorRiskFrames,
    InMemoryDataSource,
    PortfolioSchedule,
    PreparedPortfolioRun,
)
from ..portfolio_types import (
    AlphaSpec,
    DataProvenance,
    PortfolioConstraints,
    PortfolioObjective,
    PortfolioProblem,
)


class Tuda2UnavailableError(ImportError):
    """调用 tuda2 适配器但当前环境未安装 tuda2 时抛出。"""

    pass


@dataclass(frozen=True)
class Tuda2LoadedData:
    """一次批量 tuda2 请求得到的核心数据。

    Attributes
    ----------
    risk_data : FactorRiskFrames
        已转换为统一因子轴和核心年化小数单位的风险模型 frame。
    benchmark : pandas.DataFrame
        使用 ``(dt, sid)`` MultiIndex 的逐日基准权重。
    """

    risk_data: FactorRiskFrames
    benchmark: pd.DataFrame


@dataclass(frozen=True)
class Tuda2PreparedSequence:
    """tuda2 已完成批量 I/O 和全区间预检的多期输入。

    Attributes
    ----------
    run : PreparedPortfolioRun
        仅持有批量 frame 引用、运行配置和预检报告的惰性问题清单。
    holding_period_returns : Mapping[pandas.Timestamp, pandas.Series] | None
        以当前调仓日为键的 C2C 区间复合收益；独立模式或无需持仓漂移时可以为 ``None``。
    """

    run: PreparedPortfolioRun
    holding_period_returns: Mapping[pd.Timestamp, pd.Series] | None


@dataclass(frozen=True)
class Tuda2DataSource:
    """批量获取 tuda2 风险、基准、可交易性和持仓漂移收益的数据源。

    适配器坚持“每种数据类型按完整区间一次取够”：日度优化循环只访问内存，不重复发起 I/O。

    Parameters
    ----------
    risk_model : str
        tuda2 风险模型版本，支持 ``"cne5"`` 和 ``"datayes"``。
    benchmark_weight_type : str
        指数权重口径，支持 ``"free"`` 和 ``"daily"``。
    module : Any | None
        可选的已导入 tuda2 兼容模块，主要用于测试；``None`` 时首次访问再延迟导入。

    Attributes
    ----------
    risk_model, benchmark_weight_type, module
        与同名构造参数一致，dataclass 为不可变对象。
    """

    risk_model: str = "datayes"
    benchmark_weight_type: str = "daily"
    module: Any | None = None

    def __post_init__(self) -> None:
        if self.risk_model not in {"cne5", "datayes"}:
            raise ValueError("risk_model must be 'cne5' or 'datayes'")
        if self.benchmark_weight_type not in {"free", "daily"}:
            raise ValueError("benchmark_weight_type must be 'free' or 'daily'")

    @cached_property
    def _risk_schema(self) -> Mapping[str, Any]:
        """返回 tuda2 已校验的完整风险模型 schema。

        optim 不再从若干独立接口拼接 country、风格和行业因子。完整因子顺序、类型、
        常量敞口及物理存储列均由 tuda2 的单一 schema 契约声明，避免两层实现随模型
        演进发生漂移。
        """

        module = self._module()
        try:
            schema = module.get_risk_model_schema(model=self.risk_model)
        except AttributeError as exc:
            raise Tuda2UnavailableError(
                "installed tuda2 does not provide get_risk_model_schema; "
                "upgrade tuda2 before using the risk-model adapter"
            ) from exc
        if not isinstance(schema, Mapping):
            raise TypeError("tuda2 get_risk_model_schema must return a mapping")
        return schema

    def load(
        self,
        *,
        dates: pd.DatetimeIndex | list[pd.Timestamp],
        benchmark_sid: str,
    ) -> Tuda2LoadedData:
        """按数据类型各调用一次，批量获取全部请求日期。

        tuda2 返回年化小数协方差和特异波动率，因此此处不做数值单位转换。行业分类只展开一次，
        并与协方差使用相同因子坐标。

        Parameters
        ----------
        dates : pandas.DatetimeIndex | list[pandas.Timestamp]
            需要严格获取的全部优化日期；会排序、去重，但不做日期替代。
        benchmark_sid : str
            基准指数证券标识。

        Returns
        -------
        Tuda2LoadedData
            批量风险模型和逐日基准权重。

        Raises
        ------
        ValueError
            日期为空，或 tuda2 返回的因子/行业结构无效。
        Tuda2UnavailableError
            当前环境未安装 tuda2 且没有注入兼容模块。
        """

        module = self._module()
        requested = pd.DatetimeIndex(pd.to_datetime(dates)).sort_values().unique()
        if len(requested) == 0:
            raise ValueError("dates must not be empty")

        # schema 只读取一次并被 cached_property 缓存；先验证接口契约，再进行较昂贵的
        # 批量风险数据和基准 I/O。
        schema = self._risk_schema
        style = tuple(str(name) for name in schema["style"])
        industry = tuple(str(name) for name in schema["industry"])
        factor_order = tuple(str(name) for name in schema["factor_order"])
        factor_types = {
            str(name): str(kind) for name, kind in schema["factor_types"].items()
        }
        schema_constants = {
            str(name): float(value)
            for name, value in schema["constant_exposures"].items()
        }

        # TODO 根据 dts 的稠密程度（跟完整交易日比）决定是读取时直接过滤（传入 dts）还是读取完了之后再过滤
        # 有时候我们是每日优化，日期是完备的，再传入 dts 会减低数据读取效率。

        exposure_raw = module.get_risk_model(
            "exposure", dts=requested, model=self.risk_model
        )
        covariance = module.get_risk_model("cov", dts=requested, model=self.risk_model)
        specific = module.get_risk_model(
            "spec_risk", dts=requested, model=self.risk_model
        )
        _validate_covariance_schema(covariance, factor_order)
        exposure, virtual_constants = _expand_exposure(
            exposure_raw,
            style,
            industry,
            factor_order,
            schema_constants,
        )
        benchmark = module.get_index_weight(
            benchmark_sid,
            dts=requested,
            type=self.benchmark_weight_type,
        )
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
                constant_exposures=virtual_constants,
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
        """在相邻调仓区间复合日度 close-to-close 收益。

        返回映射使用“当前调仓日”为键：``dates[i]`` 的值复合区间
        ``(dates[i-1], dates[i]]`` 内的日度收益。日度观测缺失时保留 NaN，使序列引擎能够
        计算缺失实际持仓质量，而不是静默跳过。

        Parameters
        ----------
        dates : pandas.DatetimeIndex | list[pandas.Timestamp]
            有序调仓日期；少于两个日期时返回空映射。
        sids : list[str] | None
            需要获取的资产集合；``None`` 使用 tuda2 默认资产域。

        Returns
        -------
        dict[pandas.Timestamp, pandas.Series]
            当前调仓日到逐资产区间复合收益的映射。

        Raises
        ------
        TypeError
            ``tuda2.get_return`` 未返回 DataFrame。
        ValueError
            返回标签、值列、日期覆盖或收益数值无效。
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
            raise ValueError(
                "tuda2 daily close returns do not contain one unambiguous value column"
            )
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
            interval = daily.loc[
                (daily.index > previous_date) & (daily.index <= current_date)
            ]
            if interval.empty or current_date not in interval.index:
                raise ValueError(
                    f"tuda2 daily close returns do not reach rebalance date {current_date.date()}"
                )
            compounded = (1.0 + interval).prod(axis=0, skipna=False) - 1.0
            compounded.name = "close_to_close_return"
            result[pd.Timestamp(current_date)] = compounded
        return result

    def prepare_sequence(
        self,
        *,
        schedule: PortfolioSchedule,
        benchmark_sid: str,
        initial_weight: pd.Series,
        objective: PortfolioObjective,
        constraints: PortfolioConstraints,
        alpha_spec: AlphaSpec | None,
        benchmark_policy: BenchmarkCoveragePolicy | None = None,
        independent_initial_weights: Mapping[pd.Timestamp, pd.Series] | None = None,
        holding_period_returns: Mapping[pd.Timestamp, pd.Series] | None = None,
        require_holding_returns: bool = True,
        tradable_universe: str | None = None,
        extra_attribute_columns: tuple[str, ...] = (),
    ) -> Tuda2PreparedSequence:
        """一次获取完整区间，预检并返回求解器无关的多期输入。

        策略日期、资产和 alpha 仍由 schedule 定义；tuda2 提供严格同日风险、基准及复合
        close-to-close 收益。该方法不调用 optimizer；普通用户通过
        ``PortfolioOptimizer.optimize_range(data_source=self, ...)`` 间接使用它。

        Parameters
        ----------
        schedule : PortfolioSchedule
            策略定义的调仓日期、资产、alpha 和属性。
        benchmark_sid : str
            基准指数证券标识。
        initial_weight : pandas.Series
            链式首日实际持仓，以 sid 为索引。
        objective : PortfolioObjective
            各日期共享的目标。
        constraints : PortfolioConstraints
            各日期共享的静态约束。
        alpha_spec : AlphaSpec | None
            alpha 单位和尺度。
        benchmark_policy : BenchmarkCoveragePolicy | None
            基准覆盖缺口的显式处理策略。
        independent_initial_weights : Mapping[pandas.Timestamp, pandas.Series] | None
            独立模式的逐日期期初权重。
        holding_period_returns : Mapping[pandas.Timestamp, pandas.Series] | None
            调用方已提供的区间复合收益；不为空时不会再次读取收益。
        require_holding_returns : bool
            是否为链式持仓漂移准备收益。为真且调用方未提供时，一次性从 tuda2 获取。
        tradable_universe : str | None
            可选 tuda2 universe 名称，用于批量附加严格同日可交易性。
        extra_attribute_columns : tuple[str, ...]
            从 schedule 物化的额外属性列。

        Returns
        -------
        Tuda2PreparedSequence
            已预检的惰性问题清单和可选持有期收益。

        Raises
        ------
        DataAlignmentError
            schedule 与 tuda2 严格同日数据无法安全对齐。
        PortfolioValidationError
            预检发现任一日期的输入或静态模型错误。
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

        returns = holding_period_returns
        if require_holding_returns and returns is None:
            sids = (
                effective_schedule.universe.index.get_level_values("sid")
                .unique()
                .astype(str)
                .tolist()
            )
            returns = self.load_close_to_close_returns(dates=dates, sids=sids)
        return Tuda2PreparedSequence(
            run=prepared,
            holding_period_returns=returns,
        )

    def build_problem(
        self,
        *,
        date: Any,
        universe: pd.DataFrame,
        benchmark_sid: str,
        initial_weight: pd.Series,
        objective: PortfolioObjective,
        constraints: PortfolioConstraints,
        alpha_spec: AlphaSpec | None,
        benchmark_policy: BenchmarkCoveragePolicy | None = None,
        tradable_universe: str | None = None,
        extra_attribute_columns: tuple[str, ...] = (),
    ) -> PortfolioProblem:
        """获取严格同日 tuda2 输入并只构造一个单期问题。

        本方法不调用求解器。普通用户通过
        ``PortfolioOptimizer.optimize(data_source=self, ...)`` 间接使用它。

        Parameters
        ----------
        date : Any
            唯一优化日期，可转换为 :class:`pandas.Timestamp`。
        universe : pandas.DataFrame
            单日资产及 alpha/属性；可使用 sid 索引，或只含请求日期的 ``(dt, sid)`` 索引。
        benchmark_sid : str
            基准指数证券标识。
        initial_weight : pandas.Series
            交易前实际持仓，以 sid 为索引。
        objective, constraints, alpha_spec
            本次请求的目标、约束及 alpha 单位。
        benchmark_policy : BenchmarkCoveragePolicy | None
            基准覆盖缺口的显式策略。
        tradable_universe : str | None
            可选 tuda2 universe 名称，用于附加该日可交易性。
        extra_attribute_columns : tuple[str, ...]
            从 universe 物化的额外属性列。
        Returns
        -------
        PortfolioProblem
            已严格对齐但尚未求解的单期问题。

        Raises
        ------
        DataAlignmentError
            任一输入缺少请求日期或资产标签无法严格对齐。
        PortfolioValidationError
            组装后问题未通过静态校验。
        """

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
        return memory_source.build_problem(
            schedule,
            objective=objective,
            constraints=constraints,
            alpha_spec=alpha_spec,
            initial_weight=initial_weight,
            extra_attribute_columns=extra_attribute_columns,
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
                "tuda2>=2.0.40 is required by this optional integration; "
                "install optim[tuda2]"
            ) from exc


def _expand_exposure(
    exposure: pd.DataFrame,
    style_factors: tuple[str, ...],
    industry_factors: tuple[str, ...],
    factor_order: tuple[str, ...],
    constant_exposures: Mapping[str, float],
) -> tuple[pd.DataFrame, Mapping[str, float]]:
    """将 tuda2 风格列和行业分类转换为与风险协方差兼容的敞口。

    schema 声明的常数因子（当前为 ``country=1``）不必在整个时间区间物理存储。
    最终逐日数组仍严格按协方差的当日行因子顺序物化，因此不会丢失常数因子的方差
    或与其他因子的协方差项。
    """

    if not isinstance(exposure.index, pd.MultiIndex) or tuple(exposure.index.names) != (
        "dt",
        "sid",
    ):
        raise ValueError("tuda2 exposure must use a (dt, sid) MultiIndex")
    constants = {str(name): float(value) for name, value in constant_exposures.items()}
    constant_names = set(constants)
    physical_style = tuple(name for name in style_factors if name not in constant_names)
    declared = set(physical_style) | set(industry_factors) | constant_names
    unknown_schema = [name for name in factor_order if name not in declared]
    if unknown_schema:
        raise ValueError(
            "tuda2 factor_order contains factors absent from exposure metadata: "
            f"{unknown_schema[:10]}"
        )
    missing_schema = [name for name in declared if name not in factor_order]
    if missing_schema:
        raise ValueError(
            f"tuda2 factor_order is missing declared factors {missing_schema[:10]}"
        )
    missing_style = [name for name in physical_style if name not in exposure.columns]
    if missing_style:
        raise ValueError(
            f"tuda2 exposure is missing style factors {missing_style[:10]}"
        )
    style = exposure.loc[:, list(physical_style)].astype(float, copy=False)
    if industry_factors:
        if "industry" in exposure.columns:
            labels = exposure["industry"]
            unknown = sorted(
                set(labels.dropna().astype(str).unique()) - set(industry_factors)
            )
            if unknown:
                raise ValueError(
                    f"tuda2 exposure contains unknown industries {unknown[:10]}"
                )
            industry = pd.get_dummies(labels, dtype=float).reindex(
                columns=list(industry_factors), fill_value=0.0
            )
        elif all(name in exposure.columns for name in industry_factors):
            industry = exposure.loc[:, list(industry_factors)].astype(float, copy=False)
        else:
            raise ValueError(
                "tuda2 exposure has neither industry labels nor dummy factors"
            )
        result = pd.concat([style, industry], axis=1, copy=False)
    else:
        result = style

    virtual_constants: dict[str, float] = {}
    for name, expected in constants.items():
        if name in exposure.columns:
            values = pd.to_numeric(exposure[name], errors="coerce").astype(float)
            if not np.allclose(
                values.to_numpy(copy=False), expected, rtol=0.0, atol=1e-12
            ):
                raise ValueError(
                    f"tuda2 constant exposure {name!r} must equal {expected}"
                )
            result = pd.concat([result, values.rename(name)], axis=1, copy=False)
        else:
            virtual_constants[name] = expected

    physical_order = [name for name in factor_order if name in result.columns]
    if list(result.columns) != physical_order:
        result = result.loc[:, physical_order]
    values = result.to_numpy(copy=False)
    if not np.all(np.isfinite(values)):
        raise ValueError("tuda2 exposure contains NaN or infinity after expansion")
    return result, virtual_constants


def _validate_covariance_schema(
    covariance: pd.DataFrame,
    factor_order: tuple[str, ...],
) -> None:
    """验证批量协方差坐标服从 tuda2 风险模型 schema。

    DataYes 的协方差列是全历史因子并集；2019-12-03 行业分类变更前后，每个日期的
    有效因子集合并不相同，权威坐标是 ``(dt, factor)`` 行索引。这里只验证行因子
    属于 schema 且存在同名列，不把批量 columns 错当成每日因子集合。
    """

    if not isinstance(covariance, pd.DataFrame):
        raise TypeError("tuda2 covariance must be a pandas DataFrame")
    if not isinstance(covariance.index, pd.MultiIndex) or tuple(
        covariance.index.names
    ) != ("dt", "factor"):
        raise ValueError("tuda2 covariance must use a (dt, factor) MultiIndex")
    columns = tuple(str(name) for name in covariance.columns)
    if len(columns) != len(set(columns)):
        raise ValueError("tuda2 covariance contains duplicate factor columns")
    row_values = tuple(
        str(name) for name in covariance.index.get_level_values("factor")
    )
    if len(row_values) == 0:
        raise ValueError("tuda2 covariance contains no row factors")
    row_factors = set(row_values)
    schema_factors = set(factor_order)
    unknown = sorted(row_factors - schema_factors)
    if unknown:
        raise ValueError(
            f"tuda2 covariance contains factors absent from schema: {unknown[:10]}"
        )
    missing = sorted(row_factors - set(columns))
    if missing:
        raise ValueError(
            f"tuda2 covariance columns do not cover row factors: {missing[:10]}"
        )


def _single_date_schedule(
    date: pd.Timestamp, universe: pd.DataFrame
) -> PortfolioSchedule:
    if not isinstance(universe, pd.DataFrame):
        raise TypeError("universe must be a pandas DataFrame")
    frame = universe.copy()
    if isinstance(frame.index, pd.MultiIndex):
        if tuple(frame.index.names) != ("dt", "sid"):
            raise ValueError("universe MultiIndex names must be exactly ('dt', 'sid')")
        dates = pd.DatetimeIndex(frame.index.get_level_values("dt").unique())
        if len(dates) != 1 or pd.Timestamp(dates[0]) != date:
            raise ValueError(
                "single-period universe must contain exactly the requested date"
            )
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
