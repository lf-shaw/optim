"""面向 Notebook 与手工单期请求的带标签数据装配入口。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace

import numpy as np
import pandas as pd

from ..portfolio_types import (
    AlphaSpec,
    DataProvenance,
    FactorRiskModel,
    FullCovarianceRiskModel,
    PortfolioData,
    RiskModel,
)
from .alignment import BenchmarkCoveragePolicy, DataAlignmentError, align_benchmark
from ._reindex import _reindex_rows
from .contracts import FactorRiskFrames, _exact_covariance, _exact_xs, _single_date_values


def make_portfolio_data(
    *,
    date: str | pd.Timestamp,
    universe: pd.DataFrame,
    benchmark: pd.Series | None = None,
    initial_weight: pd.Series | None = None,
    risk_model: RiskModel | None = None,
    alpha_spec: AlphaSpec | None = None,
    alpha_column: str = "alpha",
    tradable_column: str = "tradable",
    extra_attribute_columns: Sequence[str] = (),
    benchmark_policy: BenchmarkCoveragePolicy | None = None,
    provenance: DataProvenance | None = None,
) -> PortfolioData:
    """从股票表和带标签权重构造单期数据，不取数、不求解。

    Parameters
    ----------
    date : str | pandas.Timestamp
        必填的信息日。普通单层索引表视为调用方声明的该日数据；若提供日期 MultiIndex，
        必须只含一个日期且等于 date，不裁剪多日数据。无日期标签的数据无法核实实际日期。
    universe : pandas.DataFrame
        股票表，非空且唯一的股票索引确定所有输出数组的顺序。单期表使用单层股票索引；
        也接受只含 date 当天的 (dt, sid) MultiIndex；不接受多个日期。alpha 列可省略，
        此时 data.alpha 为 None，由目标决定是否允许；tradable 列可省略，默认全为 True。
        被提取的列不允许缺失值；tradable 必须为 bool，不隐式转换字符串或数字。
    benchmark : pandas.Series | None
        非负、合计为 1 的基准。带标签的稀疏成分权重中未列出的样本股票填零；样本外基准
        依 benchmark_policy 处理。接受 sid 或只含 date 当天的 (dt, sid) 索引。
        None 表示未提供基准，不自动以期初持仓代替。
    initial_weight : pandas.Series | None
        交易前实际权重。带标签时允许只列持仓股票，其余样本股票填零；样本外非零持仓报错。
        接受 sid 或只含 date 当天的 (dt, sid) 索引。不归一化、不强制合计为 1，预算匹配
        由优化器校验。None 不会自动替换成基准。
    risk_model : FactorRiskModel | FullCovarianceRiskModel | None
        已构造、日期准确且按 universe 当日股票索引排列的位置风险对象。它不带股票标签，不能据此
        检测顺序是否放错。None 不附加风险模型。原始风险表可先交给 make_factor_risk_model，
        使用同一个 universe 的股票索引作为 assets。
    alpha_spec : AlphaSpec | None
        alpha 单位和尺度声明，原样保留；例如标准化得分使用 AlphaSpec(units="standardized_score")。
    alpha_column : str
        目标分数列名，默认 alpha；不存在则不提供 alpha，不自动推断其经济单位。
    tradable_column : str
        交易状态列名，默认 tradable；不存在则全为可交易。
    extra_attribute_columns : Sequence[str]
        显式选择的额外逐股数值属性列，必须存在且为有限实数；其余列忽略。默认不提取。
    benchmark_policy : BenchmarkCoveragePolicy | None
        样本外基准权重的处理策略；默认严格报错。仅显式选择 renormalize_within_tolerance
        才允许按该策略归一化。不能用于丢弃期初持仓。
    provenance : DataProvenance | None
        单期数据来源；默认记录 memory 和 date。若声明 source_date，则必须与 date 一致。
        基准覆盖处理的审计信息追加到 metadata。

    Returns
    -------
    PortfolioData
        已完成标签、日期、shape 和有限数值检查的不可变数据容器。目标依赖、预算、约束
        可行性及协方差半正定性仍由 optimizer.validate()/solve() 检查。

    Raises
    ------
    DataAlignmentError
        日期缺失、标签重复、必需数据不覆盖股票/因子、样本外持仓非零或数值/shape 无效。
    TypeError
        表格/风险对象类型不受支持，或交易状态不是 bool。
    ValueError
        未提供基准却指定了基准覆盖策略。

    Notes
    -----
    风险单位必须是年化小数，不执行隐式年化或百分数换算。同序且 dtype 匹配的数组可共享
    内存；如需隔离调用方后续的原地修改，请在传入之前复制。查看参数可在 Notebook 使用
    make_portfolio_data? 或在函数左括号后按 Shift+Tab。
    """
    day = pd.Timestamp(date)
    if pd.isna(day):
        raise DataAlignmentError("date must not be missing")
    if not isinstance(universe, pd.DataFrame):
        raise TypeError("universe must be a labeled DataFrame")
    universe = _single_date_values(universe, day, "universe")
    universe = _day_table(universe, day, "sid", "universe")
    _check_index(universe.columns, "universe.columns")
    assets = universe.index
    if not len(assets):
        raise DataAlignmentError("universe must not be empty")
    if isinstance(extra_attribute_columns, str):
        raise TypeError("extra_attribute_columns must be a sequence of column names")
    for name, value in (("benchmark", benchmark), ("initial_weight", initial_weight)):
        if value is not None and not isinstance(value, pd.Series):
            raise TypeError(f"{name} must be a labeled Series")
    if benchmark is not None:
        benchmark = _single_date_values(benchmark, day, "benchmark")
    if initial_weight is not None:
        initial_weight = _single_date_values(initial_weight, day, "initial_weight")
    origin = provenance or DataProvenance(source="memory", source_date=day)
    _check_source_date(origin, day, "provenance")
    metadata = dict(origin.metadata)

    alpha_values = (
        _vector(universe[alpha_column], assets, day, "alpha")
        if alpha_column in universe.columns
        else None
    )
    initial = None
    if initial_weight is not None:
        initial_day = _day_vector(initial_weight, day, "initial_weight")
        if isinstance(initial_day, pd.Series):
            original = _numeric(initial_day.to_numpy(), "initial_weight")
            omitted = original[~initial_day.index.isin(assets)]
            if np.any(omitted != 0.0):
                raise DataAlignmentError(
                    "initial_weight contains nonzero holdings outside assets"
                )
        initial = _vector(initial_day, assets, day, "initial_weight", fill_zero=True)
    bench = None
    if benchmark is not None:
        value = _day_vector(benchmark, day, "benchmark")
        _numeric(value.to_numpy(), "benchmark")
        aligned = align_benchmark(value, assets, benchmark_policy)
        bench = aligned.values
        metadata.update(
            benchmark_missing_mass=aligned.missing_mass,
            benchmark_missing_assets=aligned.missing_assets,
            benchmark_maximum_missing_weight=aligned.maximum_missing_weight,
            benchmark_renormalization_factor=aligned.renormalization_factor,
            benchmark_source_date=day,
        )
    elif benchmark_policy is not None:
        raise ValueError("benchmark_policy requires benchmark")
    tradable_values = (
        np.ones(len(assets), dtype=bool)
        if tradable_column not in universe.columns
        else _vector(universe[tradable_column], assets, day, "tradable", boolean=True)
    )
    missing_columns = set(extra_attribute_columns) - set(universe.columns)
    if missing_columns:
        raise DataAlignmentError(
            f"missing extra attribute columns: {sorted(missing_columns)}"
        )
    extra = {
        name: _vector(universe[name], assets, day, f"extra_attributes.{name}")
        for name in extra_attribute_columns
    }
    if risk_model is not None:
        risk_model = _check_risk(risk_model, day, len(assets))
    if alpha_values is not None:
        metadata["alpha_source_date"] = day
    return PortfolioData(
        date=day,
        assets=assets,
        alpha=alpha_values,
        benchmark=bench,
        initial_weight=initial,
        tradable=tradable_values,
        risk_model=risk_model,
        alpha_spec=alpha_spec,
        extra_attributes=extra,
        provenance=replace(origin, source_date=day, metadata=metadata),
    )


def make_factor_risk_model(
    *,
    date: str | pd.Timestamp,
    assets: pd.Index | Sequence[str],
    exposure: pd.DataFrame,
    factor_covariance: pd.DataFrame,
    specific_volatility: pd.Series | pd.DataFrame,
    factor_types: Mapping[str, str],
    constant_exposures: Mapping[str, float] | None = None,
    provenance: DataProvenance | None = None,
) -> FactorRiskModel:
    """将三张带标签风险表装配为指定股票顺序的单期因子风险对象。

    Parameters
    ----------
    date : str | pandas.Timestamp
        信息日。单层索引表由调用方声明为该日；日期 MultiIndex 必须包含准确同日记录。
    assets : pandas.Index | Sequence[str]
        非空、唯一的目标股票顺序，通常传 universe.index；多日 universe 应先取当天切片。
        暴露和特异风险必须覆盖所有目标股票，多余股票裁掉，不补缺失值。
    exposure : pandas.DataFrame
        股票 × 因子暴露表，通常无量纲；单层股票索引或 (dt, sid) 索引。
    factor_covariance : pandas.DataFrame
        年化小数协方差，行因子顺序为准，对齐同名列和暴露列；允许裁掉多余历史因子列。
        单层因子索引或 (dt, factor) 索引；不补缺失协方差、不取平均修复非对称输入。
    specific_volatility : pandas.Series | pandas.DataFrame
        年化小数特异波动率（标准差，不是方差，也不是百分数），股票索引或 (dt, sid)
        索引；DataFrame 必须仅含一列。不得为负。
    factor_types : Mapping[str, str]
        必须覆盖所有有效因子的类型声明，例如 {"SIZE": "style", "bank": "industry"}。
    constant_exposures : Mapping[str, float] | None
        显式声明未存储的常数暴露，例如 {"country": 1.0}。仅允许协方差中的因子，
        不得与暴露表已有列重叠；默认不猜测、不补常数列。
    provenance : DataProvenance | None
        风险来源，默认 memory 和 date；已声明的 source_date 必须等于 date。

    Returns
    -------
    FactorRiskModel
        通过日期、标签、shape 和有限数值检查的位置数组对象；因子顺序见 factor_names。
        协方差对称性和半正定性由后续优化器校验，本函数不进行昂贵的分解检查。

    Raises
    ------
    DataAlignmentError
        日期、股票或因子缺失/重复，或有效风险数据的 shape、数值不合法。
    TypeError
        输入不是声明的带标签表格类型。
    ValueError
        常数因子声明无效，或因子类型信息不完整。

    Notes
    -----
    不自动换算单位或年化。返回对象不携带股票标签，后续必须配合相同 assets 顺序使用。
    同序数据可以共享内存；如需隔离原地修改，请在调用前复制输入。
    """
    day = pd.Timestamp(date)
    if pd.isna(day):
        raise DataAlignmentError("date must not be missing")
    assets = pd.Index(assets, copy=False)
    _check_index(assets, "assets")
    if not len(assets):
        raise DataAlignmentError("assets must not be empty")
    if not isinstance(exposure, pd.DataFrame) or not isinstance(
        factor_covariance, pd.DataFrame
    ):
        raise TypeError("exposure and factor_covariance must be labeled DataFrames")
    if not isinstance(specific_volatility, (pd.Series, pd.DataFrame)):
        raise TypeError("specific_volatility must be a labeled Series or DataFrame")
    origin = provenance or DataProvenance(source="memory", source_date=day)
    _check_source_date(origin, day, "provenance")
    exp = _day_table(exposure, day, "sid", "exposure")
    cov = _day_table(factor_covariance, day, "factor", "factor_covariance")
    _check_index(exp.columns, "exposure.columns")
    _check_index(cov.columns, "factor_covariance.columns")
    spec = _day_vector(specific_volatility, day, "specific_volatility")
    # 复用统一风险表装配；提前拒绝复数，避免底层 float 转换丢弃虚部。
    for name, table in (
        ("exposure", exp),
        ("factor_covariance", cov),
        ("specific_volatility", spec),
    ):
        if np.iscomplexobj(table.to_numpy()):
            raise DataAlignmentError(f"{name} must be real-valued")
    constants = dict(constant_exposures or {})
    unknown = {str(name) for name in constants} - set(cov.index.astype(str))
    if unknown:
        raise DataAlignmentError(
            f"constant exposures are not covariance factors: {sorted(unknown)}"
        )
    frames = FactorRiskFrames(
        exposure=_with_date(exp, day, "sid"),
        covariance=_with_date(cov, day, "factor"),
        specific_volatility=_with_date(spec, day, "sid"),
        factor_types=factor_types,
        constant_exposures=constants,
        provenance=origin,
    )
    try:
        result = frames.materialize(day, assets)
    except DataAlignmentError:
        raise
    except (TypeError, ValueError) as exc:
        raise DataAlignmentError(f"invalid risk table values: {exc}") from exc
    # 保留准确的返回类型；与位置输入使用相同数值检查。
    checked = _check_risk(result, day, len(assets))
    assert isinstance(checked, FactorRiskModel)
    return checked


def _check_index(index: pd.Index, field: str) -> None:
    """单期坐标禁止多层、缺失、重复以及字符串化后冲突的标签。"""
    if isinstance(index, pd.MultiIndex) or index.hasnans or index.has_duplicates:
        raise DataAlignmentError(
            f"{field} must have unique, non-missing single-level labels"
        )
    if index.astype(str).has_duplicates:
        raise DataAlignmentError(f"{field} labels collide after string conversion")


def _day_table(value, day: pd.Timestamp, coordinate: str, field: str):
    """只选择准确同日切片；单层表由调用方声明其信息日。"""
    if isinstance(value.index, pd.MultiIndex):
        value = (
            _exact_covariance(value, day)
            if coordinate == "factor"
            else _exact_xs(value, day, field)
        )
    _check_index(value.index, field)
    return value


def _day_vector(value, day: pd.Timestamp, field: str):
    """统一单列 DataFrame/Series，未标记位置数组不添加推测标签。"""
    if isinstance(value, (pd.Series, pd.DataFrame)):
        value = _day_table(value, day, "sid", field)
        if isinstance(value, pd.DataFrame):
            if value.shape[1] != 1:
                raise DataAlignmentError(f"{field} must have exactly one value column")
            value = value.iloc[:, 0]
    return value


def _numeric(value, field: str) -> np.ndarray:
    """转换为有限实数，保留浮点数组的零复制路径。"""
    if np.iscomplexobj(value):
        raise DataAlignmentError(f"{field} must be real-valued")
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise DataAlignmentError(f"{field} must contain numeric values") from exc
    if not np.all(np.isfinite(result)):
        raise DataAlignmentError(f"{field} contains NaN or infinity")
    return result


def _vector(
    value,
    assets: pd.Index,
    day: pd.Timestamp,
    field: str,
    *,
    fill_zero: bool = False,
    boolean: bool = False,
) -> np.ndarray:
    """按标签对齐向量，只有明确的稀疏权重输入允许为缺失标签补零。"""
    value = _day_vector(value, day, field)
    if isinstance(value, pd.Series):
        if not value.index.equals(assets):
            if not fill_zero and len(assets.difference(value.index)):
                raise DataAlignmentError(f"{field} does not cover all assets")
            value = (
                _reindex_rows(value, assets, fill_value=0.0)
                if fill_zero
                else _reindex_rows(value, assets)
            )
        value = value.to_numpy()
    array = np.asarray(value)
    if array.shape != (len(assets),):
        raise DataAlignmentError(
            f"{field} expected shape {(len(assets),)}, observed {array.shape}"
        )
    if boolean:
        if array.dtype == np.dtype(bool):
            return array
        if not all(isinstance(item, (bool, np.bool_)) for item in array):
            raise TypeError(f"{field} must contain bool values without missing entries")
        return array.astype(bool, copy=False)
    return _numeric(array, field)


def _with_date(value, day: pd.Timestamp, coordinate: str):
    """借用既有风险表装配逻辑；只浅复制表头并附加单期日期坐标。"""
    result = value.copy(deep=False)
    result.index = pd.MultiIndex.from_arrays(
        [[day] * len(value), value.index], names=["dt", coordinate]
    )
    return result


def _check_source_date(value: DataProvenance, day: pd.Timestamp, field: str) -> None:
    if value.source_date is not None and pd.Timestamp(value.source_date) != day:
        raise DataAlignmentError(f"{field}.source_date must equal date")


def _check_risk(value: RiskModel, day: pd.Timestamp, n_assets: int) -> RiskModel:
    """检查位置风险输入的日期、单位、shape 和有限性，不推测其无标签坐标。"""
    if not isinstance(value, (FactorRiskModel, FullCovarianceRiskModel)):
        raise TypeError("risk_model must be a supported RiskModel object")
    if pd.Timestamp(value.asof) != day or value.annualization != "annualized_decimal":
        raise DataAlignmentError(
            "risk_model must use the exact date and annualized_decimal units"
        )
    _check_source_date(value.provenance, day, "risk_model.provenance")
    covariance = _numeric(value.covariance, "risk_model.covariance")
    size = len(value.factor_names) if isinstance(value, FactorRiskModel) else n_assets
    if covariance.shape != (size, size):
        raise DataAlignmentError("risk_model covariance has incorrect shape")
    if isinstance(value, FullCovarianceRiskModel):
        return replace(value, covariance=covariance)
    exposure = _numeric(value.exposure, "risk_model.exposure")
    specific = _numeric(value.specific_volatility, "risk_model.specific_volatility")
    if exposure.shape != (n_assets, size) or specific.shape != (n_assets,):
        raise DataAlignmentError("risk_model asset dimensions do not match assets")
    if np.any(specific < 0.0):
        raise DataAlignmentError("specific_volatility must be non-negative")
    _check_index(pd.Index(value.factor_names), "risk_model.factor_names")
    if len(value.factor_types) != size:
        raise DataAlignmentError("risk_model factor_types must match factor_names")
    return replace(
        value, exposure=exposure, covariance=covariance, specific_volatility=specific
    )
