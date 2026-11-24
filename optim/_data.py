"""2.x 入口与 tuda2 批量数据接口之间的适配。"""

import numpy as np
import pandas as pd
import tuda2
from carry2 import reindex


def _panel(frame, coordinate, label, dates=None):
    """检查严格同日数据，输入坐标已有序时直接复用。"""
    index = frame.index
    if not isinstance(index, pd.MultiIndex) or tuple(index.names) != ("dt", coordinate):
        raise ValueError(f"{label} 必须使用 (dt, {coordinate}) 索引")
    if index.has_duplicates:
        raise ValueError(f"{label} 存在重复坐标")
    if dates is not None:
        missing = dates.difference(index.unique(level="dt"))
        if len(missing):
            raise ValueError(f"{label} 缺少日期 {missing[:5].strftime('%Y-%m-%d').tolist()}")
    return frame if index.is_monotonic_increasing else frame.sort_index()


def benchmark_frame(value, dates):
    """指数代码由 tuda2 一次读取，手工基准沿用同一坐标和权重口径。"""
    if isinstance(value, str):
        value = tuda2.get_index_weight(value, dts=dates, type="daily")
    if isinstance(value, pd.Series):
        value = value.to_frame("weight")
    value = _panel(value, "sid", "基准", dates)
    return value[["weight"]].astype(float)


def risk_frame(kind, dates, model, *, factor_type=None):
    """各表按调仓日批量读取，tuda2 返回的风险量已采用年化小数。"""
    options = {} if factor_type is None else {"factor_type": factor_type}
    frame = tuda2.get_risk_model(kind, dts=dates, model=model, **options)
    if kind == "cov":
        # 协方差行顺序本身是当日有效因子顺序，排序仅适用于股票表。
        index = frame.index
        if not isinstance(index, pd.MultiIndex) or tuple(index.names) != ("dt", "factor"):
            raise ValueError("因子协方差必须使用 (dt, factor) 索引")
        if index.has_duplicates or frame.columns.has_duplicates:
            raise ValueError("因子协方差存在重复坐标")
        missing = dates.difference(index.unique(level="dt"))
        if len(missing):
            raise ValueError(f"因子协方差缺少日期 {missing[:5].strftime('%Y-%m-%d').tolist()}")
        return frame
    return _panel(frame, "sid", f"{model} {kind}", dates)


def daily_returns(dates, supplied=None, sids=None):
    """按区间一次读取 C2C 日收益。停牌或缺失个股沿用 2.x 的零收益估值。"""
    frame = supplied
    if frame is None:
        frame = tuda2.get_return(
            since=dates[0], until=dates[-1], sids=sids, freq="D", window=1,
            shift=False, market_side="close", price_type="vwap", price_window=0,
        )
    if isinstance(frame, pd.DataFrame):
        frame = frame["m0"] if "m0" in frame else frame.iloc[:, 0]
    frame = _panel(frame, "sid", "日收益", dates)
    return frame.loc[dates[0]:dates[-1]].astype(float).fillna(0.0)


def covariance_day(covariance, date):
    """按当日行因子选同名列，支持 DataYes 历史行业切换。"""
    block = covariance.xs(date, level="dt")
    missing = block.index.difference(block.columns)
    if len(missing):
        raise ValueError(f"{date.date()} 协方差缺少因子列 {missing.tolist()[:10]}")
    values = block.loc[:, block.index].to_numpy(float)
    if not np.isfinite(values).all() or not np.allclose(values, values.T, rtol=0.0, atol=1e-10):
        raise ValueError(f"{date.date()} 因子协方差含非有限值或不对称")
    return block.index, (values + values.T) * 0.5


def quadratic_exposure(exposure, schema):
    """展开分类行业，并补入 schema 声明的不落盘常量敞口。"""
    industries = schema["industry"]
    labels = exposure["industry"]
    unknown = pd.Index(labels.dropna().unique()).difference(industries)
    if len(unknown):
        raise ValueError(f"风险模型含未知行业 {unknown.tolist()[:10]}")
    dummies = pd.get_dummies(labels, dtype=float).reindex(columns=industries, fill_value=0.0)
    result = pd.concat([exposure.drop(columns="industry"), dummies, labels], axis=1)
    constants = schema["constant_exposures"]
    for name, value in constants.items():
        if name in result and not np.allclose(result[name], value, rtol=0.0, atol=1e-12):
            raise ValueError(f"常量敞口 {name!r} 与风险模型 schema 不一致")
    return result.assign(**constants)
