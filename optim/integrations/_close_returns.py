"""把已读取的 (dt, sid) 日收益组装到调仓区间和有效股票坐标。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..data._date_slices import _DateSlices


def _ordered_rows(index: pd.MultiIndex) -> tuple[pd.DatetimeIndex, np.ndarray] | None:
    """用整数坐标确认日期块及块内股票严格有序，兼顾重复检查。"""
    date_codes, sid_codes = index.codes
    if (
        not len(index) or not isinstance(index.levels[0], pd.DatetimeIndex)
        or date_codes.min() < 0 or sid_codes.min() < 0
    ):
        return None
    boundaries = date_codes[1:] != date_codes[:-1]
    starts = np.r_[0, np.flatnonzero(boundaries) + 1, len(index)]
    days = index.levels[0].take(date_codes[starts[:-1]])
    if not days.is_monotonic_increasing:
        return None
    unordered = sid_codes[1:] <= sid_codes[:-1]
    unordered[boundaries] = False
    if unordered.any():
        return None
    return days, starts


def _intervals(days: pd.DatetimeIndex, requested: pd.DatetimeIndex):
    """确定每个 (上期日期, 当期日期] 区间的实际交易日位置。"""
    ends = days.get_indexer(requested[1:])
    starts = days.searchsorted(requested[:-1], side="right")
    if np.any(ends < starts):
        bad = requested[1:][ends < starts][0]
        raise ValueError(f"tuda2 daily close returns do not reach rebalance date {bad.date()}")
    return starts, ends


def _on_ordered_rows(index, observations, assets, requested, days, offsets):
    starts, ends = _intervals(days, requested)
    sid_codes = index.codes[1]
    # 股票标签只映射一次，后续各日直接使用整数坐标。
    positions = assets.get_indexer(index.levels[1])
    fully_covered = bool((positions >= 0).all())

    def gross_day(day):
        rows = slice(offsets[day], offsets[day + 1])
        daily = np.full(len(assets), np.nan)
        target = positions[sid_codes[rows]]
        values = observations[rows]
        if fully_covered:
            daily[target] = 1.0 + values
        else:
            keep = target >= 0
            daily[target[keep]] = 1.0 + values[keep]
        return daily

    returns = {}
    for date, start, end in zip(requested[1:], starts, ends):
        # 多日区间仅形成各自的小矩阵，单日直接切片。保留 NaN 的传播规则。
        gross = (
            gross_day(start) if start == end
            else np.prod([gross_day(day) for day in range(start, end + 1)], axis=0)
        )
        gross -= 1.0
        returns[date] = pd.Series(gross, index=assets, name="close_to_close_return")
    return returns


def _on_unordered_rows(values, assets, requested, day_slices):
    """对可对齐的无序输入保留按标签组装行为。"""
    days = day_slices._dates
    starts, ends = _intervals(days, requested)
    if np.array_equal(starts, ends):
        return {
            date: pd.Series(
                (1.0 + day_slices._day(date).reindex(assets).to_numpy(float, na_value=np.nan)) - 1.0,
                index=assets, name="close_to_close_return",
            )
            for date in requested[1:]
        }
    daily = values.unstack("sid").reindex(index=days, columns=assets).to_numpy(float, na_value=np.nan)
    return {
        date: pd.Series(
            np.prod(1.0 + daily[start:end + 1], axis=0) - 1.0,
            index=assets, name="close_to_close_return",
        )
        for date, start, end in zip(requested[1:], starts, ends)
    }


def assemble_close_returns(
    frame: pd.DataFrame,
    requested: pd.DatetimeIndex,
    sids: list[str] | None,
    *,
    filter_sids: bool = False,
) -> dict[pd.Timestamp, pd.Series]:
    """按有效股票及调仓区间组装收益，保留缺失观测并检查日期覆盖。"""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("tuda2.get_return must return a pandas DataFrame")
    index = frame.index
    if not isinstance(index, pd.MultiIndex) or tuple(index.names) != ("dt", "sid"):
        raise ValueError("tuda2 returns must use a (dt, sid) MultiIndex")
    if "m0" in frame.columns:
        column = "m0"
    elif frame.shape[1] == 1:
        column = frame.columns[0]
    else:
        raise ValueError("tuda2 daily close returns do not contain one unambiguous value column")
    values = frame[column]
    if filter_sids:
        # 提前读取的是风险裁剪前的股票并集。全区间被裁剪的股票也不参与收益的日期覆盖检查。
        keep = index.levels[1].isin(sids)
        if not keep.all():
            codes = index.codes[1]
            mask = (codes >= 0) & keep[codes]
            values = values.iloc[mask]
            index = values.index

    assets = (
        pd.Index(sids, name="sid") if sids is not None
        else index.unique(level="sid").sort_values()
    )
    ordered = _ordered_rows(index) if assets.is_unique else None
    if ordered is None:
        if index.has_duplicates:
            raise ValueError("tuda2 returns contain duplicate (dt, sid) rows")
        day_slices = _DateSlices(values, "sid", "tuda2 returns")

    observations = values.to_numpy(float, na_value=np.nan)
    invalid = np.isfinite(observations) & (observations < -1.0)
    if invalid.any():
        invalid_assets = index[invalid].get_level_values("sid")
        if sids is None or invalid_assets.isin(assets).any():
            raise ValueError("tuda2 daily return below -100% is invalid")
    if ordered is not None:
        days, offsets = ordered
        return _on_ordered_rows(index, observations, assets, requested, days, offsets)
    return _on_unordered_rows(values, assets, requested, day_slices)
