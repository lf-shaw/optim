"""批量表的日期位置索引；缓存限于所属数据对象，不使用全局缓存。"""

import numpy as np
import pandas as pd

from .alignment import DataAlignmentError


class _DateSlices:
    """一次验证坐标并记录日期位置，日循环不再扫描整个 MultiIndex。"""

    def __init__(self, frame, coordinate, field):
        index = frame.index
        if not isinstance(index, pd.MultiIndex) or tuple(index.names) != (
            "dt",
            coordinate,
        ):
            raise DataAlignmentError(
                f"{field} requires a (dt, {coordinate}) MultiIndex"
            )
        if index.has_duplicates:
            raise DataAlignmentError(f"{field} contains duplicate coordinates")
        if not isinstance(index.levels[0], pd.DatetimeIndex) or any(
            np.any(c < 0) for c in index.codes
        ):
            raise DataAlignmentError(
                f"{field} requires valid timestamp and {coordinate} labels"
            )
        self._frame = frame
        self._field = field
        self._positions = {}
        self._indexes = {}
        codes = index.codes[0]
        if len(codes):
            boundaries = np.r_[
                0, np.flatnonzero(codes[1:] != codes[:-1]) + 1, len(codes)
            ]
            chunks = codes[boundaries[:-1]]
            if len(np.unique(chunks)) == len(chunks):
                self._positions = {
                    index.levels[0][codes[start]]: slice(int(start), int(stop))
                    for start, stop in zip(boundaries[:-1], boundaries[1:])
                }
            else:
                # 手工输入可不按日期连续；一次建立位置数组，不重排其股票顺序。
                groups = (
                    pd.Series(np.arange(len(index))).groupby(codes, sort=False).indices
                )
                self._positions = {
                    index.levels[0][code]: positions
                    for code, positions in groups.items()
                }
        self._dates = pd.DatetimeIndex(list(self._positions)).sort_values()

    def _day(self, date):
        date = pd.Timestamp(date)
        if date not in self._positions:
            raise DataAlignmentError(
                f"{self._field} has no exact data for {date.date()}"
            )
        result = self._frame.iloc[self._positions[date]].copy(deep=False)
        if date not in self._indexes:
            self._indexes[date] = result.index.droplevel("dt")
        result.index = self._indexes[date]
        return result
