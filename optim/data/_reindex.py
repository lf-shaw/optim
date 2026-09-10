"""可选的 carry2 行对齐适配，不改变 pandas 的标签和缺失值语义。"""

from functools import lru_cache
from importlib import import_module

import numpy as np
import pandas as pd


@lru_cache(maxsize=1)
def _carry_reindex():
    """只探测一次可选依赖；已安装但损坏的扩展不静默吞掉异常。"""
    try:
        return import_module("carry2").reindex
    except ModuleNotFoundError as exc:
        if exc.name != "carry2":
            raise
        return None


def _supported(value, index):
    """仅启用无需排序、包装和 dtype 转换即可保持 pandas 语义的输入。"""
    for axis in (value.index, index):
        if not isinstance(axis, pd.MultiIndex) or axis.nlevels != 2:
            return False
        if tuple(axis.names) != ("dt", "sid"):
            return False
        if not axis.is_unique:
            return False
        if any(np.any(codes < 0) for codes in axis.codes):
            return False
        if (
            not isinstance(axis.levels[0].dtype, np.dtype)
            or axis.levels[0].dtype.kind != "M"
        ):
            return False
        if axis.levels[1].inferred_type != "string":
            return False
    if value.index.levels[0].dtype != index.levels[0].dtype:
        return False
    dtypes = value.dtypes if isinstance(value, pd.DataFrame) else [value.dtype]
    if any(
        not isinstance(dtype, np.dtype) or dtype.kind not in "fbiu" for dtype in dtypes
    ):
        return False
    # carry2 用整数哨兵表示缺失，bool 也不能保留 pandas 的缺失表示；有新增行时不用它。
    if any(dtype.kind != "f" for dtype in dtypes) and not index.isin(value.index).all():
        return False
    return not isinstance(value, pd.DataFrame) or value.columns.is_unique


def _reindex_rows(value, index, *, fill_value=np.nan):
    """优先 carry2，未安装或输入不适用则原生对齐；不前填、不修改输入。"""
    if value.index.equals(index):
        result = value.copy(deep=False)
        result.index = index
        return result
    if not _supported(value, index):
        return value.reindex(index, fill_value=fill_value)
    reindex = _carry_reindex()
    if reindex is None:
        return value.reindex(index, fill_value=fill_value)
    if not value.index.is_monotonic_increasing or not index.is_monotonic_increasing:
        raise ValueError("carry2 alignment requires monotonic (dt, sid) indexes")
    result = reindex(value, index=index, ffill=False, copy=False)
    if isinstance(value, pd.Series):
        result = result.iloc[:, 0]
        result.name = value.name
    else:
        result.columns = value.columns
    # pandas fill_value 只填新增标签，不得把已有 NaN 一并填零。
    if not pd.isna(fill_value):
        missing = ~index.isin(value.index)
        if missing.any():
            result.iloc[missing] = fill_value
    return result
