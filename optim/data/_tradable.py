"""交易状态输入的统一校验与布尔化。"""

import numpy as np


def _as_tradable(value, field: str = "tradable") -> np.ndarray:
    """接受布尔值或严格的实数 0/1；不将字符串、缺失值或其他数值解释为真值。

    常规布尔数组直接复用；数值数组批量检查。对象数组用于兼容 pandas 可空类型，
    逐项确认类型后转换，避免 pd.NA 的真值歧义及字符串的隐式转换。
    """
    array = np.asarray(value)
    if array.dtype.kind == "b":
        return array
    if array.dtype.kind in "iuf":
        valid = bool(np.all((array == 0) | (array == 1)))
    elif array.dtype.kind == "O":
        valid = all(
            isinstance(item, (bool, np.bool_, int, np.integer, float, np.floating))
            and item in (0, 1)
            for item in array.flat
        )
    else:
        valid = False
    if not valid:
        raise TypeError(f"{field} must contain bool or numeric 0/1 values without missing entries")
    return array.astype(bool, copy=False)
