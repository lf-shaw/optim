"""编码优化必须兼容既有复现包，且不能把字符串枚举当成普通字符串。"""

import hashlib
from enum import Enum

import numpy as np
import pytest

from optim.fingerprint import _chunk, _feed


class Label(str, Enum):
    A = "alpha"


@pytest.mark.parametrize("value, expected", [
    ("000300.SH", "f52fc9ff7a9affdd1bd210241de7da2aa939fac5a6372df891d786f252762cf5"),
    (np.array(["甲", "000001.SZ", ""], dtype=object),
     "781e6c1b042ccae463e6510d08716b227fd6be8e67063269116913187a959fb3"),
    (Label.A, "3139d235b86de88d0d3455252a158cc7587afd0a36bd89ba4bc3ab80ed573916"),
    ({"a": ["x", 1, None], "b": np.array([.1, .2])},
     "aa2c200a54edfac055479cd6c90d00537eb94bd05a9d505791c3f472c2e00f1e"),
])
def test_legacy_hash_bytes(value, expected):
    """固定值来自优化前的编码器，不依赖工作区 Git 状态。"""
    digest = hashlib.sha256()
    _feed(digest, "test", value)
    assert digest.hexdigest() == expected


@pytest.mark.parametrize("size", [0, 1, 1024, 1025, 5200])
def test_batched_strings_equal_scalar_stream(size):
    """跨分块边界也必须与逐字符串的原有字节流一致。"""
    values = np.array([f"股票{i}" for i in range(size)], dtype=object)
    actual, expected = hashlib.sha256(), hashlib.sha256()
    _feed(actual, "assets", values)
    _chunk(expected, "assets.shape", np.asarray(values.shape, dtype="<i8").tobytes())
    for i, value in enumerate(values):
        _feed(expected, f"assets[{i}]", value)
    assert actual.digest() == expected.digest()
