"""只验证 optim 的对齐适配契约，不承担 carry2 算法性能测试。"""

import numpy as np
import pandas as pd
import pytest

from optim.data import _reindex as adapter


def _inputs():
    source = pd.MultiIndex.from_product(
        [pd.to_datetime(["2026-01-02", "2026-01-05"]), ["a", "c"]], names=["dt", "sid"]
    )
    target = pd.MultiIndex.from_product(
        [pd.to_datetime(["2026-01-02", "2026-01-05"]), ["a", "b", "c"]],
        names=["dt", "sid"],
    )
    return pd.Series([1.0, np.nan, 3.0, 4.0], index=source, name="alpha"), target


@pytest.mark.parametrize("frame", [False, True])
@pytest.mark.parametrize("fill", [np.nan, 0.0])
def test_installed_carry_matches_pandas(monkeypatch, frame, fill):
    carry = pytest.importorskip("carry2")
    value, target = _inputs()
    if frame:
        value = value.to_frame()
    calls = []

    def invoke(*args, **kwargs):
        calls.append(kwargs)
        return carry.reindex(*args, **kwargs)

    monkeypatch.setattr(adapter, "_carry_reindex", lambda: invoke)
    result = adapter._reindex_rows(value, target, fill_value=fill)
    expected = value.reindex(target, fill_value=fill)
    if frame:
        pd.testing.assert_frame_equal(result, expected)
    else:
        pd.testing.assert_series_equal(result, expected)
    assert len(calls) == 1 and calls[0]["ffill"] is False


def test_uninstalled_and_native_cases(monkeypatch):
    value, target = _inputs()
    monkeypatch.setattr(adapter, "_carry_reindex", lambda: None)
    pd.testing.assert_series_equal(
        adapter._reindex_rows(value, target), value.reindex(target)
    )

    def unexpected():
        pytest.fail("native-only input must not import carry2")

    monkeypatch.setattr(adapter, "_carry_reindex", unexpected)
    value = pd.Series([1.0, 2.0], index=["b", "a"])
    pd.testing.assert_series_equal(
        adapter._reindex_rows(value, pd.Index(["a", "b"])), value.reindex(["a", "b"])
    )
    value, target = _inputs()
    value = value.fillna(0).astype(bool)
    pd.testing.assert_series_equal(
        adapter._reindex_rows(value, target), value.reindex(target)
    )


def test_missing_module_only_falls_back(monkeypatch):
    adapter._carry_reindex.cache_clear()

    def absent(name):
        raise ModuleNotFoundError(name=name)

    monkeypatch.setattr(adapter, "import_module", absent)
    assert adapter._carry_reindex() is None
    adapter._carry_reindex.cache_clear()

    def broken(name):
        raise ModuleNotFoundError(name="carry2.libcarry2")

    monkeypatch.setattr(adapter, "import_module", broken)
    with pytest.raises(ModuleNotFoundError):
        adapter._carry_reindex()
    adapter._carry_reindex.cache_clear()


def test_nonmonotonic_and_native_errors_not_hidden(monkeypatch):
    value, target = _inputs()
    carry = pytest.importorskip("carry2")
    monkeypatch.setattr(adapter, "_carry_reindex", lambda: carry.reindex)
    with pytest.raises(ValueError, match="monotonic"):
        adapter._reindex_rows(value.iloc[::-1], target)

    def broken(*args, **kwargs):
        raise RuntimeError("native failure")

    monkeypatch.setattr(adapter, "_carry_reindex", lambda: broken)
    with pytest.raises(RuntimeError, match="native failure"):
        adapter._reindex_rows(value, target)
