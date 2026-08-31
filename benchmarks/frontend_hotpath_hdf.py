#!/usr/bin/env python3
"""Benchmark post-alignment portfolio front-end work on a cropped HDF5 panel.

This benchmark intentionally does not compare pandas/carry/carry2 alignment.  It
starts from one already coordinated ``(dt, sid)`` price panel and measures daily
materialization, derived-array computation, validation-style scans, hashing and
weight-result construction.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import tables


COLUMNS = ("close", "amount", "volume", "pre_close")


@dataclass(frozen=True)
class Timing:
    case: str
    repeats: int
    median_s: float
    minimum_s: float
    maximum_s: float
    per_day_median_ms: float


@dataclass(frozen=True)
class CroppedPanel:
    frame: pd.DataFrame
    dates: pd.DatetimeIndex
    offsets: np.ndarray
    sid_values: np.ndarray


def _decode(values: np.ndarray) -> list[str]:
    return [value.decode() if isinstance(value, bytes) else str(value) for value in values]


def load_px_tail(path: Path, dates_count: int) -> CroppedPanel:
    """Load only the last ``dates_count`` date rows from the fixed-format /px key."""

    with tables.open_file(path, mode="r") as handle:
        group = handle.get_node("/px")
        date_levels = pd.DatetimeIndex(pd.to_datetime(group.axis1_level0[:]))
        if dates_count <= 0 or dates_count > len(date_levels):
            raise ValueError(f"dates must be within [1, {len(date_levels)}]")
        wanted_codes = np.arange(len(date_levels) - dates_count, len(date_levels))

        date_codes_all = group.axis1_label0[:]
        first = int(np.searchsorted(date_codes_all, wanted_codes[0], side="left"))
        stop = int(np.searchsorted(date_codes_all, wanted_codes[-1], side="right"))
        date_codes = date_codes_all[first:stop]
        if not np.all(np.isin(date_codes, wanted_codes)):
            raise ValueError("selected /px date rows are not contiguous")

        sid_codes = group.axis1_label1[first:stop]
        sid_levels = np.asarray(_decode(group.axis1_level1[:]), dtype=object)
        all_columns = _decode(group.axis0[:])
        positions = np.asarray([all_columns.index(column) for column in COLUMNS], dtype=int)
        values = np.asarray(group.block0_values[first:stop, positions], dtype=np.float64)

    dates = date_levels.take(wanted_codes)
    index = pd.MultiIndex.from_arrays(
        [date_levels.take(date_codes), sid_levels.take(sid_codes)],
        names=["dt", "sid"],
    )
    frame = pd.DataFrame(values, index=index, columns=COLUMNS, copy=False)
    if not frame.index.is_monotonic_increasing or not frame.index.is_unique:
        raise ValueError("cropped /px index must be sorted and unique")

    changes = np.flatnonzero(np.diff(date_codes)) + 1
    offsets = np.concatenate(([0], changes, [len(frame)])).astype(np.int64, copy=False)
    if len(offsets) != len(dates) + 1:
        raise ValueError("date offset count does not match selected dates")
    return CroppedPanel(
        frame=frame,
        dates=dates,
        offsets=offsets,
        sid_values=sid_levels.take(sid_codes),
    )


def _derive(values: np.ndarray) -> tuple[float, int, int]:
    close = values[:, 0]
    amount = values[:, 1]
    volume = values[:, 2]
    pre_close = values[:, 3]
    finite = np.isfinite(values)
    valid_return = np.isfinite(close) & np.isfinite(pre_close) & (pre_close > 0.0)
    returns = np.zeros(len(values), dtype=np.float64)
    np.divide(close, pre_close, out=returns, where=valid_return)
    returns[valid_return] -= 1.0
    tradable = valid_return & (amount > 0.0) & (volume > 0.0)
    return float(returns[valid_return].sum()), int(tradable.sum()), int(finite.sum())


def pandas_daily(panel: CroppedPanel) -> tuple[float, int, int]:
    checksum = 0.0
    tradable_count = 0
    finite_count = 0
    for date in panel.dates:
        values = panel.frame.xs(date, level="dt").to_numpy(dtype=np.float64, copy=False)
        part = _derive(values)
        checksum += part[0]
        tradable_count += part[1]
        finite_count += part[2]
    return checksum, tradable_count, finite_count


def numpy_daily(panel: CroppedPanel) -> tuple[float, int, int]:
    values = panel.frame.to_numpy(dtype=np.float64, copy=False)
    checksum = 0.0
    tradable_count = 0
    finite_count = 0
    for start, stop in zip(panel.offsets[:-1], panel.offsets[1:]):
        part = _derive(values[start:stop])
        checksum += part[0]
        tradable_count += part[1]
        finite_count += part[2]
    return checksum, tradable_count, finite_count


def numpy_batch(panel: CroppedPanel) -> tuple[float, int, int]:
    return _derive(panel.frame.to_numpy(dtype=np.float64, copy=False))


def prepare_returns(panel: CroppedPanel) -> np.ndarray:
    values = panel.frame.to_numpy(dtype=np.float64, copy=False)
    close = values[:, 0]
    pre_close = values[:, 3]
    valid = np.isfinite(close) & np.isfinite(pre_close) & (pre_close > 0.0)
    returns = np.zeros(len(values), dtype=np.float64)
    np.divide(close, pre_close, out=returns, where=valid)
    returns[valid] -= 1.0
    return returns


def prepared_daily_views(panel: CroppedPanel, returns: np.ndarray) -> tuple[int, int]:
    """Materialize only ndarray views after static derived arrays are prepared."""

    values = panel.frame.to_numpy(dtype=np.float64, copy=False)
    rows = 0
    view_count = 0
    for start, stop in zip(panel.offsets[:-1], panel.offsets[1:]):
        daily_values = values[start:stop]
        daily_returns = returns[start:stop]
        rows += len(daily_values) + len(daily_returns)
        view_count += 2
    return rows, view_count


def daily_content_hash(panel: CroppedPanel) -> str:
    values = panel.frame.to_numpy(dtype=np.float64, copy=False)
    combined = hashlib.blake2b(digest_size=16)
    for start, stop in zip(panel.offsets[:-1], panel.offsets[1:]):
        daily = values[start:stop]
        digest = hashlib.blake2b(memoryview(daily).cast("B"), digest_size=16).digest()
        combined.update(digest)
    return combined.hexdigest()


def build_weight_inputs(panel: CroppedPanel, top_n: int = 500) -> list[np.ndarray]:
    values = panel.frame.to_numpy(dtype=np.float64, copy=False)
    weights: list[np.ndarray] = []
    for start, stop in zip(panel.offsets[:-1], panel.offsets[1:]):
        amount = np.nan_to_num(values[start:stop, 1], nan=-np.inf)
        count = min(top_n, len(amount))
        chosen = np.argpartition(amount, -count)[-count:]
        weight = np.zeros(len(amount), dtype=np.float64)
        weight[chosen] = 1.0 / count
        weights.append(weight)
    return weights


def dense_weight_series(panel: CroppedPanel, weights: list[np.ndarray]) -> int:
    total = 0
    for i, weight in enumerate(weights):
        start, stop = panel.offsets[i : i + 2]
        result = pd.Series(weight, index=panel.sid_values[start:stop], copy=False)
        total += len(result)
    return total


def sparse_weight_series(panel: CroppedPanel, weights: list[np.ndarray]) -> int:
    total = 0
    for i, weight in enumerate(weights):
        start, stop = panel.offsets[i : i + 2]
        keep = np.flatnonzero(np.abs(weight) >= 1e-5)
        result = pd.Series(
            weight[keep],
            index=panel.sid_values[start:stop][keep],
            copy=False,
        )
        total += len(result)
    return total


def measure(case: str, function: Callable[[], object], repeats: int, days: int) -> Timing:
    function()
    samples: list[float] = []
    for _ in range(repeats):
        gc.collect()
        started = time.perf_counter()
        function()
        samples.append(time.perf_counter() - started)
    median = statistics.median(samples)
    return Timing(
        case=case,
        repeats=repeats,
        median_s=median,
        minimum_s=min(samples),
        maximum_s=max(samples),
        per_day_median_ms=1000.0 * median / days,
    )


def run(path: Path, dates_count: int, repeats: int) -> dict[str, object]:
    load_started = time.perf_counter()
    panel = load_px_tail(path, dates_count)
    cropped_load_s = time.perf_counter() - load_started

    oracle = pandas_daily(panel)
    np.testing.assert_allclose(numpy_daily(panel)[0], oracle[0], rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(numpy_batch(panel)[0], oracle[0], rtol=0.0, atol=1e-12)
    assert numpy_daily(panel)[1:] == oracle[1:]
    assert numpy_batch(panel)[1:] == oracle[1:]

    weights = build_weight_inputs(panel)
    prepared_returns = prepare_returns(panel)
    timings = [
        measure("pandas_xs_daily_derive", lambda: pandas_daily(panel), repeats, dates_count),
        measure("numpy_offset_daily_derive", lambda: numpy_daily(panel), repeats, dates_count),
        measure("numpy_batch_derive", lambda: numpy_batch(panel), repeats, dates_count),
        measure(
            "prepared_daily_views",
            lambda: prepared_daily_views(panel, prepared_returns),
            repeats,
            dates_count,
        ),
        measure("daily_content_hash", lambda: daily_content_hash(panel), repeats, dates_count),
        measure(
            "dense_weight_series",
            lambda: dense_weight_series(panel, weights),
            repeats,
            dates_count,
        ),
        measure(
            "sparse_weight_series",
            lambda: sparse_weight_series(panel, weights),
            repeats,
            dates_count,
        ),
    ]

    values = panel.frame.to_numpy(dtype=np.float64, copy=False)
    return {
        "path": str(path),
        "selected_dates": dates_count,
        "date_min": str(panel.dates.min().date()),
        "date_max": str(panel.dates.max().date()),
        "rows": len(panel.frame),
        "columns": list(panel.frame.columns),
        "cropped_mib": panel.frame.memory_usage(index=True, deep=True).sum() / (1024**2),
        "cropped_load_s": cropped_load_s,
        "values_c_contiguous": bool(values.flags.c_contiguous),
        "values_f_contiguous": bool(values.flags.f_contiguous),
        "daily_rows_min": int(np.diff(panel.offsets).min()),
        "daily_rows_median": float(np.median(np.diff(panel.offsets))),
        "daily_rows_max": int(np.diff(panel.offsets).max()),
        "dense_weight_rows": int(sum(map(len, weights))),
        "sparse_weight_rows": int(sum(np.count_nonzero(weight) for weight in weights)),
        "timings": [asdict(timing) for timing in timings],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("hdf", type=Path)
    parser.add_argument("--dates", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    print(json.dumps(run(args.hdf, args.dates, args.repeats), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
