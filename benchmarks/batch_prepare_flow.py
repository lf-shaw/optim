"""固定种子比较静态准备与再物化流程，不计 I/O 和求解，不测试 carry2 算法本身。

仓库根目录运行：PYTHONPATH=. python benchmarks/batch_prepare_flow.py --days 120 --assets 5200
"""

import argparse
from dataclasses import replace
from time import perf_counter
import json

import numpy as np
import pandas as pd

from optim import (
    AlphaSpec,
    FactorRiskFrames,
    InMemoryDataSource,
    MaximizeAlpha,
    PortfolioConstraints,
    PortfolioSchedule,
)


def main():
    """准备确定性数据并分别计时预检与后续逐日物化，核对数值摘要。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=120)
    parser.add_argument("--assets", type=int, default=5200)
    parser.add_argument("--factors", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if min(args.days, args.assets, args.factors, args.repeats) <= 0 or args.assets < 2:
        parser.error("dimensions and repeats must be positive, assets >= 2")
    rng = np.random.default_rng(20260910)
    dates = pd.bdate_range("2025-01-02", periods=args.days)
    assets = pd.Index([f"s{i:06d}" for i in range(args.assets)], name="sid")
    index = pd.MultiIndex.from_product([dates, assets], names=["dt", "sid"])
    factors = [f"factor{i}" for i in range(args.factors)]
    exposure = pd.DataFrame(
        rng.normal(size=(len(index), args.factors)), index=index, columns=factors
    )
    risk = FactorRiskFrames(
        exposure=exposure,
        covariance=pd.DataFrame(
            np.tile(np.eye(args.factors) * 0.001, (len(dates), 1)),
            index=pd.MultiIndex.from_product([dates, factors], names=["dt", "factor"]),
            columns=factors,
        ),
        specific_volatility=pd.Series(0.2, index=index),
        factor_types={name: "style" for name in factors},
    )
    target = pd.MultiIndex.from_product([dates, assets[::2]], names=["dt", "sid"])
    universe = pd.DataFrame({"alpha": rng.normal(size=len(target))}, index=target)
    benchmark = pd.Series(
        1.0, index=pd.MultiIndex.from_product([dates, [assets[0]]], names=["dt", "sid"])
    )
    options = dict(
        objective=MaximizeAlpha(),
        constraints=PortfolioConstraints(),
        alpha_spec=AlphaSpec(),
        initial_weight=pd.Series({assets[0]: 1.0}),
    )
    report = {
        "dimensions": vars(args),
        "seed": 20260910,
        "input_exposure_bytes": exposure.to_numpy().nbytes,
    }
    checksum = None
    for mode in (
        "daily_alignment_with_cached_dates",
        "batch_alignment_with_cached_dates",
    ):
        times = []
        for repeat in range(args.repeats + 1):
            source = InMemoryDataSource(risk_data=replace(risk), benchmark=benchmark)
            schedule = PortfolioSchedule(universe)
            if mode.startswith("daily"):
                # 对照仅关闭批量对齐；保留同一套日期缓存和校验，避免把不同校验成本混为一谈。
                source._batch_for_schedule = lambda schedule: source
            start = perf_counter()
            run = source.prepare_run(schedule, **options)
            run.validation.raise_for_errors()
            prepared = perf_counter()
            value = 0.0
            for date in run.dates:
                data = run.problem_at(date).data
                value += (
                    data.risk_model.exposure.sum()
                    + data.benchmark.sum()
                    + data.alpha.sum()
                )
            finished = perf_counter()
            if checksum is None:
                checksum = value
            np.testing.assert_allclose(value, checksum, rtol=1e-12, atol=1e-8)
            if repeat:
                times.append([prepared - start, finished - prepared, finished - start])
        report[mode] = dict(
            zip(
                ("prepare_s", "rematerialize_s", "total_s"),
                np.median(times, axis=0).tolist(),
            )
        )
    report["aligned_exposure_bytes"] = len(target) * args.factors * 8
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
