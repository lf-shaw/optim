#!/usr/bin/env python3
"""Reproducible canonical prepare benchmark without any external data source."""

from __future__ import annotations

import argparse
import json
import sys
import statistics
import time
from pathlib import Path
import numpy as np
import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from optim import (
    AlphaSpec,
    ExposureBounds,
    FactorRiskModel,
    LowerBound,
    MaximizeAlpha,
    PortfolioConstraints,
    PortfolioData,
    PortfolioOptimizer,
    PortfolioProblem,
    SymmetricBound,
    TrackingErrorLimit,
    TurnoverLimit,
    WeightBounds,
)


def make_problem(n_assets: int, n_factors: int, seed: int) -> PortfolioProblem:
    rng = np.random.default_rng(seed)
    assets = pd.Index([f"s{index:06d}" for index in range(n_assets)], name="sid")
    benchmark = rng.random(n_assets)
    benchmark /= benchmark.sum()
    initial = np.zeros(n_assets)
    support = np.argpartition(benchmark, -min(500, n_assets))[-min(500, n_assets) :]
    initial[support] = benchmark[support]
    initial /= initial.sum()
    n_style = min(10, n_factors)
    factor_names = tuple(f"factor_{index}" for index in range(n_factors))
    factor_types = tuple(
        "style" if index < n_style else "industry" for index in range(n_factors)
    )
    risk_model = FactorRiskModel(
        asof=pd.Timestamp("2026-01-02"),
        exposure=rng.normal(size=(n_assets, n_factors)),
        covariance=np.diag(rng.uniform(0.01, 0.05, n_factors)),
        specific_volatility=rng.uniform(0.12, 0.35, n_assets),
        factor_names=factor_names,
        factor_types=factor_types,
    )
    data = PortfolioData(
        date=pd.Timestamp("2026-01-02"),
        assets=assets,
        alpha=rng.normal(size=n_assets),
        benchmark=benchmark,
        initial_weight=initial,
        tradable=np.ones(n_assets, dtype=bool),
        risk_model=risk_model,
        alpha_spec=AlphaSpec(),
    )
    constraints = PortfolioConstraints(
        asset_weight=WeightBounds(0.0, 0.01),
        active_weight=SymmetricBound(0.01),
        total_active=1.8,
        turnover=TurnoverLimit(0.05),
        benchmark_member_weight=LowerBound(0.81),
        style=ExposureBounds(default=(-0.60, 0.60)),
        industry=ExposureBounds(default=(-0.05, 0.05)),
        tracking_error=TrackingErrorLimit(0.02),
    )
    return PortfolioProblem(data, MaximizeAlpha(), constraints)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", type=int, default=5200)
    parser.add_argument("--factors", type=int, default=47)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    problem = make_problem(args.assets, args.factors, args.seed)
    optimizer = PortfolioOptimizer()
    timings = []
    prepared = None
    for repeat in range(args.warmups + args.repeats):
        started = time.perf_counter()
        prepared = optimizer.prepare(problem)
        elapsed = time.perf_counter() - started
        prepared.validation.raise_for_errors()
        if repeat >= args.warmups:
            timings.append(elapsed)
    assert prepared is not None and prepared.fingerprint is not None
    print(
        json.dumps(
            {
                "assets": args.assets,
                "factors": args.factors,
                "repeats": args.repeats,
                "median_prepare_s": statistics.median(timings),
                "min_prepare_s": min(timings),
                "max_prepare_s": max(timings),
                "compiler_version": prepared.fingerprint.compiler_version,
                "compiler_optimizations": prepared.compiler_optimizations,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
