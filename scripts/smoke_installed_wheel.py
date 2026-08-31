#!/usr/bin/env python3
"""在脱离源码目录的环境中验证已安装 optim wheel 的三条主路径。"""

from __future__ import annotations

from dataclasses import replace
from importlib import resources
from importlib.metadata import version
from pathlib import Path
import tomllib

import numpy as np
import pandas as pd

import optim
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
    RiskAdjustedAlpha,
    SymmetricBound,
    TrackingErrorLimit,
    TurnoverLimit,
    WeightBounds,
)
from optim._core import engine


def _problem() -> PortfolioProblem:
    """构造同时适用于 LP、QP 和 Factor-QCQP 的小型确定性问题。"""

    assets = pd.Index(["a", "b", "c", "d"], name="sid")
    risk = FactorRiskModel(
        asof=pd.Timestamp("2026-01-02"),
        exposure=np.array(
            [
                [1.0, 1.0],
                [-1.0, 1.0],
                [0.5, 0.0],
                [-0.5, 0.0],
            ]
        ),
        covariance=np.diag([0.04, 0.02]),
        specific_volatility=np.full(4, 0.10),
        factor_names=("size", "industry_a"),
        factor_types=("style", "industry"),
    )
    data = PortfolioData(
        date=pd.Timestamp("2026-01-02"),
        assets=assets,
        alpha=np.array([1.0, 0.5, -0.2, 0.1]),
        benchmark=np.full(4, 0.25),
        initial_weight=np.full(4, 0.25),
        tradable=np.ones(4, dtype=bool),
        risk_model=risk,
        alpha_spec=AlphaSpec(),
    )
    constraints = PortfolioConstraints(
        asset_weight=WeightBounds(0.0, 0.60),
        active_weight=SymmetricBound(0.30),
        total_active=1.0,
        turnover=TurnoverLimit(0.50),
        benchmark_member_weight=LowerBound(0.80),
        style=ExposureBounds(
            default=(-0.60, 0.60),
            overrides={"size": (-0.30, 0.30)},
        ),
        industry=ExposureBounds(default=(-0.50, 0.50)),
    )
    return PortfolioProblem(data, MaximizeAlpha(), constraints)


def main() -> None:
    """验证版本、资源、扩展模块和三类优化结果。"""

    distribution_version = version("optim")
    if optim.__version__ != distribution_version:
        raise RuntimeError(
            "package and distribution versions differ: "
            f"package={optim.__version__}, distribution={distribution_version}"
        )
    engine_path = Path(engine.__file__)
    if engine_path.suffix not in {".so", ".pyd"}:
        raise RuntimeError(f"core engine is not a binary extension: {engine_path}")
    package_files = resources.files("optim")
    for relative in (
        "LIBRARY.toml",
        "references/api_overview.md",
        "references/gotchas.md",
        "references/recipes.md",
    ):
        if not package_files.joinpath(relative).is_file():
            raise RuntimeError(f"installed wheel misses resource: {relative}")
    with package_files.joinpath("LIBRARY.toml").open("rb") as stream:
        catalog_version = tomllib.load(stream)["meta"]["version"]
    if catalog_version != distribution_version:
        raise RuntimeError(
            "catalog and distribution versions differ: "
            f"catalog={catalog_version}, distribution={distribution_version}"
        )

    base = _problem()
    problems = {
        "lp": base,
        "qp": replace(base, objective=RiskAdjustedAlpha()),
        "factor_qcqp": replace(
            base,
            constraints=replace(
                base.constraints,
                tracking_error=TrackingErrorLimit(0.20),
            ),
        ),
    }
    routes = {}
    for name, problem in problems.items():
        result = PortfolioOptimizer().solve(problem)
        weight = result.require_weights()
        if not np.isclose(weight.sum(), 1.0, atol=1e-6):
            raise RuntimeError(f"{name} weights do not sum to one")
        routes[name] = [attempt.backend for attempt in result.route]

    print(
        {
            "version": optim.__version__,
            "package": optim.__file__,
            "engine": str(engine_path),
            "routes": routes,
        }
    )


if __name__ == "__main__":
    main()
