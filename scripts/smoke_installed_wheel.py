#!/usr/bin/env python3
"""在脱离源码目录的环境中验证已安装 optim wheel 的三条主路径。"""

from __future__ import annotations

from dataclasses import replace
from importlib import resources
from importlib.metadata import version
from pathlib import Path
from importlib import import_module
import tempfile
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
    for name in (
        "compiler",
        "solver_adapter",
        "diagnostic_engine",
        "asset_bounds",
        "solution",
    ):
        module = import_module(f"optim._impl.{name}")
        if Path(module.__file__).suffix not in {".so", ".pyd"}:
            raise RuntimeError(
                f"impl module is not a binary extension: {module.__file__}"
            )
    compiler_module = import_module("optim._impl.compiler")
    if (
        compiler_module.__doc__
        or compiler_module._DomainBuilder._add_sparse_total_active.__doc__
    ):
        raise RuntimeError(
            "compiled implementation exposes internal algorithm docstrings"
        )
    if not PortfolioConstraints.__doc__ or not PortfolioOptimizer.optimize.__doc__:
        raise RuntimeError("public API documentation must remain available")
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

    # 检查编译后的业务实现支持诊断和序列，并保持复现文件只依赖公开数据契约。
    optimizer = PortfolioOptimizer()
    failed = optimizer.solve(
        base.with_constraints(turnover=TurnoverLimit(0.0)).with_data(
            initial_weight=np.array([1.0, 0.0, 0.0, 0.0])
        )
    )
    assert not failed.status.has_solution
    report = optimizer.diagnose(failed)
    assert report.linear_feasible is False
    with tempfile.TemporaryDirectory(prefix="optim-wheel-repro-") as folder:
        path = optim.export_repro(
            Path(folder) / "problem.zip", result=failed, report=report
        )
        assert not optim.load_repro(path).solve().status.has_solution
    next_problem = base.with_data(
        date=base.data.date + pd.Timedelta(days=1),
        risk_model=replace(
            base.data.risk_model, asof=base.data.date + pd.Timedelta(days=1)
        ),
    )
    sequence = optimizer.solve_sequence(
        [base, next_problem],
        holding_period_returns={
            next_problem.data.date: pd.Series(0.0, index=base.data.assets)
        },
    )
    assert sequence.stopped_date is None

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
