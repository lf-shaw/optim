from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from optim import (
    AlphaSpec,
    MaximizeAlpha,
    PortfolioConstraints,
    PortfolioData,
    PortfolioOptimizer,
    PortfolioProblem,
    SolveStatus,
    TurnoverLimit,
    WeightBounds,
)


def test_deep_diagnostic_reports_minimum_required_turnover():
    data = PortfolioData(
        date=pd.Timestamp("2026-01-02"),
        assets=pd.Index(["old", "new"], name="sid"),
        alpha=np.array([0.0, 1.0]),
        benchmark=np.array([0.0, 1.0]),
        initial_weight=np.array([1.0, 0.0]),
        tradable=np.ones(2, dtype=bool),
        alpha_spec=AlphaSpec(),
    )
    constraints = PortfolioConstraints(
        asset_weight=WeightBounds(
            lower=np.array([0.0, 1.0]),
            upper=np.array([0.0, 1.0]),
        ),
        turnover=TurnoverLimit(0.10),
    )
    problem = PortfolioProblem(data, MaximizeAlpha(), constraints)
    optimizer = PortfolioOptimizer()
    result = optimizer.solve(problem)
    assert result.status is SolveStatus.INFEASIBLE
    assert result.diagnostics is None

    report = optimizer.diagnose(problem, prior_result=result, level="deep")
    assert report.linear_feasible is False
    assert np.isclose(report.turnover_linear_lower_bound, 2.0, atol=1e-9)
    assert report.turnover_limit == 0.10
    assert "最小双边换手率" in report.summary_text
    assert report.relaxations


def test_failure_returns_result_and_require_weights_raises():
    data = PortfolioData(
        date=pd.Timestamp("2026-01-02"),
        assets=pd.Index(["a", "b"], name="sid"),
        alpha=np.array([1.0, 0.0]),
        benchmark=np.array([1.0, 0.0]),
        initial_weight=np.array([1.0, 0.0]),
        tradable=np.ones(2, dtype=bool),
        alpha_spec=AlphaSpec(),
    )
    problem = PortfolioProblem(
        data,
        MaximizeAlpha(),
        PortfolioConstraints(
            asset_weight=WeightBounds(lower=np.array([0.0, 0.5]), upper=1.0),
            total_active=0.1,
        ),
    )
    optimizer = PortfolioOptimizer()
    result = optimizer.solve(problem)
    assert result.status is SolveStatus.INFEASIBLE
    with pytest.raises(RuntimeError, match="did not produce usable weights"):
        result.require_weights()


def test_diagnostic_rejects_prior_result_for_different_problem(sample_lp_problem):
    optimizer = PortfolioOptimizer()
    result = optimizer.solve(sample_lp_problem)
    changed = replace(
        sample_lp_problem,
        data=replace(sample_lp_problem.data, alpha=sample_lp_problem.data.alpha + 0.1),
    )
    with pytest.raises(ValueError, match="fingerprint"):
        optimizer.diagnose(changed, prior_result=result)
