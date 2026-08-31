from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from optim import (
    AlphaSpec,
    ExposureBounds,
    FactorRiskModel,
    LowerBound,
    MaximizeAlpha,
    PortfolioConstraints,
    PortfolioData,
    PortfolioProblem,
    SymmetricBound,
    TurnoverLimit,
    WeightBounds,
)


@pytest.fixture
def sample_data() -> PortfolioData:
    assets = pd.Index(["a", "b", "c", "d"], name="sid")
    risk_model = FactorRiskModel(
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
    return PortfolioData(
        date=pd.Timestamp("2026-01-02"),
        assets=assets,
        alpha=np.array([1.0, 0.5, -0.2, 0.1]),
        benchmark=np.full(4, 0.25),
        initial_weight=np.full(4, 0.25),
        tradable=np.ones(4, dtype=bool),
        risk_model=risk_model,
        alpha_spec=AlphaSpec(units="standardized_score", scale=1.0),
    )


@pytest.fixture
def sample_constraints() -> PortfolioConstraints:
    return PortfolioConstraints(
        asset_weight=WeightBounds(lower=0.0, upper=0.60),
        active_weight=SymmetricBound(0.30),
        total_active=1.0,
        turnover=TurnoverLimit(0.50),
        benchmark_member_weight=LowerBound(0.80),
        style=ExposureBounds(default=(-0.60, 0.60), overrides={"size": (-0.30, 0.30)}),
        industry=ExposureBounds(default=(-0.50, 0.50)),
    )


@pytest.fixture
def sample_lp_problem(sample_data, sample_constraints) -> PortfolioProblem:
    return PortfolioProblem(
        data=sample_data,
        objective=MaximizeAlpha(),
        constraints=sample_constraints,
    )
